"""Pre-cache validation: build a validator that scores a compiled function
by agreement with a ONE-TIME LLM labeling of the sample rows.

Why. temperature=1 compilation is high-variance: a draw may run fine yet
produce wrong outputs (e.g. nirvana Q8 map 1.0 -> 0.0 after a re-compile).
The fix is to score each draw on a few sample rows before trusting it and
keep only a draw that agrees with the reference. The reference here is the
LLM's own per-row judgment (the native semantics), computed ONCE on the
samples and cached — so retries are scored for free.

The validator is data-agnostic to sembaker.core's callers: we label the SAMPLE
rows the compiler already has, no external gold needed.
"""
from __future__ import annotations

import json
import os
from typing import Callable

from openai import OpenAI

from sembaker.core import cache as _cache


def _client(api_key=None):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (sembaker.core.validate needs the OpenAI API)")
    return OpenAI(api_key=key)


# Label calls are independent per sample, so issue them concurrently. This is
# the dominant one-time cost of the validation gate; parallelizing cuts it
# from K*latency (serial) to ~latency (one wave).
_LABEL_WORKERS = int(os.environ.get("CX_LABEL_WORKERS", "12"))


def _ask(cli, model_id, msg):
    r = cli.chat.completions.create(model=model_id, temperature=1,
                                    messages=[{"role": "user", "content": msg}])
    return r.choices[0].message.content.strip()


def _parallel_map(fn, items):
    """Run fn over items concurrently via the shared backend-agnostic scheduler,
    preserving order. (Same pool the compile path uses — all parallelism lives in
    sembaker.core.scheduler.) Label closures default on error, so scheduler's captured
    exceptions never reach the result."""
    from sembaker.core import scheduler

    return scheduler.run_map(fn, items, workers=_LABEL_WORKERS, label="validate-label")


def _label_filter(rows, intent, model_id, api_key=None):
    """LLM True/False per row for a filter predicate (concurrent)."""
    cli = _client(api_key)

    def one(row):
        try:
            msg = (f'Predicate: "{intent}"\nRow: {json.dumps(row, ensure_ascii=False, default=str)}\n'
                   'Answer strictly TRUE or FALSE: does the row satisfy the predicate?')
            return "true" in _ask(cli, model_id, msg).lower()[:6]
        except Exception:
            return False

    return _parallel_map(one, rows)


def _label_map(rows, intent, model_id, api_key=None):
    """LLM output value per row for a map instruction (concurrent)."""
    cli = _client(api_key)

    def one(row):
        try:
            msg = (f'Instruction: "{intent}"\nRow: {json.dumps(row, ensure_ascii=False, default=str)}\n'
                   'Output ONLY the resulting value, no commentary.')
            return _ask(cli, model_id, msg).lower()
        except Exception:
            return ""

    return _parallel_map(one, rows)


def _label_join(pairs, intent, model_id, api_key=None):
    """LLM True/False per (left,right) pair for a join condition (concurrent)."""
    cli = _client(api_key)

    def one(pair):
        try:
            lr, rr = pair
            msg = (f'Join condition: "{intent}"\n'
                   f'Left: {json.dumps(lr, ensure_ascii=False, default=str)}\n'
                   f'Right: {json.dumps(rr, ensure_ascii=False, default=str)}\n'
                   'Answer strictly TRUE or FALSE: should these two rows be joined?')
            return "true" in _ask(cli, model_id, msg).lower()[:6]
        except Exception:
            return False

    return _parallel_map(one, pairs)


def _norm(v):
    return "" if v is None else str(v).strip().lower()


def make_validator(kind, intent, model_id, *, samples=None,
                   samples_left=None, samples_right=None, api_key=None,
                   label_cache_key=None, max_label=12, max_pairs=20):
    """Return validator(compiled_fn) -> score in [0,1].

    Labels the samples ONCE (LLM), caches them, and scores any compiled fn
    by agreement on those labels. kind: filter | map | join."""
    # --- build (or load) the reference labels once ---
    labels = None
    if label_cache_key is not None:
        cached = _cache.get_text(label_cache_key)
        if cached is not None:
            try:
                labels = json.loads(cached)
            except Exception:
                labels = None

    if kind == "join":
        pairs = []
        L = (samples_left or [])[:6]
        R = (samples_right or [])[:6]
        for i, lr in enumerate(L):
            for j, rr in enumerate(R):
                pairs.append((lr, rr))
        pairs = pairs[:max_pairs]
        if labels is None:
            labels = _label_join(pairs, intent, model_id, api_key)
            if label_cache_key is not None:
                _cache.put_text(label_cache_key, json.dumps(labels))

        def validator(fn):
            if not pairs:
                return 1.0
            ok = 0
            for (lr, rr), gold in zip(pairs, labels):
                try:
                    pred = bool(fn(lr, rr))
                except Exception:
                    pred = False
                ok += (pred == bool(gold))
            return ok / len(pairs)

        return validator

    rows = (samples or [])[:max_label]
    if kind == "filter":
        if labels is None:
            labels = _label_filter(rows, intent, model_id, api_key)
            if label_cache_key is not None:
                _cache.put_text(label_cache_key, json.dumps(labels))

        def validator(fn):
            if not rows:
                return 1.0
            ok = 0
            for row, gold in zip(rows, labels):
                try:
                    pred = bool(fn(row))
                except Exception:
                    pred = False
                ok += (pred == bool(gold))
            return ok / len(rows)

        return validator

    if kind == "map":
        if labels is None:
            labels = _label_map(rows, intent, model_id, api_key)
            if label_cache_key is not None:
                _cache.put_text(label_cache_key, json.dumps(labels))

        def validator(fn):
            if not rows:
                return 1.0
            ok = 0
            for row, gold in zip(rows, labels):
                try:
                    pred = _norm(fn(row))
                except Exception:
                    pred = ""
                g = _norm(gold)
                ok += (pred == g or (len(g) >= 2 and (g in pred or pred in g)))
            return ok / len(rows)

        return validator

    raise ValueError(f"unknown kind {kind!r}")
