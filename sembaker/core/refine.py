"""Predicate refinement: turn a fuzzy NL predicate into a concrete,
column-grounded one BEFORE compilation.

Motivation (the fuzzy-semantics problem). A predicate like "these two
reviews express opposite sentiments" leaves the compiler too much freedom:
it invents a text-sentiment heuristic that is both noisy (low quality) and
unstable across temperature=1 draws (high variance). But the data often
already carries a usable signal — e.g. a `scoreSentiment` column — that a
human would obviously key on. We add ONE LLM pass that, given the schema
and a few sample rows, rewrites the fuzzy intent into a concrete recipe
that names exact columns and rules. The downstream compile then has an
unambiguous target -> higher quality AND lower variance.

This is the active counterpart to the empirical finding that compile-time
sample visibility drives quality: instead of hoping the compiler discovers
the `scoreSentiment` column, refinement tells it to use it.

Cost: one extra LLM call per UNIQUE predicate, amortized by the same cache
as compilation (refined text is cached; execution stays local/zero-cost).
"""

from __future__ import annotations

import json
import os
import time

from openai import OpenAI

from sembaker.core.compiler import _strip_code_fences  # reuse fence stripper if needed  # noqa: F401


def _trunc(v, n=120):
    s = "" if v is None else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_samples(samples, k=8):
    return "\n".join(
        "  " + json.dumps({kk: _trunc(vv) for kk, vv in s.items()},
                          ensure_ascii=False, default=str)
        for s in samples[:k]
    )


_REFINE_SYS = (
    "You turn a fuzzy natural-language data predicate into a CONCRETE, "
    "deterministic recipe that a Python function can implement. You are shown "
    "the actual schema and sample rows. Prefer keying on explicit structured "
    "columns (labels, scores, ids, dates) when they directly encode the intent; "
    "fall back to free-text rules only when no structured column applies. "
    "Name exact column(s) and the exact rule. Keep it short. Do NOT write code."
)


def _refine_filter_or_map(kind: str, intent: str, samples: list[dict], model_id: str,
                          api_key: str | None = None) -> tuple[str, dict]:
    cols = list(samples[0].keys()) if samples else []
    user = (
        f"Operator: {kind}\n"
        f"Fuzzy intent: \"{intent}\"\n\n"
        f"Schema (columns): {', '.join(cols)}\n"
        f"Sample rows:\n{_fmt_samples(samples)}\n\n"
        "Rewrite the intent into a concrete recipe naming exact columns and the "
        "exact decision rule (thresholds, keyword sets, comparisons). If a "
        "structured column already encodes the intent, key on it. Output only the "
        "rewritten recipe text, one short paragraph."
    )
    return _call(user, model_id, api_key)


def _refine_join(condition: str, samples_left: list[dict], samples_right: list[dict],
                 model_id: str, api_key: str | None = None) -> tuple[str, dict]:
    cl = list(samples_left[0].keys()) if samples_left else []
    cr = list(samples_right[0].keys()) if samples_right else []
    user = (
        f"Operator: join\n"
        f"Fuzzy join condition: \"{condition}\"\n\n"
        f"Left schema: {', '.join(cl)}\n"
        f"Left sample rows:\n{_fmt_samples(samples_left)}\n\n"
        f"Right schema: {', '.join(cr)}\n"
        f"Right sample rows:\n{_fmt_samples(samples_right)}\n\n"
        "Rewrite the join condition into a concrete recipe over row_left and "
        "row_right naming exact columns and the exact comparison rule. If a "
        "structured column on each side already encodes the intent (e.g. a "
        "sentiment label, an id, a category), key on it. Output only the "
        "rewritten recipe text, one short paragraph."
    )
    return _call(user, model_id, api_key)


def _call(user: str, model_id: str, api_key: str | None) -> tuple[str, dict]:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (sembaker.core.refine needs the OpenAI API)")
    client = OpenAI(api_key=key)
    start = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": _REFINE_SYS},
                  {"role": "user", "content": user}],
        temperature=1,
    )
    text = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    stats = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
        "duration_secs": time.time() - start,
    }
    return text, stats


def refine_predicate(kind: str, intent: str, model_id: str, *,
                     samples: list[dict] | None = None,
                     samples_left: list[dict] | None = None,
                     samples_right: list[dict] | None = None,
                     api_key: str | None = None,
                     cache_key: str | None = None,
                     print_refined: bool = True) -> str:
    """Return a concrete, column-grounded predicate for `intent`.

    kind: "filter" | "map" | "join". For join pass samples_left/samples_right;
    otherwise pass samples. Cache-aware (refined text is a tiny artifact reused
    across runs and backends, exactly like compiled code)."""
    if cache_key is not None:
        from sembaker.core import cache as _cache

        hit = _cache.get_text(cache_key)
        if hit is not None:
            if print_refined:
                print(f"[refine:{kind}] cache HIT ({cache_key})")
            return hit

    if kind == "join":
        refined, stats = _refine_join(intent, samples_left or [], samples_right or [],
                                      model_id, api_key)
    else:
        refined, stats = _refine_filter_or_map(kind, intent, samples or [],
                                               model_id, api_key)

    if print_refined:
        print(f"[refine:{kind}] {stats['duration_secs']:.1f}s  in={stats['input_tokens']} "
              f"out={stats['output_tokens']}")
        try:
            print(f"  fuzzy:    {intent[:100]}")
            print(f"  concrete: {refined[:300]}")
        except UnicodeEncodeError:
            # Non-UTF-8 console (e.g. GBK on Chinese Windows) — degrade, never crash.
            safe = f"  fuzzy:    {intent[:100]}\n  concrete: {refined[:300]}"
            print(safe.encode("ascii", "backslashreplace").decode("ascii"))

    if cache_key is not None:
        from sembaker.core import cache as _cache

        _cache.put_text(cache_key, refined)

    return refined
