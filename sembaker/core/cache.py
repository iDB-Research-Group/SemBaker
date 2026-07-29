"""Persistent compile-artifact cache.

Key design: (op_kind, fn_name, predicate_text, sorted schema columns,
model_id) — deliberately EXCLUDES the sample rows embedded in the prompt,
so repeated queries hit the cache even though each run draws different
random samples. The cached payload is the generated CODE; on hit we exec
it locally (no LLM call, zero cost, microseconds).

This also acts as variance control: once a compile draw has been produced
(and, when the eager path is used, smoke-validated), every later run reuses
exactly that artifact instead of re-rolling temperature=1 generation.

Storage: one JSON file per key under CX_CACHE_DIR (default .cx_cache/ at
repo root; override with env CX_CACHE_DIR). Safe to delete anytime.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

_DEFAULT_DIR = Path(os.environ.get("CX_CACHE_DIR", ".cx_cache"))

# Canonicalization is for cross-backend cache sharing: the same semantic task
# is phrased with cosmetic differences by each backend's runner — PZ rebuilds a
# map instruction as "fieldname: <desc>", Lotus/PZ keep a "Review: {reviewText}"
# template line, capitalization/whitespace vary. Folding away ONLY this cosmetic
# noise lets identical tasks land on one key (so PZ/Lotus/Nirvana share the
# compiled artifact) while genuinely different wordings stay distinct. Set
# CX_NO_CANON=1 to disable (keys revert to verbatim predicate text).
_NO_CANON = os.environ.get("CX_NO_CANON") == "1"
_FIELD_PREFIX = re.compile(r"^[A-Za-z_]\w*:\s+")          # "sentiment: ..." (PZ map)
_TEMPLATE = re.compile(r"\{[^}]*\}")                       # {reviewText} placeholders
_REVIEW_LINE = re.compile(r"(?im)^\s*review\s*:\s*.*$")    # "Review: {reviewText}" line


def canon_predicate(predicate: str) -> str:
    """Fold cosmetic phrasing noise so equivalent tasks share a cache key.

    Conservative: only removes a leading "fieldname: " prefix, "{...}"
    placeholders, "Review: ..." template lines, and normalizes quotes / case /
    whitespace. Different sentences (e.g. "Return POSITIVE if..." vs "Classify
    the sentiment...") still differ — this never merges genuinely distinct
    instructions, it only strips formatting."""
    if _NO_CANON or not predicate:
        return predicate
    s = predicate
    s = _FIELD_PREFIX.sub("", s)
    s = _REVIEW_LINE.sub(" ", s)
    s = _TEMPLATE.sub(" ", s)
    s = s.replace("'", "").replace('"', "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def make_key(kind: str, predicate: str, columns: list[str], model_id: str,
             extra: str = "") -> str:
    """Stable cache key for one operator compile.

    The predicate is canonicalized (canon_predicate) so the same task phrased
    differently across backends shares one artifact; columns/model/kind still
    bind the key (so reviewText-only vs full-row stay separate)."""
    payload = json.dumps(
        {"kind": kind, "predicate": canon_predicate(predicate),
         "columns": sorted(columns), "model": model_id, "extra": extra},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _path(key: str) -> Path:
    return _DEFAULT_DIR / f"{key}.json"


def get(key: str):
    """Return a CompiledArtifact rebuilt from cached code, or None."""
    from sembaker.core.compiler import CompiledArtifact, _exec_code

    p = _path(key)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        fn = _exec_code(rec["code"], rec["fn_name"])
    except Exception:
        return None  # corrupt entry -> treat as miss
    return CompiledArtifact(
        fn=fn, code=rec["code"], fn_name=rec["fn_name"],
        model_id=rec.get("model_id", ""),
        input_tokens=0, output_tokens=0, cost_usd=0.0, duration_secs=0.0,
        from_cache=True,
    )


def put(key: str, artifact) -> None:
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "fn_name": artifact.fn_name,
        "code": artifact.code,
        "model_id": artifact.model_id,
        "input_tokens": artifact.input_tokens,
        "output_tokens": artifact.output_tokens,
        "cost_usd": artifact.cost_usd,
        "compile_secs": artifact.duration_secs,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _path(key).write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def get_text(key: str):
    """Return a cached plain-text artifact (e.g. a refined predicate), or None."""
    p = _DEFAULT_DIR / f"{key}.txt"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def put_text(key: str, text: str) -> None:
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    (_DEFAULT_DIR / f"{key}.txt").write_text(text, encoding="utf-8")


def clear() -> int:
    """Delete all cached artifacts; returns count removed."""
    n = 0
    if _DEFAULT_DIR.exists():
        for pat in ("*.json", "*.txt"):
            for p in _DEFAULT_DIR.glob(pat):
                p.unlink()
                n += 1
    return n
