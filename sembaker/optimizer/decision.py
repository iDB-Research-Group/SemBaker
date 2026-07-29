"""decide(): the per-operator compile-vs-native decision.

This is the single shared decision layer for ALL backends (PZ / Lotus /
Nirvana). A backend rewriter walks the user's pipeline, calls decide() for
each semantic operator, and swaps the op to its cx_* counterpart when
use_cx=True. The rewriter owns the swap mechanics; this module owns the
judgment.

Decision logic (v1, deliberately simple):
  1. N below the empirical crossover  -> native (compile cost can't amortize)
  2. predicate not codifiable         -> native (heuristic won't capture it)
  3. otherwise                        -> cx

Compileability is judged either by a zero-cost keyword heuristic (default)
or by one cached LLM call per unique predicate (judge="llm").
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from sembaker.optimizer.cost_model import crossover


@dataclass(frozen=True)
class Decision:
    use_cx: bool
    reason: str


# ---- compileability ---------------------------------------------------------

# Markers that historically compiled WELL (structural / lexical / comparable):
# substring & keyword predicates, numeric/date comparisons, field equality.
# Markers that historically compiled POORLY: judgments that need reading
# comprehension of free text with no lexical anchor.
_HARD_SEMANTIC_MARKERS = (
    "tone", "sarcas", "irony", "implies", "implied", "nuance", "reads as",
    "subjective", "style", "metaphor", "humor", "humour", "persuasive",
    "argues", "refutes", "contradicts", "reasoning", "multi-hop",
)

_LLM_JUDGE_PROMPT = """You judge whether a natural-language data-processing predicate can be
faithfully compiled into a deterministic Python function (regex / keyword /
field comparison / arithmetic), given only the row's fields.

Predicate: "{predicate}"
Available fields: {fields}

Answer ONLY with JSON: {{"codifiable": true|false, "reason": "<one short sentence>"}}"""

_judge_cache: dict[str, bool] = {}


def _heuristic_codifiable(predicate: str) -> tuple[bool, str]:
    p = predicate.lower()
    hits = [m for m in _HARD_SEMANTIC_MARKERS if m in p]
    if hits:
        return False, f"hard-semantic markers: {hits[:3]}"
    return True, "no hard-semantic markers"


def _llm_codifiable(predicate: str, fields: list[str], model_id: str) -> tuple[bool, str]:
    key = f"{model_id}::{predicate}"
    if key in _judge_cache:
        return _judge_cache[key], "cached LLM judgment"
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": _LLM_JUDGE_PROMPT.format(
            predicate=predicate, fields=", ".join(fields))}],
        temperature=1,
        response_format={"type": "json_object"},
    )
    try:
        verdict = json.loads(resp.choices[0].message.content)
        ok = bool(verdict.get("codifiable", True))
        reason = str(verdict.get("reason", ""))[:120]
    except Exception:
        ok, reason = True, "judge parse failure -> default codifiable"
    _judge_cache[key] = ok
    return ok, f"LLM judge: {reason}"


# ---- decide -----------------------------------------------------------------


def decide(
    op_type: str,
    predicate: str,
    est_n: int,
    fields: list[str] | None = None,
    *,
    objective: str = "wall",
    judge: str = "heuristic",
    judge_model: str = os.getenv("CX_JUDGE_MODEL", os.getenv("CX_COMPILE_MODEL", "gpt-5-mini-2025-08-07")),
) -> Decision:
    """Per-operator compile-vs-native decision.

    Args:
        op_type: "filter" | "map" | "join".
        predicate: the natural-language condition / instruction.
        est_n: estimated records (filter/map) or candidate pairs (join)
            the operator will process.
        fields: input field names (used by the LLM judge).
        objective: "wall" (default) or "cost" — which crossover to use.
        judge: "heuristic" (free, default) or "llm" (1 cached call per
            unique predicate).
    """
    if op_type not in ("filter", "map", "join"):
        return Decision(False, f"unknown op_type {op_type!r} -> native")

    n_star = crossover(op_type, objective)
    if est_n < n_star:
        return Decision(False, f"N={est_n} < crossover {n_star} ({objective}) -> compile cost cannot amortize")

    if judge == "llm" and fields:
        ok, why = _llm_codifiable(predicate, fields, judge_model)
    else:
        ok, why = _heuristic_codifiable(predicate)
    if not ok:
        return Decision(False, f"predicate not codifiable ({why})")

    return Decision(True, f"N={est_n} >= crossover {n_star} ({objective}); codifiable ({why})")
