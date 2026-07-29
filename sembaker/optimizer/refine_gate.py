"""Shared refine gate for all backends.

When CX_REFINE=1, every compile site first rewrites its fuzzy predicate into
a concrete, column-grounded one (sembaker.core.refine_predicate) before building
the compile prompt. The refined text is cached (keyed on the original intent
+ schema + model), and the DOWNSTREAM compile keys on the refined text — so
refined and non-refined artifacts live in separate cache namespaces and a
refine-mode run reuses refine-mode artifacts.

Off by default; flip with env CX_REFINE=1.
"""
from __future__ import annotations

import os

from sembaker.core import make_key, refine_predicate
from sembaker.core.validate import make_validator

REFINE = os.environ.get("CX_REFINE") == "1"
VALIDATE = os.environ.get("CX_VALIDATE") == "1"

# Shared "original predicate -> refined predicate" map. Backends whose
# operators stream records one-at-a-time (PZ) cannot sample at compile time;
# a warm phase (which DOES have the data) calls register_refined() so those
# operators can later look up the refined text by the original intent and
# compute the SAME cache key the warm phase used -> cache hit on the
# validated artifact. Keyed by (kind, original_intent).
_REFINED: dict[tuple[str, str], str] = {}


def register_refined(kind: str, intent: str, refined: str) -> None:
    _REFINED[(kind, intent)] = refined


def lookup_refined(kind: str, intent: str) -> str:
    """Return the registered refined predicate, or the original if none
    (and refine is off / no warm phase ran)."""
    return _REFINED.get((kind, intent), intent)


# Sample registry for STREAMING backends (PZ): the operator sees one record at
# a time and cannot batch-sample at compile time, so a warm phase (which DOES
# have the data) registers a few sample rows here keyed by op kind. The
# streaming op then looks them up at first compile to drive refine + validate
# inline — using its OWN intent/columns, so no string reconstruction is needed.
# Keyed by kind alone: within one streamed query at most one predicate per kind
# is active, so this is unambiguous. Pass intent for a tighter (kind,intent) key
# when several predicates of the same kind coexist (eager all-at-once warm).
_SAMPLES: dict[tuple[str, str], list[dict]] = {}


def register_samples(kind: str, rows: list[dict], intent: str | None = None) -> None:
    _SAMPLES[(kind, "")] = list(rows or [])
    if intent is not None:
        _SAMPLES[(kind, intent)] = list(rows or [])


def lookup_samples(kind: str, intent: str | None = None) -> list[dict]:
    if intent is not None and (kind, intent) in _SAMPLES:
        return _SAMPLES[(kind, intent)]
    return _SAMPLES.get((kind, ""), [])


def clear_samples() -> None:
    _SAMPLES.clear()


def maybe_refine(kind: str, intent: str, model_id: str, *,
                 samples: list[dict] | None = None,
                 samples_left: list[dict] | None = None,
                 samples_right: list[dict] | None = None) -> str:
    """Return the predicate text to compile with: refined if CX_REFINE=1,
    else the original intent unchanged."""
    if not REFINE:
        return intent
    if kind == "join":
        cols = (list(samples_left[0].keys()) if samples_left else []) + ["|"] + \
               (list(samples_right[0].keys()) if samples_right else [])
    else:
        cols = list(samples[0].keys()) if samples else []
    rk = make_key("refine:" + kind, intent, cols, model_id)
    refined = refine_predicate(kind, intent, model_id, samples=samples,
                               samples_left=samples_left, samples_right=samples_right,
                               cache_key=rk)
    register_refined(kind, intent, refined)
    return refined


def maybe_validator(kind: str, intent: str, model_id: str, *,
                    samples: list[dict] | None = None,
                    samples_left: list[dict] | None = None,
                    samples_right: list[dict] | None = None):
    """Return a validator(compiled_fn)->score if CX_VALIDATE=1, else None.

    `intent` should be the SAME text used to build the compile prompt (i.e.
    the refined predicate when refine is on) so labels match what the
    compiled fn is meant to do. Labels are LLM-judged once and cached."""
    if not VALIDATE:
        return None
    if kind == "join":
        cols = (list(samples_left[0].keys()) if samples_left else []) + ["|"] + \
               (list(samples_right[0].keys()) if samples_right else [])
    else:
        cols = list(samples[0].keys()) if samples else []
    lk = make_key("labels:" + kind, intent, cols, model_id)
    return make_validator(kind, intent, model_id, samples=samples,
                          samples_left=samples_left, samples_right=samples_right,
                          label_cache_key=lk)
