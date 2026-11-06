"""Unified, ablation-ready compile entry point for ALL backends.

Every backend routes its operator compile through `compile_operator(...)`, so the
four ablation knobs live in ONE place and apply uniformly across PZ / Lotus /
Nirvana / DocETL:

  CX_COMPILE  = e2e (default) | ir1 | ir2    which compiler
      e2e -> free-form code (sembaker.core.compiler.compile_artifact)
      ir1 -> flat DSL IR      (sembaker.core.compile_ir)     -> deterministic transpile
      ir2 -> expression tree  (sembaker.core.compile_ir2)    -> deterministic transpile
  CX_CACHE    = 1 (default) | 0              reuse/store compiled artifacts
  CX_SEMCACHE = 0 (default) | 1              semantic cache: on an exact-key miss,
      reuse an artifact compiled for a REWORDING of this operator, if the
      classifier in sembaker.core.semcache judges the two equivalent. Requires
      CX_CACHE=1 and the `semcache` extra; off by default because it loads
      ~1.5GB of sentence-transformer encoders.
  CX_REFINE   = 0 (default) | 1              rewrite the predicate first (applies to all methods)
  CX_VALIDATE = 0 (default) | 1              pre-cache validation gate / best-of-N

The compiled artifact is backend-agnostic, so the cache key excludes the backend
but INCLUDES the method (e2e/ir1/ir2 produce different code, must not collide).
"""
from __future__ import annotations

import os
import threading

from sembaker.core import (
    CompiledArtifact,
    build_filter_prompt,
    build_join_prompt,
    build_map_prompt,
    compile_artifact,
    make_key,
)
from sembaker.core import cache as _cache
from sembaker.core.redact import describe
from sembaker.optimizer.refine_gate import maybe_refine, maybe_validator

_IR_MAX_ATTEMPTS = int(os.environ.get("CX_IR_MAX_ATTEMPTS", "3"))

# ---- single-flight table ------------------------------------------------------
# The rendezvous between the compile lane and the execution lane, at OPERATOR
# granularity. Entries are keyed by the artifact's CACHE KEY (content-derived
# from kind+predicate+columns+model), so an executing operator can never grab
# the wrong artifact — and two operators with identical content dedupe to ONE
# compile. Three states at the rendezvous:
#   cache hit          -> take it
#   key in this table  -> someone (e.g. a background pipeline-warm task) is
#                         compiling it right now: WAIT for that one compile
#                         instead of duplicating it
#   neither            -> compile it yourself (and register so others wait)
_INFLIGHT: dict[str, threading.Event] = {}
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_TIMEOUT = float(os.environ.get("CX_INFLIGHT_TIMEOUT", "900"))


def method() -> str:
    return os.environ.get("CX_COMPILE", "e2e").lower()


def _cache_on() -> bool:
    return os.environ.get("CX_CACHE", "1") != "0"


def _sem_on() -> bool:
    return os.environ.get("CX_SEMCACHE", "0") == "1"


def _semantic_hit(prefix, pred, cols_key, model_id, tag):
    """Ask the semantic cache whether an existing artifact already does this.

    Returns a CompiledArtifact on an `equivalent` verdict, else None. Every
    failure mode here (no model file, encoders missing, a stale index entry
    whose artifact has been deleted) degrades to None: a semantic miss costs
    one compile, while letting an exception escape would break a pipeline that
    worked fine before the flag was set."""
    try:
        from sembaker.core import semcache as _sem

        sc = _sem.get_cache()
        if sc is None:
            return None
        _hydrate(sc)
        d = sc.lookup(prefix, pred, list(cols_key), model_id)
        if not d.hit or d.stage != "semantic":
            return None
        art = _cache.get(d.key)
        if art is None:                  # index points at a deleted artifact
            return None
        print(f"[{tag or prefix}] semantic cache HIT ({d.score:.3f}) -> 0 LLM calls\n"
              f"    query  : {pred!r}\n"
              f"    reusing: {d.entry.predicate!r}")
        return art
    except Exception as exc:
        print(f"[semcache] lookup failed, compiling instead: {describe(exc)}")
        return None


def _semantic_insert(cache_meta, key):
    """Register a freshly compiled artifact so later rewordings can find it."""
    if cache_meta is None or not _sem_on():
        return
    try:
        from sembaker.core import semcache as _sem

        sc = _sem.get_cache()
        if sc is not None:
            sc.insert(cache_meta["kind"], cache_meta["predicate"],
                      cache_meta["columns"], cache_meta["model_id"], key=key)
    except Exception:
        pass


def _hydrate(sc):
    """Seed the semantic index from artifacts already on disk, once per cache.

    The flag lives on the cache object rather than in a module global so that
    replacing the cache (semcache.reset_cache(), or a test pointing CX_CACHE_DIR
    somewhere else) gets a fresh hydration instead of silently inheriting a
    "already done" that was true of a different cache."""
    if getattr(sc, "_hydrated", False):
        return
    sc._hydrated = True
    n = 0
    for key, meta in _cache.index():
        try:
            sc.insert(meta["kind"], meta["predicate"], meta["columns"],
                      meta["model_id"], key=key)
            n += 1
        except Exception:
            continue
    if n:
        print(f"[semcache] indexed {n} cached artifacts")


def _cols_key(kind, cols, samples, samples_left, samples_right):
    if kind == "join":
        return ((list(samples_left[0].keys()) if samples_left else []) + ["|"] +
                (list(samples_right[0].keys()) if samples_right else []))
    return cols or (list(samples[0].keys()) if samples else [])


def _ir_compile(m, kind, pred, model_id, cols_key, samples, samples_left, samples_right):
    """One IR draft -> CompiledArtifact (deterministic transpile, no cache/validate)."""
    if m == "ir2":
        from sembaker.core.compile_ir2 import compile_via_tree as _c
    else:
        from sembaker.core.compile_ir import compile_via_ir as _c
    ir, code, fn = _c(kind, pred, model_id,
                      cols=(None if kind == "join" else cols_key),
                      samples=samples, samples_left=samples_left, samples_right=samples_right)
    return CompiledArtifact(fn=fn, code=code, fn_name=f"compiled_{kind}", model_id=model_id,
                            input_tokens=0, output_tokens=0, cost_usd=0.0, duration_secs=0.0)


def compile_operator(kind, intent, model_id, *, cols=None, samples=None,
                     samples_left=None, samples_right=None, extra="",
                     usd_per_input_token=0.0, usd_per_output_token=0.0, tag=None):
    """Compile one operator honoring CX_COMPILE / CX_CACHE / CX_REFINE / CX_VALIDATE.
    Returns a CompiledArtifact (use .fn)."""
    m = method()
    # CX_REFINE: rewrite the predicate first (no-op unless CX_REFINE=1). Applies to every method.
    pred = maybe_refine(kind, intent, model_id, samples=samples,
                        samples_left=samples_left, samples_right=samples_right)
    cols_key = _cols_key(kind, cols, samples, samples_left, samples_right)
    key = None
    cache_meta = None
    if _cache_on():
        # method in the key namespace so e2e/ir1/ir2 artifacts never collide
        prefix = kind if m == "e2e" else f"{m}:{kind}"
        key = make_key(prefix, pred, cols_key, model_id, extra=extra)
        cache_meta = {"kind": prefix, "predicate": pred,
                      "columns": list(cols_key), "model_id": model_id}
        hit = _cache.get(key)
        if hit is not None:
            return hit

        # CX_SEMCACHE=1: the exact key missed, but a REWORDING of this operator
        # may already be compiled. Only an `equivalent` verdict is acted on;
        # anything else falls through to a real compile.
        if _sem_on():
            art = _semantic_hit(prefix, pred, cols_key, model_id, tag)
            if art is not None:
                return art

    # single-flight rendezvous: if this exact artifact is being compiled by
    # another thread (a background pipeline-warm task, or a sibling operator
    # with identical content), wait for THAT compile rather than duplicating.
    own_ev = None
    if key is not None:
        with _INFLIGHT_LOCK:
            ev = _INFLIGHT.get(key)
            if ev is None:
                own_ev = threading.Event()
                _INFLIGHT[key] = own_ev
        if own_ev is None:
            ev.wait(timeout=_INFLIGHT_TIMEOUT)
            hit = _cache.get(key)
            if hit is not None:
                return hit
            # in-flight compile failed/timed out -> compile ourselves (no registration)

    try:
        validator = maybe_validator(kind, pred, model_id, samples=samples,
                                    samples_left=samples_left, samples_right=samples_right)

        if m == "e2e":
            prompt = (build_join_prompt(pred, samples_left, samples_right) if kind == "join"
                      else (build_filter_prompt if kind == "filter" else build_map_prompt)(pred, cols_key))
            # compile_artifact does its own cache + best-of-N via the validator
            art = compile_artifact(
                prompt=prompt, fn_name=f"compiled_{kind}", model_id=model_id,
                tag=tag or f"CX/{kind}", cache_key=key, cache_meta=cache_meta,
                validator=validator,
                usd_per_input_token=usd_per_input_token, usd_per_output_token=usd_per_output_token)
            _semantic_insert(cache_meta, key)
            return art

        # ir1 / ir2 : draft the IR (best-of-N if a validator is active), transpile deterministically
        attempts = _IR_MAX_ATTEMPTS if validator is not None else 1
        best, best_score = None, -1.0
        for _ in range(attempts):
            try:
                art = _ir_compile(m, kind, pred, model_id, cols_key, samples, samples_left, samples_right)
            except Exception:
                continue
            if validator is None:
                best = art
                break
            try:
                score = float(validator(art.fn))
            except Exception:
                score = 0.0
            if score > best_score:
                best, best_score = art, score
            if score >= 0.8:
                break
        if best is None:
            # IR couldn't express this operator. CX_IR_FALLBACK=1 -> fall back to
            # end-to-end (never worse than the current default); else surface it.
            if os.environ.get("CX_IR_FALLBACK") == "1":
                prompt = (build_join_prompt(pred, samples_left, samples_right) if kind == "join"
                          else (build_filter_prompt if kind == "filter" else build_map_prompt)(pred, cols_key))
                best = compile_artifact(
                    prompt=prompt, fn_name=f"compiled_{kind}", model_id=model_id,
                    tag=(tag or f"CX/{kind}") + "-fallback", cache_key=None, validator=validator)
            else:
                raise RuntimeError(f"{m} compile produced no valid artifact for {kind}")
        if key is not None:
            _cache.put(key, best, meta=cache_meta)
            _semantic_insert(cache_meta, key)
        return best
    finally:
        if own_ev is not None:
            with _INFLIGHT_LOCK:
                _INFLIGHT.pop(key, None)
            own_ev.set()  # wake all waiters; they re-read the cache
