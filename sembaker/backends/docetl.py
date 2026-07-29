"""DocETL backend for the external optimizer: Frame map/filter interception.

DocETL (ucbepic/docetl) is a document-level semantic-operator system: a lazy
`Frame` records operations like `.map(prompt=...)` and `.filter(prompt=...)`,
each of which calls an LLM once *per document* at execution. Crucially, DocETL
already ships *code operators* — `.code_map(code=...)` and
`.code_filter(code=...)` — that run a local Python `transform(doc)` with NO LLM
call. That is the perfect injection point for compile-then-execute.

Usage (user keeps writing NATIVE DocETL):

    import docetl
    import sembaker.backends.docetl as cxdoc
    cxdoc.apply()                       # monkeypatch Frame.map / Frame.filter

    f = docetl.from_list(docs)
    f = f.map(prompt="Score 1-5 ... {{ input.reviewText }}", output={"score": "int"})
    f = f.filter(prompt="Keep clearly positive reviews. {{ input.reviewText }}")
    out = f.collect()                   # compiled ops run locally, 0 LLM/doc

How it works
------------
apply() wraps Frame.map / Frame.filter. On each call the wrapper:
  1. parses the natural-language instruction + referenced fields out of the
     Jinja2 prompt;
  2. asks sembaker.optimizer.decide() whether to compile (FORCE_CX bypasses);
  3. if yes, compiles the instruction into a deterministic function via sembaker.core
     (refine + pre-cache validation + persistent cache, all shared with the
     PZ / Lotus / Nirvana backends — the cache key excludes the backend, so a
     DocETL filter can reuse an artifact another backend already compiled), then
     rewrites the op into the native `.code_map` / `.code_filter` with the
     compiled `transform(doc)` as the code body;
  4. otherwise falls through to DocETL's native LLM operator.

Scope: filter and map have native code slots, so they compile. DocETL's join
(`equijoin`) has no code slot, so joins stay native (reported, not rewritten).
"""

from __future__ import annotations

import os
import re
from typing import Any

from sembaker.core import (
    build_filter_prompt,
    build_map_prompt,
    compile_artifact,
    make_key,
)
from sembaker.core.cache import canon_predicate
from sembaker.optimizer.decision import decide
from sembaker.optimizer.refine_gate import maybe_refine, maybe_validator

# Compile model is independent of whatever model DocETL would use to execute the
# native operator — and it is the SAME bare id the other backends compile with,
# so cache keys line up for cross-backend reuse.
COMPILE_MODEL_ID = os.environ.get("CX_COMPILE_MODEL", "gpt-5-mini-2025-08-07")

_FORCE_CX = os.environ.get("FORCE_CX") == "1"
_VERBOSE = os.environ.get("CX_DOCETL_VERBOSE", "1") == "1"
_DEFAULT_EST_N = 1000  # used when the frame's cardinality is unknown (lazy source)
# compile method (e2e / ir1 / ir2), cache, refine, validate are all controlled
# by env switches inside sembaker.optimizer.compile_op — shared by every backend.


# ---- prompt parsing ---------------------------------------------------------

_JINJA = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_FIELD = re.compile(r"\{\{\s*(?:input\.)?([A-Za-z_]\w*)\s*\}\}")
# a trailing field label like "Review:" / "Text:" left behind after stripping
# the Jinja placeholder — drop it so the instruction matches the other backends'
# verbatim predicate (enabling cross-backend cache hits).
_TRAIL_LABEL = re.compile(r"(?i)\b(review|reviews|text|input|document|doc|content)\s*:?\s*$")


def _parse_prompt(prompt: str) -> tuple[str, list[str]]:
    """Return (natural-language instruction, referenced field names)."""
    fields = list(dict.fromkeys(_FIELD.findall(prompt or "")))
    instr = _JINJA.sub(" ", prompt or "")
    instr = re.sub(r"\s+", " ", instr).strip()
    # strip a dangling field label left where the placeholder used to be
    # (e.g. "...positive. Review:" -> "...positive."). Keep sentence
    # punctuation so the instruction matches the other backends verbatim
    # (canon then folds case/space) -> cross-backend cache hits.
    for _ in range(3):
        new = _TRAIL_LABEL.sub("", instr).strip()
        if new == instr:
            break
        instr = new
    return instr, fields


# ---- compile -> transform source --------------------------------------------

# Sample clipping for the compile/validate context. DEFAULT = 0 = NO clipping:
# validate must score the compiled fn on the SAME input distribution it will run
# on (the full document), otherwise a fn that looks right on a clipped sample can
# fail on the full text (e.g. extra numbers later in a transcript) yet pass the
# gate. Only set a positive cap (CX_DOCETL_SAMPLE_CHARS) when documents are so
# long that full samples would blow the compile-prompt budget; in that regime the
# faithful fix is DocETL's own split/gather rather than silent truncation.
_SAMPLE_CHARS = int(os.environ.get("CX_DOCETL_SAMPLE_CHARS", "0"))


def _clip(v):
    if _SAMPLE_CHARS > 0 and isinstance(v, str) and len(v) > _SAMPLE_CHARS:
        return v[:_SAMPLE_CHARS]
    return v


def _sample_docs(frame, fields: list[str], k: int = 10) -> list[dict]:
    """Best-effort: pull a few in-memory sample docs (sliced to `fields`, each
    field clipped) so refine/validate have context. [] for non-memory sources."""
    try:
        ds = frame._datasets.get(frame._first_dataset, {})
        if ds.get("type") == "memory" and isinstance(ds.get("path"), list):
            rows = ds["path"][:k]
            cols = fields or (list(rows[0].keys()) if rows else [])
            return [{c: _clip(r.get(c)) for c in cols} for r in rows]
    except Exception:
        pass
    return []


def _est_n(frame) -> int:
    try:
        ds = frame._datasets.get(frame._first_dataset, {})
        if ds.get("type") == "memory" and isinstance(ds.get("path"), list):
            return max(1, len(ds["path"]))
    except Exception:
        pass
    return _DEFAULT_EST_N


def _compile(kind: str, instr: str, fields: list[str], samples: list[dict]):
    """Compile via the shared, ablation-aware dispatcher (honors CX_COMPILE /
    CX_CACHE / CX_REFINE / CX_VALIDATE). Returns a CompiledArtifact."""
    from sembaker.optimizer.compile_op import compile_operator

    cols = fields or (list(samples[0].keys()) if samples else [])
    return compile_operator(kind, instr, COMPILE_MODEL_ID, cols=cols, samples=samples,
                            tag=f"CX/docetl-{kind}")


def _filter_transform_src(artifact) -> str:
    return artifact.code + (
        "\n\ndef transform(doc):\n"
        "    try:\n"
        "        return bool(compiled_filter(doc))\n"
        "    except Exception:\n"
        "        return False\n"
    )


def _map_transform_from_code(code: str, out_field: str) -> str:
    """Wrap a `compiled_map(row)` source into a DocETL `transform(doc)->dict`."""
    return code + (
        "\n\ndef transform(doc):\n"
        "    try:\n"
        "        v = compiled_map(doc)\n"
        "    except Exception:\n"
        "        v = None\n"
        f"    return {{{out_field!r}: ('' if v is None else v)}}\n"
    )


def _map_transform_src(artifact, out_field: str) -> str:
    return _map_transform_from_code(artifact.code, out_field)


def _out_field(output: dict | None, default: str) -> str:
    if isinstance(output, dict) and output:
        return next(iter(output.keys()))
    return default


# ---- monkeypatch ------------------------------------------------------------

_PATCHED = False


def apply(verbose: bool | None = None) -> None:
    """Wrap Frame.map / Frame.filter so semantic ops compile-then-execute via
    DocETL's native code operators. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    from docetl.frame import Frame

    v = _VERBOSE if verbose is None else verbose
    _orig_map = Frame.map
    _orig_filter = Frame.filter

    def _decide(kind, instr, frame, fields):
        if _FORCE_CX:
            return True, "FORCE_CX"
        d = decide(kind, instr, _est_n(frame), fields)
        return d.use_cx, d.reason

    def map(self, name=None, *, prompt=None, output=None, model=None, **kw):  # type: ignore[override]
        if not prompt:
            return _orig_map(self, name, prompt=prompt, output=output, model=model, **kw)
        instr, fields = _parse_prompt(prompt)
        use_cx, reason = _decide("map", instr, self, fields)
        if v:
            print(f"[cx/docetl map] N~{_est_n(self)} {'-> CX' if use_cx else '-> native'} "
                  f"({reason})  {instr[:60]!r}")
        if not use_cx:
            return _orig_map(self, name, prompt=prompt, output=output, model=model, **kw)
        # compile method (e2e / ir1 / ir2) is chosen by CX_COMPILE inside _compile
        art = _compile("map", instr, fields, _sample_docs(self, fields))
        src = _map_transform_src(art, _out_field(output, "_map"))
        return self.code_map(name=name, code=src)

    def filter(self, name=None, *, prompt=None, output=None, model=None, **kw):  # type: ignore[override]
        if not prompt:
            return _orig_filter(self, name, prompt=prompt, output=output, model=model, **kw)
        instr, fields = _parse_prompt(prompt)
        use_cx, reason = _decide("filter", instr, self, fields)
        if v:
            print(f"[cx/docetl filter] N~{_est_n(self)} {'-> CX' if use_cx else '-> native'} "
                  f"({reason})  {instr[:60]!r}")
        if not use_cx:
            return _orig_filter(self, name, prompt=prompt, output=output, model=model, **kw)
        art = _compile("filter", instr, fields, _sample_docs(self, fields))
        src = _filter_transform_src(art)
        return self.code_filter(name=name, code=src)

    Frame.map = map
    Frame.filter = filter
    Frame._cx_orig_map = _orig_map
    Frame._cx_orig_filter = _orig_filter
    _PATCHED = True
    if v:
        print("[cx/docetl] Frame.map / Frame.filter patched (compile-then-execute)")


def restore() -> None:
    """Undo apply() (restore native Frame.map / Frame.filter)."""
    global _PATCHED
    if not _PATCHED:
        return
    from docetl.frame import Frame

    Frame.map = Frame._cx_orig_map
    Frame.filter = Frame._cx_orig_filter
    _PATCHED = False


# ---- warm (parallel pre-compile) --------------------------------------------

def _samples_from_docs(docs, fields, k: int = 10) -> list[dict]:
    rows = (docs or [])[:k]
    cols = fields or (list(rows[0].keys()) if rows else [])
    return [{c: _clip(r.get(c)) for c in cols} for r in rows]


def warm(op_specs, *, k: int = 10, workers=None, label: str = "docetl-warm") -> int:
    """Pre-compile a set of DocETL semantic ops in PARALLEL into the shared cache.

    Unlike PZ (which compiles lazily on the first record) DocETL compiles at BUILD
    time — each `.map`/`.filter` call compiles synchronously — so a pipeline's ops
    otherwise compile one-at-a-time as the frame is chained. warm compiles them
    concurrently up front via sembaker.core.scheduler; the later `.map`/`.filter` calls
    then hit the cache (identical key: kind + parsed instruction + fields + model).
    Rides the existing compile path — NO new monkeypatch, NO DocETL source edit.

    op_specs: iterable of {"kind": "map"|"filter", "prompt": str, "docs": list?}.
              `output` is ignored (it never enters the cache key); `docs` is only
              refine/validate context (the key never depends on samples), so it may
              be omitted when those switches are off. Duplicate ops (same kind +
              canonicalized instruction + fields — e.g. two queries sharing a
              rubric prompt) are compiled once.
    Returns the number of distinct compile tasks issued."""
    from sembaker.core import scheduler

    tasks, seen = [], set()
    for spec in op_specs:
        kind = spec.get("kind")
        prompt = spec.get("prompt")
        if kind not in ("map", "filter") or not prompt:
            continue
        instr, fields = _parse_prompt(prompt)
        dedup = (kind, canon_predicate(instr), tuple(sorted(fields)))
        if dedup in seen:
            continue
        seen.add(dedup)
        docs = spec.get("docs") or []
        n = len(docs) or _DEFAULT_EST_N
        if not (_FORCE_CX or decide(kind, instr, n, fields).use_cx):
            continue
        samples = _samples_from_docs(docs, fields, k)
        tasks.append(
            lambda kind=kind, instr=instr, fields=fields, samples=samples:
            _compile(kind, instr, fields, samples))
    if tasks:
        scheduler.run(tasks, workers=workers, label=label)
    return len(tasks)
