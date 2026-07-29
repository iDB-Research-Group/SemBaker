"""PZ warm: pre-compile a query plan's semantic filter & join operators into
the shared cache, in parallel, WITHOUT editing Palimpzest.

Palimpzest is streaming: the Compiled* operators compile on the FIRST record,
which stalls the stream for one LLM round-trip. This module walks the Dataset
plan ahead of time (public `_operator` / `_sources` — no PZ source edit),
extracts each FilteredScan / JoinOp predicate + input columns + source sample
rows, and compiles them CONCURRENTLY via sembaker.core.scheduler into the same cache
the streaming operators read. On execution the operators hit the cache instead
of compiling inline, so:

  - within-query : a query's filter/join compile together (not one-at-a-time
    on their respective first rows);
  - cross-query  : several queries' operators all compile at once.

The compiled artifact and its cache key are backend-agnostic, so a warmed
artifact is reused whether the op later runs under PZ, Lotus, Nirvana or DocETL.

Scope: filter and join only. sem_map (ConvertScan) is skipped — its cache key
depends on the physical op's *reconstructed* instruction (built from the
generated field's description at run time), which we cannot reproduce from the
logical plan alone; warming it would miss and double-compile.
"""
from __future__ import annotations

from sembaker.core import scheduler


def _walk(ds, seen):
    """Yield (operator, owner_dataset) for every node in the plan.

    Uses only public plan structure: each Dataset carries its logical operator
    on `_operator` and its upstream Datasets on `_sources`."""
    if ds is None or id(ds) in seen:
        return
    seen.add(id(ds))
    op = getattr(ds, "_operator", None)
    if op is not None:
        yield op, ds
    for src in (getattr(ds, "_sources", None) or []):
        yield from _walk(src, seen)


def _source_vals(ds):
    """The DataFrame feeding this sub-plan: walk down to the nearest node that
    carries a `vals` table (a MemoryDataset BaseScan)."""
    seen = set()
    stack = [ds]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        vals = getattr(cur, "vals", None)
        if vals is not None and hasattr(vals, "columns"):
            return vals
        stack.extend(getattr(cur, "_sources", None) or [])
    return None


def _sample_records(df, cols, k):
    if df is None:
        return []
    if cols and all(c in df.columns for c in cols):
        df = df[cols]
    return df.head(k).to_dict("records")


def plan_compile_tasks(ds, model_id, *, k=10, op_filter=None):
    """Zero-arg compile tasks for every semantic filter/join in `ds`'s plan.

    Each task calls the SAME `compile_operator(...)` the streaming Compiled*
    operator would, with the SAME (predicate, columns, model) — so it writes the
    exact cache key the operator later looks up. Samples don't enter the key;
    they only feed refine/validate when those switches are on.

    op_filter: optional callable(logical_op) -> bool; ops it rejects are
    skipped (used by decision-routed runs to warm ONLY the ops decide()
    sends to the compiled path)."""
    from sembaker.optimizer.compile_op import compile_operator

    tasks = []
    for op, owner in _walk(ds, set()):
        if op_filter is not None and not op_filter(op):
            continue
        t = type(op).__name__
        if t == "FilteredScan":
            filt = getattr(op, "filter", None)
            # filter_fn set -> deterministic lambda filter, nothing to compile.
            if filt is None or getattr(filt, "filter_fn", None) is not None:
                continue
            cond = filt.filter_condition
            # depends_on == the columns the streaming CompiledFilter keys on
            # (its get_input_fields()); make_key sorts columns so order is moot.
            fields = list(getattr(op, "depends_on", None) or [])
            samples = _sample_records(_source_vals(owner), fields, k)
            # Register the samples so the STREAMING CompiledFilter (which has no
            # batch at exec time) can rebuild the SAME refine cache key when
            # CX_REFINE/CX_VALIDATE are on -> exec hits the warm artifact
            # instead of re-refining with empty samples (key mismatch).
            from sembaker.optimizer.refine_gate import register_samples
            register_samples("filter", samples, intent=cond)
            tasks.append(
                lambda cond=cond, fields=fields, samples=samples: compile_operator(
                    "filter", cond, model_id,
                    cols=fields, samples=samples, tag="pz-warm-filter"))
        elif t == "JoinOp":
            cond = getattr(op, "condition", "") or ""
            if not cond:
                continue
            srcs = getattr(owner, "_sources", None) or []
            if len(srcs) < 2:
                continue
            # Left/right source tables carry the exact columns the join records
            # expose (the right table is already *_right-renamed upstream), and
            # the join cache key is the union of both column sets (sorted).
            sl = _sample_records(_source_vals(srcs[0]), None, k)
            sr = _sample_records(_source_vals(srcs[1]), None, k)
            tasks.append(
                lambda cond=cond, sl=sl, sr=sr: compile_operator(
                    "join", cond, model_id,
                    samples_left=sl, samples_right=sr, tag="pz-warm-join"))
    return tasks


def warm(datasets, model_id, *, k=10, workers=None, label="pz-warm"):
    """Pre-compile all filter/join ops across the given (un-run) PZ query plans,
    in parallel, into the cache. Returns the number of compile tasks issued.

    model_id: BARE model id (e.g. 'gpt-5-mini-2025-08-07') — matching what the
    streaming Compiled* ops derive via model.value.split('/')[-1]."""
    tasks = []
    for ds in datasets:
        tasks.extend(plan_compile_tasks(ds, model_id, k=k))
    if tasks:
        scheduler.run(tasks, workers=workers, label=label)
    return len(tasks)
