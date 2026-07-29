"""Nirvana backend for the external optimizer: UDF-slot injection.

Usage (user keeps writing NATIVE Nirvana APIs):

    import nirvana as nv
    import sembaker.backends.nirvana as cxnv

    df = nv.DataFrame(reviews_df)
    df = df.semantic_filter("the review is clearly positive", input_columns=["reviewText"])
    report = cxnv.optimize(df)        # walk lineage plan, decide, inject
    out, cost, secs = df.execute()    # Nirvana executes the rewritten plan

How the "rewrite" works
-----------------------
Nirvana plans are LineageNode chains (df.leaf_node -> left_child -> ... ->
scan). Every semantic operation natively carries an OPTIONAL UDF slot
(`func` / `operator.tool`): when set, Nirvana's `_execute_by_func` runs the
function per row at zero token cost and AUTOMATICALLY falls back to the
per-row LLM path if the function raises. We exploit exactly that designed
extension point:

  for each filter/map node where decide() says CX:
      node.operator.tool = FunctionCallTool.from_function(
          func=<lazily-compiled cx function>, name="cx_compiled_*")

No node replacement, no rule gating — Nirvana's own UDF dispatch does the
rest, including graceful LLM fallback on any compiled-function error.
Filter/map/join are injected; rank/reduce injection not yet implemented
(reported native).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:
    from nirvana.executors.tools import FunctionCallTool
except ImportError as e:
    raise ImportError(
        "The Nirvana backend requires the 'nirvana-ai' package: "
        "pip install nirvana-ai  (or: pip install sembaker[nirvana])"
    ) from e

from sembaker.core import build_filter_prompt, build_join_prompt, build_map_prompt, compile_artifact, make_key
from sembaker.optimizer.decision import Decision, decide

COMPILE_MODEL_ID = os.getenv("CX_COMPILE_MODEL", "gpt-5-mini-2025-08-07")

_DEFAULT_ROOT_N = 100
_FILTER_SELECTIVITY = 0.5


# ---- upstream bug patches ----------------------------------------------------


def apply_upstream_patches(verbose: bool = True) -> None:
    """Runtime patches for two nirvana-ai bugs (present in PyPI 1.3.1 AND
    GitHub 1.3.2) that break EVERY semantic_join plan — native or cx:

    Bug 1: DataFrame.semantic_join fills the NodeFields "form" with the
        wrong field names (input_left_fields/input_right_fields instead of
        left_input_fields/right_input_fields) -> TypeError at plan build.
        Patch: NodeFields.__init__ accepts both spellings.

    Bug 2: LineageNode.run's join branch adds a temp "keys" column to BOTH
        frames, then calls pandas .join(on="keys") with no suffix — the
        temp column itself overlaps, so every join raises "columns
        overlap". Patch: re-implement just the join branch with
        .merge(on="keys", suffixes=(":left", ":right")) (which is plainly
        what the code intended); all other op branches delegate to the
        original method.

    Idempotent; applied automatically when this module is imported (the
    plan-build bug fires BEFORE optimize() could run, so import time is
    the only reliable hook). Remove once fixed upstream:
    https://github.com/JunHao-Zhu/nirvana
    """
    import nirvana.lineage.abstractions as _abs

    if getattr(_abs, "_cx_upstream_patched", False):
        return

    # ---- Bug 1: kwarg-name mismatch ----
    _orig_init = _abs.NodeFields.__init__

    def _patched_init(self, *args, **kwargs):
        for bad, good in (("input_left_fields", "left_input_fields"),
                          ("input_right_fields", "right_input_fields")):
            if bad in kwargs and good not in kwargs:
                kwargs[good] = kwargs.pop(bad)
        kwargs.setdefault("output_fields", [])
        known = {"left_input_fields", "right_input_fields", "output_fields"}
        kwargs = {k: v for k, v in kwargs.items() if k in known}
        _orig_init(self, *args, **kwargs)

    _abs.NodeFields.__init__ = _patched_init

    # ---- Bug 2: suffix-less pandas join on an overlapping temp column ----
    _orig_run = _abs.LineageNode.run

    async def _patched_run(self, input=None):
        if self.op_name == "join":
            op_outputs = await self.operator.execute(left_data=input[0], right_data=input[1])
            if not getattr(op_outputs, "join_pairs", None):
                # zero matches (or an empty side): collate would try to assign
                # empty key lists onto non-empty frames -> length mismatch.
                return _abs.NodeOutput(output=pd.DataFrame(), cost=op_outputs.cost)
            left = input[0].copy()
            right = input[1].copy()
            left["keys"] = op_outputs.left_join_keys
            right["keys"] = op_outputs.right_join_keys
            output = left.merge(
                right, on="keys", how=self.operator.how, suffixes=(":left", ":right"),
            ).drop(columns=["keys"])
            return _abs.NodeOutput(output=output, cost=op_outputs.cost)
        return await _orig_run(self, input)

    _abs.LineageNode.run = _patched_run

    # ---- Bug 3: dtype inference samples more rows than exist ----
    # arrays/utils.infer_dtype does col.sample(10) unconditionally; any
    # DataFrame with < 10 rows crashes nv.DataFrame() construction
    # ("Cannot take a larger sample than population").
    import nirvana.dataframe.arrays.utils as _au

    _orig_infer = _au.infer_dtype

    def _patched_infer(col):
        if len(col) < 10:
            col = pd.concat([col] * (10 // max(1, len(col)) + 1), ignore_index=True)
        return _orig_infer(col)

    _au.infer_dtype = _patched_infer
    # (infer_and_convert_dtype resolves infer_dtype via the module global,
    # so patching the module attribute is effective.)

    # ---- Bug 5: empty-side join crashes on a bad kwarg ----
    # ops/join.py's execute() early-returns JoinOpOutputs(output=[], ...) when
    # either input is empty, but the dataclass has no `output` field ->
    # TypeError on EVERY join whose upstream filter emptied one side (the
    # common case on HybridQA). Return a correctly-formed empty result.
    from nirvana.ops.join import JoinOperation as _JoinOp
    from nirvana.ops.join import JoinOpOutputs as _JOO

    _orig_join_exec = _JoinOp.execute

    async def _patched_join_exec(self, left_data, right_data, **kw):
        if left_data.empty or right_data.empty:
            return _JOO(join_pairs=[], left_join_keys=[], right_join_keys=[], cost=0.0)
        return await _orig_join_exec(self, left_data, right_data, **kw)

    _JoinOp.execute = _patched_join_exec

    # ---- Shim 4: gpt-5 model compatibility ----
    # llm_backbone hardcodes temperature=0.1 on every Responses API call;
    # the gpt-5 family rejects the parameter outright (400). Strip it for
    # gpt-5* models at the client level so nirvana's NATIVE path can run
    # on current OpenAI models at all.
    import nirvana.executors.llm_backbone as _lb

    _orig_create_client = _lb._create_client

    def _patched_create_client(api_key, **kw):
        client = _orig_create_client(api_key, **kw)
        _orig_responses_create = client.responses.create

        async def _create(*a, **k):
            model = k.get("model", "")
            if isinstance(model, str) and model.startswith("gpt-5"):
                k.pop("temperature", None)
            return await _orig_responses_create(*a, **k)

        client.responses.create = _create
        return client

    _lb._create_client = _patched_create_client

    # ---- Bug 6: map postprocess crashes when the LLM omits an output tag ----
    # ops/map.py's _postprocess_map_output does `value == "" or value.lower()`
    # on llm_result.get(column, None); a missing tag yields None -> the ==
    # short-circuit fails and None.lower() raises AttributeError, killing the
    # whole plan (only KeyError is caught). Treat missing/None as None output.
    from nirvana.ops.map import MapOperation as _MapOp

    _orig_map_post = _MapOp._postprocess_map_output

    def _patched_map_post(self, llm_result, output_columns):
        if isinstance(llm_result, dict):
            llm_result = {k: ("" if v is None else v) for k, v in llm_result.items()}
            for col in output_columns:  # absent tag == empty output
                llm_result.setdefault(col, "")
        return _orig_map_post(self, llm_result, output_columns)

    _MapOp._postprocess_map_output = _patched_map_post

    # ---- Bug 7: LLM retry loop exhausts -> UnboundLocalError ----
    # executors/llm_backbone.py's __call__ swallows every exception inside its
    # retry loop; if all max_timeouts attempts fail (rate limit / connection
    # errors) `llm_output` is never assigned and line 132 raises
    # UnboundLocalError, killing the whole plan. Convert an exhausted retry
    # into an empty-output result (None tags / cost 0), which downstream
    # patches (bug 6, join None-guard) already tolerate.
    _orig_llm_call = _lb.LLMClient.__call__

    async def _patched_llm_call(self, messages, parse_tags=False, parse_code=False, **kw):
        try:
            return await _orig_llm_call(self, messages, parse_tags=parse_tags,
                                        parse_code=parse_code, **kw)
        except UnboundLocalError:  # retry budget exhausted, no response at all
            outputs = {"raw_output": None, "cost": 0.0}
            if parse_tags:
                outputs.update({tag: None for tag in kw.get("tags", [])})
            else:
                outputs["output"] = None
            return outputs

    _lb.LLMClient.__call__ = _patched_llm_call

    _abs._cx_upstream_patched = True
    if verbose:
        print("[sembaker.backends.nirvana] applied runtime patches for upstream "
              "nirvana-ai issues (NodeFields kwarg mismatch; suffix-less join collate; "
              "small-frame dtype sampling; gpt-5 temperature compat; empty-side join; "
              "None map-output postprocess)")


apply_upstream_patches()


# ---- lazily-compiled per-row callables --------------------------------------


class _LazyCompiledFilter:
    """series -> bool; compiles on first call (one sembaker.core LLM call)."""

    def __init__(self, instruction: str, columns: list[str], model_id: str = COMPILE_MODEL_ID,
                 samples: list[dict] | None = None):
        self.instruction = instruction
        self.columns = columns
        self.model_id = model_id
        self.samples = samples or []
        self.artifact = None

    def ensure_compiled(self):
        if self.artifact is None:
            from sembaker.optimizer.compile_op import compile_operator
            self.artifact = compile_operator("filter", self.instruction, self.model_id,
                                             cols=self.columns, samples=self.samples,
                                             tag="CX/nirvana-filter")

    def __call__(self, series) -> bool:
        self.ensure_compiled()
        return bool(self.artifact.fn(series.to_dict()))


class _LazyCompiledMap:
    """series -> {out_col: value}; compiles on first call."""

    def __init__(self, instruction: str, columns: list[str], output_columns: list[str],
                 model_id: str = COMPILE_MODEL_ID, samples: list[dict] | None = None):
        self.instruction = instruction
        self.columns = columns
        self.output_columns = output_columns
        self.model_id = model_id
        self.samples = samples or []
        self.artifact = None

    def ensure_compiled(self):
        if self.artifact is None:
            from sembaker.optimizer.compile_op import compile_operator
            self.artifact = compile_operator("map", self.instruction, self.model_id,
                                             cols=self.columns, samples=self.samples,
                                             tag="CX/nirvana-map")

    def __call__(self, series) -> dict:
        self.ensure_compiled()
        v = self.artifact.fn(series.to_dict())
        # ALWAYS stringify: nirvana's _postprocess_map_output calls
        # value.lower() on every output value, which crashes on int/float
        # returns (and the crash silently falls back to the per-row LLM
        # path, defeating the compile). str() keeps the UDF path engaged.
        def _s(x):
            return "" if x is None else str(x)

        # Nirvana's map collate does pd.DataFrame(list-of-row-outputs); a dict
        # per row gives the generated column(s) their proper names.
        if len(self.output_columns) == 1:
            return {self.output_columns[0]: _s(v)}
        if isinstance(v, dict):
            return {c: _s(v.get(c, "")) for c in self.output_columns}
        return {c: _s(v) for c in self.output_columns}


class _LazyCompiledJoin:
    """(left_series, right_series) -> bool; compiles on first call.

    Nirvana's join op passes each side's series sliced to left_on/right_on
    columns, so compile samples are sliced the same way — the compiled fn
    only ever sees the keys it will receive at runtime.
    """

    def __init__(self, instruction: str, samples_left: list[dict], samples_right: list[dict],
                 model_id: str = COMPILE_MODEL_ID):
        self.instruction = instruction
        self.samples_left = samples_left
        self.samples_right = samples_right
        self.model_id = model_id
        self.artifact = None

    def ensure_compiled(self):
        if self.artifact is None:
            from sembaker.optimizer.compile_op import compile_operator
            self.artifact = compile_operator("join", self.instruction, self.model_id,
                                             samples_left=self.samples_left,
                                             samples_right=self.samples_right,
                                             tag="CX/nirvana-join")

    def __call__(self, left_series, right_series) -> bool:
        self.ensure_compiled()
        return bool(self.artifact.fn(left_series.to_dict(), right_series.to_dict()))


def _scan_datasource(node):
    """Walk a chain to its scan node and return the datasource DataFrame."""
    while node is not None:
        if getattr(node, "op_name", "") == "scan":
            return getattr(node, "datasource", None)
        node = node.left_child
    return None


# ---- rewrite pass -----------------------------------------------------------


@dataclass
class RewriteReport:
    decisions: list[tuple[str, str, int, Decision]] = field(default_factory=list)

    def __str__(self):
        lines = ["=== sembaker.optimizer (nirvana) plan rewrite ==="]
        for kind, pred, n, d in self.decisions:
            arrow = "-> CX " if d.use_cx else "-> native"
            lines.append(f"  [{kind:>6}] N~{n:<8} {arrow}  {pred!r}")
            lines.append(f"           reason: {d.reason}")
        return "\n".join(lines)


def _chain(leaf_node) -> list:
    """Linear chain from scan (root) to leaf. Join right-branches are not
    descended (join injection unsupported in v1)."""
    nodes = []
    node = leaf_node
    while node is not None:
        nodes.append(node)
        node = node.left_child
    nodes.reverse()
    return nodes


def optimize(
    df,
    *,
    objective: str = "wall",
    judge: str = "heuristic",
    eager: bool = False,
    verbose: bool = True,
) -> RewriteReport:
    """Walk a Nirvana DataFrame's lineage plan; inject compiled UDFs into
    the operator `tool` slots that decide() marks as CX. In place.

    eager=True compiles every injected UDF immediately (cache-aware) instead
    of on the first row; run several queries' optimize(eager=True) from a
    thread pool to overlap compile latency across queries."""
    report = RewriteReport()
    _injected = []

    est_n = float(_DEFAULT_ROOT_N)
    for node in _chain(df.leaf_node):
        op_name = getattr(node, "op_name", "")
        op = node.operator

        if op_name == "scan":
            ds = getattr(node, "datasource", None)
            if ds is not None:
                est_n = float(len(ds))
            continue

        if op_name == "filter":
            instruction = op.user_instruction or ""
            cols = list(op.input_columns or [])
            if op.has_udf():
                d = Decision(False, "user already supplied a UDF -> untouched")
            else:
                d = decide("filter", instruction, int(est_n), cols,
                           objective=objective, judge=judge)
                if d.use_cx:
                    ds = _scan_datasource(node.left_child)
                    smp = (ds[cols].sample(min(12, len(ds)), random_state=42).to_dict("records")
                           if ds is not None and all(c in ds.columns for c in cols) else [])
                    op.tool = FunctionCallTool.from_function(
                        func=_LazyCompiledFilter(instruction, cols, samples=smp),
                        name="cx_compiled_filter",
                    )
                    _injected.append(op.tool.func)
            report.decisions.append(("filter", instruction[:80], int(est_n), d))
            est_n *= _FILTER_SELECTIVITY
            continue

        if op_name == "map":
            instruction = op.user_instruction or ""
            cols = list(op.input_columns or [])
            out_cols = list(getattr(op, "output_columns", None) or [])
            if op.has_udf():
                d = Decision(False, "user already supplied a UDF -> untouched")
            else:
                d = decide("map", instruction, int(est_n), cols,
                           objective=objective, judge=judge)
                if d.use_cx:
                    ds = _scan_datasource(node.left_child)
                    smp = (ds[cols].sample(min(12, len(ds)), random_state=42).to_dict("records")
                           if ds is not None and all(c in ds.columns for c in cols) else [])
                    op.tool = FunctionCallTool.from_function(
                        func=_LazyCompiledMap(instruction, cols, out_cols, samples=smp),
                        name="cx_compiled_map",
                    )
                    _injected.append(op.tool.func)
            report.decisions.append(("map", instruction[:80], int(est_n), d))
            continue

        if op_name == "join":
            instruction = op.user_instruction or ""
            left_on = list(getattr(op, "left_on", None) or [])
            right_on = list(getattr(op, "right_on", None) or [])
            left_ds = _scan_datasource(node.left_child)
            right_ds = _scan_datasource(node.right_child)
            ln = len(left_ds) if left_ds is not None else _DEFAULT_ROOT_N
            rn = len(right_ds) if right_ds is not None else _DEFAULT_ROOT_N
            pairs = int(ln * rn)

            if op.has_udf():
                d = Decision(False, "user already supplied a UDF -> untouched")
            else:
                d = decide("join", instruction, pairs, left_on + right_on,
                           objective=objective, judge=judge)
                if d.use_cx:
                    # FULL-ROW visibility: nirvana feeds the UDF only the
                    # left_on/right_on columns. We WIDEN them to all columns
                    # so the compiled fn (and refine) can see structured
                    # signal columns like scoreSentiment. Safe because join
                    # key mapping uses positional indices, not left_on values.
                    if left_ds is not None:
                        op.left_on = list(left_ds.columns)
                    if right_ds is not None:
                        op.right_on = list(right_ds.columns)

                    def _full_samples(ds):
                        if ds is None:
                            return []
                        return ds.sample(min(10, len(ds)), random_state=42).to_dict("records")

                    op.tool = FunctionCallTool.from_function(
                        func=_LazyCompiledJoin(instruction,
                                               _full_samples(left_ds),
                                               _full_samples(right_ds)),
                        name="cx_compiled_join",
                    )
                    _injected.append(op.tool.func)
            report.decisions.append(("join", instruction[:80], pairs, d))
            continue

        if op_name in ("rank", "reduce"):
            d = Decision(False, f"nirvana {op_name} injection not yet implemented -> native")
            report.decisions.append((op_name, str(getattr(op, "user_instruction", ""))[:80], int(est_n), d))
            continue

    if eager:
        # within-query parallelism: compile the query's injected UDFs concurrently
        # via the shared backend-agnostic scheduler (sembaker.core.scheduler).
        from sembaker.core import scheduler

        def _compile_one(w):
            try:
                w.ensure_compiled()
            except Exception as e:
                print(f'[eager] compile failed ({type(e).__name__}: {e}); will lazy-retry at run time')
                w.artifact = None

        scheduler.run_map(_compile_one, _injected, label="nirvana-eager")
    if verbose:
        print(report)
    return report
