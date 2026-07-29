"""Lotus backend for the external optimizer: LazyFrame node rewriting.

Usage (user keeps writing NATIVE Lotus APIs):

    from lotus.ast import LazyFrame
    import sembaker.backends.lotus as cxlotus

    lf = LazyFrame(df).sem_filter("The {reviewText} is clearly positive.")
    report = cxlotus.optimize(lf)       # walk nodes, decide, swap in place
    out = lf.run()                       # Lotus executes the rewritten plan

How the "rewrite" works
-----------------------
Lotus LazyFrame plans are plain node lists (lotus.ast.nodes.BaseNode
subclasses with a __call__(df) -> df contract). The rewriter walks
lf._nodes, estimates cardinality from the bound SourceNode, calls
sembaker.optimizer.decide() per semantic node, and REPLACES the node object
with a CX node when use_cx=True:

    SemFilterNode(user_instruction=...)  ->  CXFilterNode(instruction=...)
    SemMapNode(user_instruction=...)     ->  CXMapNode(instruction=...)

CX nodes compile lazily on first execution (one sembaker.core LLM call), then
evaluate the compiled Python function row-locally — the rewritten plan
never touches lotus.settings.lm. SemJoinNode rewriting is not yet
implemented (reported as native in the rewrite report).

Also exposes CXRewriteOptimizer (a lotus BaseOptimizer) so the same pass
can ride Lotus's own optimizer pipeline:

    lf.optimize(optimizers=[cxlotus.CXRewriteOptimizer()])
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from pydantic import ConfigDict

try:
    from lotus.ast.nodes import BaseNode, SemFilterNode, SemJoinNode, SemMapNode, SourceNode
    from lotus.ast.optimizer.base import BaseOptimizer
except ImportError as e:
    raise ImportError(
        "The LOTUS backend requires the 'lotus-ai' package: "
        "pip install lotus-ai  (or: pip install sembaker[lotus])"
    ) from e

from sembaker.core import (
    CompiledArtifact,
    build_filter_prompt,
    build_join_prompt,
    build_map_prompt,
    compile_artifact,
    make_key,
)
from sembaker.optimizer.decision import Decision, decide
from sembaker.optimizer.refine_gate import maybe_refine, maybe_validator

# Model used for the one-shot compile call (OpenAI, direct — independent of
# whatever LM the user configured in lotus.settings).
COMPILE_MODEL_ID = os.getenv("CX_COMPILE_MODEL", "gpt-5-mini-2025-08-07")

_DEFAULT_ROOT_N = 100
_FILTER_SELECTIVITY = 0.5


def _kept_cols(cols: list[str]) -> list[str]:
    """Experiment hook: CX_LOTUS_KEEP_COLS="reviewText,..." restricts the
    columns Lotus filter/map compile against (cache key + samples) to the
    listed ones. Off by default -> Lotus uses full rows (its full-row
    visibility is what lets joins find scoreSentiment). When set to the same
    column(s) PZ/Nirvana feed (e.g. reviewText), Lotus's filter/map cache key
    matches theirs -> the three backends share one cached artifact."""
    keep = os.environ.get("CX_LOTUS_KEEP_COLS")
    if not keep:
        return cols
    wanted = [c.strip() for c in keep.split(",") if c.strip()]
    restricted = [c for c in cols if c in wanted]
    return restricted or cols


# ---- CX nodes ---------------------------------------------------------------


class CXFilterNode(BaseNode):
    """Compile-then-execute replacement for SemFilterNode.

    Compiles `instruction` into `compiled_filter(row) -> bool` on first
    execution (ONE LLM call), then filters the DataFrame row-locally.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instruction: str
    compile_model_id: str = COMPILE_MODEL_ID
    artifact: Any = None  # CompiledArtifact, populated on first call

    def __call__(self, df: pd.DataFrame, resolver=None, **context: Any) -> pd.DataFrame:  # type: ignore[override]
        if self.artifact is None:
            from sembaker.optimizer.compile_op import compile_operator
            cols = _kept_cols(list(df.columns))
            samples = df[cols].head(10).to_dict("records")
            self.artifact = compile_operator("filter", self.instruction, self.compile_model_id,
                                             cols=cols, samples=samples, tag="CXFilterNode")
        fn = self.artifact.fn

        def _safe(row) -> bool:
            try:
                return bool(fn(row.to_dict()))
            except Exception:
                return False

        mask = df.apply(_safe, axis=1)
        return df[mask]

    def signature(self) -> str:
        return f"cx_filter({self.instruction[:40]!r})"


class CXMapNode(BaseNode):
    """Compile-then-execute replacement for SemMapNode.

    Compiles `instruction` into `compiled_map(row) -> value` on first
    execution, then assigns the result to `out_col` row-locally.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instruction: str
    out_col: str = "_map"
    compile_model_id: str = COMPILE_MODEL_ID
    artifact: Any = None

    def __call__(self, df: pd.DataFrame, resolver=None, **context: Any) -> pd.DataFrame:  # type: ignore[override]
        if self.artifact is None:
            from sembaker.optimizer.compile_op import compile_operator
            cols = _kept_cols(list(df.columns))
            samples = df[cols].head(10).to_dict("records")
            self.artifact = compile_operator("map", self.instruction, self.compile_model_id,
                                             cols=cols, samples=samples, tag="CXMapNode")
        fn = self.artifact.fn

        def _safe(row):
            try:
                v = fn(row.to_dict())
                return "" if v is None else v
            except Exception:
                return ""

        out = df.copy()
        out[self.out_col] = df.apply(_safe, axis=1)
        return out

    def signature(self) -> str:
        return f"cx_map({self.instruction[:40]!r})"


class CXJoinNode(SemJoinNode):
    """Compile-then-execute replacement for SemJoinNode (inner joins).

    Subclasses SemJoinNode so the right-side reference plumbing
    (_JoinMixin: right_df / right_lf / right_source_node + resolver) is
    inherited unchanged. On first execution, compiles the join predicate
    with K random sample rows from EACH side (so the compiler can spot
    join-key value overlap), then evaluates pairs locally.

    Output follows Lotus's sem_join column convention: columns present on
    both sides are renamed `col:left` / `col:right`.
    """

    compile_model_id: str = COMPILE_MODEL_ID
    n_compile_samples: int = 10
    artifact: Any = None

    def __call__(self, df: pd.DataFrame, resolver=None, **context: Any) -> pd.DataFrame:  # type: ignore[override]
        import random

        right = self._resolve_right(resolver)
        if self.how != "inner":
            raise NotImplementedError("CXJoinNode supports inner joins only")

        if self.artifact is None:
            from sembaker.optimizer.compile_op import compile_operator
            ls = df.sample(min(self.n_compile_samples, len(df)), random_state=42).to_dict("records")
            rs = right.sample(min(self.n_compile_samples, len(right)), random_state=42).to_dict("records")
            self.artifact = compile_operator("join", self.join_instruction, self.compile_model_id,
                                             samples_left=ls, samples_right=rs, tag="CXJoinNode")
        fn = self.artifact.fn

        # lotus sem_join convention: overlap columns -> col:left / col:right
        overlap = [c for c in df.columns if c in right.columns]
        lren = {c: f"{c}:left" for c in overlap}
        rren = {c: f"{c}:right" for c in overlap}

        rows = []
        right_records = right.to_dict("records")
        for _, lrow in df.iterrows():
            ld = lrow.to_dict()
            for rd in right_records:
                try:
                    ok = bool(fn(ld, rd))
                except Exception:
                    ok = False
                if ok:
                    merged = {lren.get(k, k): v for k, v in ld.items()}
                    merged.update({rren.get(k, k): v for k, v in rd.items()})
                    rows.append(merged)
        return pd.DataFrame(rows)

    def signature(self) -> str:
        return f"cx_join({self.join_instruction[:40]!r})"


# ---- rewrite pass -----------------------------------------------------------


@dataclass
class RewriteReport:
    decisions: list[tuple[str, str, int, Decision]] = field(default_factory=list)

    def __str__(self):
        lines = ["=== sembaker.optimizer (lotus) plan rewrite ==="]
        for kind, pred, n, d in self.decisions:
            arrow = "-> CX " if d.use_cx else "-> native"
            lines.append(f"  [{kind:>6}] N~{n:<8} {arrow}  {pred!r}")
            lines.append(f"           reason: {d.reason}")
        return "\n".join(lines)


def rewrite_nodes(
    nodes: list[BaseNode],
    *,
    objective: str = "wall",
    judge: str = "heuristic",
    report: RewriteReport | None = None,
) -> list[BaseNode]:
    """Walk a Lotus node list; return a copy with decided CX swaps applied."""
    report = report if report is not None else RewriteReport()

    # cardinality estimate flows top-down through the node list
    est_n = float(_DEFAULT_ROOT_N)
    fields: list[str] = []
    out: list[BaseNode] = []

    for node in nodes:
        if isinstance(node, SourceNode):
            if node.df is not None:
                est_n = float(len(node.df))
                fields = list(node.df.columns)
            out.append(node)
            continue

        if isinstance(node, SemFilterNode):
            d = decide("filter", node.user_instruction, int(est_n), fields,
                       objective=objective, judge=judge)
            report.decisions.append(("filter", node.user_instruction[:80], int(est_n), d))
            if d.use_cx:
                out.append(CXFilterNode(instruction=node.user_instruction))
            else:
                out.append(node)
            est_n *= _FILTER_SELECTIVITY
            continue

        if isinstance(node, SemMapNode):
            d = decide("map", node.user_instruction, int(est_n), fields,
                       objective=objective, judge=judge)
            report.decisions.append(("map", node.user_instruction[:80], int(est_n), d))
            if d.use_cx:
                out.append(CXMapNode(instruction=node.user_instruction, out_col=node.suffix))
            else:
                out.append(node)
            continue

        if isinstance(node, SemJoinNode) and not isinstance(node, CXJoinNode):
            # estimate candidate pairs = left N x right N (right side from
            # whichever reference the join node carries)
            right_n = float(_DEFAULT_ROOT_N)
            if node.right_df is not None:
                right_n = float(len(node.right_df))
            elif node.right_lf is not None and getattr(node.right_lf, "_source", None) is not None \
                    and node.right_lf._source.df is not None:
                right_n = float(len(node.right_lf._source.df))
            pairs = int(est_n * right_n)

            d = decide("join", node.join_instruction, pairs, fields,
                       objective=objective, judge=judge)
            if d.use_cx and node.how != "inner":
                d = Decision(False, f"cx join supports inner only (how={node.how}) -> native")
            report.decisions.append(("join", node.join_instruction[:80], pairs, d))
            if d.use_cx:
                out.append(CXJoinNode(
                    join_instruction=node.join_instruction,
                    how=node.how,
                    suffix=node.suffix,
                    right_source_node=node.right_source_node,
                    right_lf=node.right_lf,
                    right_df=node.right_df,
                ))
            else:
                out.append(node)
            continue

        out.append(node)

    return out


def _smoke_ok(fn, sample_dicts, pairwise=False, right_dicts=None) -> bool:
    """Artifact sanity check: the compiled fn must run without raising on at
    least one sample. (Catches truncated/broken generations before caching
    them or putting them on the execution path.)"""
    ok = 0
    try:
        if pairwise:
            for ld in sample_dicts[:5]:
                for rd in (right_dicts or [])[:5]:
                    fn(ld, rd)
                    ok += 1
        else:
            for d in sample_dicts[:10]:
                fn(d)
                ok += 1
    except Exception:
        return ok > 0
    return ok > 0 or not sample_dicts


def _compile_join_eager(node, samples, verbose: bool) -> None:
    """Eagerly compile one CXJoinNode against sample rows from both sides."""
    right = node.right_df
    if right is None and node.right_lf is not None \
            and getattr(node.right_lf, "_source", None) is not None:
        right = node.right_lf._source.df
    if right is None:
        return
    from sembaker.optimizer.compile_op import compile_operator
    ls = samples.sample(min(node.n_compile_samples, len(samples)), random_state=42)
    rs = right.sample(min(node.n_compile_samples, len(right)), random_state=42)
    lsd = ls.to_dict("records"); rsd = rs.to_dict("records")
    node.artifact = compile_operator("join", node.join_instruction, node.compile_model_id,
                                     samples_left=lsd, samples_right=rsd, tag="CXJoinNode(eager)")
    if not _smoke_ok(node.artifact.fn, ls.to_dict("records"),
                     pairwise=True, right_dicts=rs.to_dict("records")):
        if verbose:
            print("[eager] join artifact failed smoke check -> dropping (will lazy-recompile)")
        node.artifact = None


def _eager_compile(nodes, verbose: bool = True) -> None:
    """Compile all CX nodes of one plan via the shared scheduler, so a query's
    filter/map/join compiles run CONCURRENTLY (within-query parallelism).

    Each node draws its OWN sample rows from the SOURCE (we drop the old
    sample-propagation so the compiles are independent and parallelizable):
    compilation only needs example rows, not the upstream operator's output.
    The scheduler (sembaker.core.scheduler) owns the parallelism — the same
    entry point a warm phase uses for cross-query concurrency."""
    from sembaker.core import scheduler

    src = next((n.df for n in nodes
                if isinstance(n, SourceNode) and n.df is not None), None)
    if src is None:
        return
    samples = src.sample(min(10, len(src)), random_state=42)

    tasks = []
    for node in nodes:
        if isinstance(node, (CXFilterNode, CXMapNode)):
            tasks.append(lambda n=node: n(samples))          # __call__ triggers the compile
        elif isinstance(node, CXJoinNode):
            tasks.append(lambda n=node: _compile_join_eager(n, samples, verbose))
    scheduler.run(tasks, label="lotus-eager")


def optimize(
    lf,
    *,
    objective: str = "wall",
    judge: str = "heuristic",
    eager: bool = False,
    verbose: bool = True,
) -> RewriteReport:
    """Rewrite a Lotus LazyFrame IN PLACE (lf._nodes is replaced).

    eager=True additionally compiles every CX node immediately, with the query's
    operators compiled CONCURRENTLY via sembaker.core.scheduler (within-query
    parallelism). Several queries' optimize(eager=True) can themselves be run
    through the scheduler for cross-query concurrency."""
    report = RewriteReport()
    lf._nodes = rewrite_nodes(lf._nodes, objective=objective, judge=judge, report=report)
    if eager:
        _eager_compile(lf._nodes, verbose=verbose)
    if verbose:
        print(report)
    return report


class CXRewriteOptimizer(BaseOptimizer):
    """The same pass packaged as a Lotus BaseOptimizer, so it can ride
    lf.optimize(optimizers=[...]) alongside Lotus's own passes."""

    requires_train_data = False

    def __init__(self, objective: str = "wall", judge: str = "heuristic", verbose: bool = True):
        self.objective = objective
        self.judge = judge
        self.verbose = verbose

    def optimize(self, nodes, train_data=None):
        report = RewriteReport()
        new_nodes = rewrite_nodes(nodes, objective=self.objective, judge=self.judge, report=report)
        if self.verbose:
            print(report)
        return new_nodes
