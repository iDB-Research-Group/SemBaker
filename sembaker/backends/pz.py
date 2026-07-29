"""PZ backend for the external optimizer: plan walk + decision + rule gating.

Usage (user keeps writing NATIVE PZ APIs — nothing in their pipeline changes):

    import sembaker.backends.pz as cxpz

    ds = (pz.MemoryDataset(id="reviews", vals=df)
            .sem_filter("the review is clearly positive")
            .sem_map(cols=[...]))
    report = cxpz.optimize(ds)          # walk plan, decide per op, install gates
    out = ds.run(config)                # PZ executes the rewritten plan

How the "rewrite" works
-----------------------
PZ chooses each logical op's physical implementation through
IMPLEMENTATION_RULES at run() time. We do not mutate the user's Dataset
objects; instead `optimize()`:

  1. Walks the Dataset DAG bottom-up, estimating cardinality at each node
     (records for filter/map, candidate pairs for join).
  2. Calls sembaker.optimizer.decide() per semantic op and stores the Decision in
     a registry keyed by the op's semantic fingerprint (op kind + predicate
     text — stable across Dataset.copy()).
  3. Replaces the relevant rules in IMPLEMENTATION_RULES with *gated*
     subclasses: for an op with a registered decision, ONLY the chosen
     side's rules (cx_* or native LLM) are allowed to match. Ops without a
     decision (or non-semantic ops) keep PZ's default behavior.

This is exactly the dual-API design at plan level: the user's sem_filter
stays sem_filter in their code; the optimizer flips which physical operator
implements it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

try:
    import palimpzest.query.optimizer as opt
except ImportError as e:
    raise ImportError(
        "The PZ backend requires the 'palimpzest' package: "
        "pip install palimpzest  (or: pip install sembaker[pz])"
    ) from e
from palimpzest.constants import NAIVE_EST_FILTER_SELECTIVITY, NAIVE_EST_JOIN_SELECTIVITY
from palimpzest.query.operators.logical import (
    Aggregate,
    ConvertScan,
    FilteredScan,
    GroupByAggregate,
    JoinOp as LogicalJoinOp,
    LimitScan,
)

from sembaker.optimizer.decision import Decision, decide

logger = logging.getLogger(__name__)

# Default cardinality when a root dataset's size cannot be determined.
_DEFAULT_ROOT_N = 100

# Native LLM-driven rules that compete with our cx rules for the same
# logical patterns. When a decision exists for an op, the losing side's
# rules are gated off FOR THAT OP ONLY.
_NATIVE_LLM_RULE_NAMES = {
    "filter": {"LLMFilterRule", "RAGRule", "MixtureOfAgentsRule", "CritiqueAndRefineRule", "SplitRule"},
    "map": {"LLMConvertBondedRule", "RAGRule", "MixtureOfAgentsRule", "CritiqueAndRefineRule", "SplitRule"},
    "join": {"NestedLoopsJoinRule", "EmbeddingJoinRule"},
}
_CX_RULE_NAMES = {
    "filter": {"CompiledFilterRule"},
    "map": {"CompiledMapRule"},
    "join": {"CompiledNestedLoopsJoinRule"},
}

# fingerprint -> Decision, consulted by the gated rules at run() time.
_REGISTRY: dict[tuple[str, str], Decision] = {}
# original IMPLEMENTATION_RULES content, for reset().
_SAVED_RULES: list | None = None


# ---- fingerprinting ---------------------------------------------------------


def _convert_instruction(op: ConvertScan) -> str:
    """Stable text fingerprint for a semantic map: the generated fields'
    names + descriptions (the same intent text CompiledMap compiles)."""
    in_fields = set(op.input_schema.model_fields) if op.input_schema is not None else set()
    gen = {
        name: (f.description or "")
        for name, f in op.output_schema.model_fields.items()
        if name not in in_fields
    }
    return json.dumps(gen, sort_keys=True, ensure_ascii=False)


def _fingerprint(logical_op) -> tuple[str, str] | None:
    """(kind, predicate_text) for semantic ops; None for everything else."""
    if isinstance(logical_op, FilteredScan) and logical_op.filter.filter_fn is None:
        return ("filter", logical_op.filter.filter_condition)
    if isinstance(logical_op, ConvertScan) and logical_op.udf is None:
        return ("map", _convert_instruction(logical_op))
    if isinstance(logical_op, LogicalJoinOp) and not logical_op.on:
        return ("join", logical_op.condition)
    return None


# ---- cardinality walker -----------------------------------------------------


def _estimate(node, memo: dict, plan_ops: list) -> float:
    """Bottom-up cardinality estimate. Appends (fingerprint, est_n, fields)
    for each semantic op to plan_ops (est_n = pairs for joins)."""
    key = id(node)
    if key in memo:
        return memo[key]

    if node.is_root:
        try:
            n = float(len(node))
        except Exception:
            n = float(_DEFAULT_ROOT_N)
        memo[key] = n
        return n

    src_ns = [_estimate(s, memo, plan_ops) for s in node._sources]
    op = node._operator
    n_in = src_ns[0] if src_ns else float(_DEFAULT_ROOT_N)

    fp = _fingerprint(op)
    fields = list(op.input_schema.model_fields) if op.input_schema is not None else []

    if isinstance(op, FilteredScan):
        if fp is not None:
            plan_ops.append((fp, int(n_in), fields))
        n_out = n_in * NAIVE_EST_FILTER_SELECTIVITY
    elif isinstance(op, ConvertScan):
        if fp is not None:
            plan_ops.append((fp, int(n_in), fields))
        n_out = n_in
    elif isinstance(op, LogicalJoinOp):
        pairs = src_ns[0] * (src_ns[1] if len(src_ns) > 1 else float(_DEFAULT_ROOT_N))
        if fp is not None:
            plan_ops.append((fp, int(pairs), fields))
        n_out = pairs * NAIVE_EST_JOIN_SELECTIVITY
    elif isinstance(op, LimitScan):
        n_out = min(n_in, float(op.limit))
    elif isinstance(op, (Aggregate, GroupByAggregate)):
        n_out = 1.0
    else:
        n_out = n_in

    memo[key] = n_out
    return n_out


# ---- rule gating ------------------------------------------------------------


def _gate_side(rule_name: str) -> tuple[str, str] | None:
    """Which (kind, side) a rule belongs to, or None if never gated."""
    for kind, names in _CX_RULE_NAMES.items():
        if rule_name in names:
            return (kind, "cx")
    for kind, names in _NATIVE_LLM_RULE_NAMES.items():
        if rule_name in names:
            return (kind, "native")
    return None


def _make_gated(rule_cls, side: str):
    """Subclass `rule_cls` so matches_pattern consults the decision registry."""

    class Gated(rule_cls):
        @classmethod
        def matches_pattern(cls, logical_expression):
            if not super().matches_pattern(logical_expression):
                return False
            fp = _fingerprint(logical_expression.operator)
            if fp is None:
                return True
            d = _REGISTRY.get(fp)
            if d is None:
                return True  # no decision -> PZ default (cost-based) behavior
            return d.use_cx if side == "cx" else (not d.use_cx)

    Gated.__name__ = rule_cls.__name__  # keep diagnostics readable
    Gated.__qualname__ = rule_cls.__qualname__
    return Gated


def _install_gates() -> None:
    """Swap gateable rules in IMPLEMENTATION_RULES for gated subclasses.
    Idempotent; remembers the originals for reset()."""
    global _SAVED_RULES
    if _SAVED_RULES is None:
        _SAVED_RULES = list(opt.IMPLEMENTATION_RULES)

    new_rules = []
    for rule in opt.IMPLEMENTATION_RULES:
        if getattr(rule, "_cx_gated", False):
            new_rules.append(rule)
            continue
        gs = _gate_side(rule.__name__)
        if gs is None:
            new_rules.append(rule)
            continue
        gated = _make_gated(rule, gs[1])
        gated._cx_gated = True
        new_rules.append(gated)
    opt.IMPLEMENTATION_RULES[:] = new_rules


def reset() -> None:
    """Clear all decisions and restore the original (ungated) rules."""
    global _SAVED_RULES
    _REGISTRY.clear()
    if _SAVED_RULES is not None:
        opt.IMPLEMENTATION_RULES[:] = _SAVED_RULES
        _SAVED_RULES = None


# ---- public API -------------------------------------------------------------


@dataclass
class RewriteReport:
    decisions: list[tuple[str, str, int, Decision]] = field(default_factory=list)
    # (kind, predicate[:80], est_n, decision)

    def __str__(self):
        lines = ["=== sembaker.optimizer plan rewrite ==="]
        for kind, pred, n, d in self.decisions:
            arrow = "-> CX " if d.use_cx else "-> native"
            lines.append(f"  [{kind:>6}] N~{n:<8} {arrow}  {pred!r}")
            lines.append(f"           reason: {d.reason}")
        return "\n".join(lines)


def optimize(
    dataset,
    *,
    objective: str = "wall",
    judge: str = "heuristic",
    verbose: bool = True,
) -> RewriteReport:
    """Walk a PZ Dataset plan, decide each semantic op, install rule gates.

    Must be called after sembaker.backends.pz_ops.install() (additive mode). The user's
    Dataset is not mutated; decisions take effect when .run() builds the
    physical plan.
    """
    # ensure our cx rules are registered (additive — native rules stay)
    import sembaker.backends.pz_ops

    sembaker.backends.pz_ops.install()

    plan_ops: list[tuple[tuple[str, str], int, list[str]]] = []
    _estimate(dataset, {}, plan_ops)

    report = RewriteReport()
    for fp, est_n, fields in plan_ops:
        kind, pred = fp
        d = decide(kind, pred, est_n, fields, objective=objective, judge=judge)
        _REGISTRY[fp] = d
        report.decisions.append((kind, pred[:80], est_n, d))

    _install_gates()

    if verbose:
        print(report)
    return report
