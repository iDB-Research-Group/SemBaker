"""Compile-then-execute physical operators for Palimpzest, registered at runtime.

PZ has no official plugin API, so we ship our operators + rules as a separate
package and `install()` them by mutating PZ's `IMPLEMENTATION_RULES` list in
place. This keeps PZ's source tree untouched and our research code in its own
git repo.

Currently provides three operator/rule pairs:
  - CompiledFilter           / CompiledFilterRule
  - CompiledNestedLoopsJoin  / CompiledNestedLoopsJoinRule
  - CompiledMap              / CompiledMapRule

Usage (default — additive; optimizer chooses based on cost):
    import sembaker.backends.pz_ops
    sembaker.backends.pz_ops.install()

Usage (forced — use Compiled* on every sem_filter / sem_join / sem_map, for A/B):
    sembaker.backends.pz_ops.install(force_compiled=True)
"""

import os as _os

# Force litellm (pulled in by palimpzest) to use its bundled local model-cost
# map instead of fetching it from GitHub at import time. The remote fetch is
# flaky on some networks (ConnectionReset) and, on failure, PZ's default Model
# registry construction crashes building a Together/Llama model. Setting this
# BEFORE the palimpzest import below immunizes every PZ run. Does not affect
# our pinned OpenAI models (their pricing comes from PZ's metrics_manager).
_os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

try:
    import palimpzest as _palimpzest  # noqa: F401
except ImportError as e:
    raise ImportError(
        "sembaker.backends.pz_ops requires the 'palimpzest' package: "
        "pip install palimpzest  (or: pip install sembaker[pz])"
    ) from e

from sembaker.backends.pz_ops.compiled_filter import CompiledFilter, CompiledFilterRule
from sembaker.backends.pz_ops.compiled_join import CompiledNestedLoopsJoin, CompiledNestedLoopsJoinRule
from sembaker.backends.pz_ops.compiled_map import CompiledMap, CompiledMapRule

# Names of all PZ-built-in LLM-driven rules that should be removed when
# force_compiled=True (so the optimizer is FORCED to use our compiled ops).
# We intentionally do NOT remove RelationalJoinRule (deterministic equi-join,
# matches when sem_join is given an `on=` arg with empty condition) or
# AggregateRule (dispatches to count/sum/avg/etc. for non-LLM aggregates).
_BUILTIN_LLM_RULE_NAMES = {
    # filter
    "LLMFilterRule",
    "NonLLMFilterRule",
    # join
    "NestedLoopsJoinRule",
    "EmbeddingJoinRule",
    # map / convert
    "LLMConvertBondedRule",
    # NOTE: SemanticAggregateRule is intentionally NOT stripped — we no longer
    # ship a CompiledAggregate, so sem_agg must keep falling back to PZ's
    # native semantic aggregate even under force_compiled.
}

_OUR_RULES = (
    CompiledFilterRule,
    CompiledNestedLoopsJoinRule,
    CompiledMapRule,
)


def install(force_compiled: bool = False) -> None:
    """Register all Compiled* rules into PZ's optimizer.

    Mutates `palimpzest.query.optimizer.IMPLEMENTATION_RULES` in place so
    that optimizer.py's bound reference to that list also picks up the
    change.

    Args:
        force_compiled: If True, ALSO removes PZ's built-in LLM-driven
            filter/join/aggregate rules (LLMFilterRule, NonLLMFilterRule,
            NestedLoopsJoinRule, EmbeddingJoinRule, SemanticAggregateRule)
            so the optimizer is forced to substitute every semantic op with
            its compile-then-execute counterpart. Use for fair A/B
            comparison runs where small cardinalities would otherwise let
            the cost-based optimizer keep picking the LLM-per-record op.

    Idempotent: safe to call multiple times.
    """
    import palimpzest.query.optimizer as opt

    for rule in _OUR_RULES:
        if rule not in opt.IMPLEMENTATION_RULES:
            opt.IMPLEMENTATION_RULES.append(rule)

    if force_compiled:
        # In-place removal so optimizer.py's bound reference sees the change
        opt.IMPLEMENTATION_RULES[:] = [
            r for r in opt.IMPLEMENTATION_RULES
            if r.__name__ not in _BUILTIN_LLM_RULE_NAMES
        ]


def uninstall() -> None:
    """Restore PZ's optimizer to its original state.

    Removes our rules and re-adds any built-in LLM rules that may have
    been stripped by `install(force_compiled=True)`. Idempotent.
    """
    import palimpzest.query.optimizer as opt
    from palimpzest.query.optimizer.rules import (
        EmbeddingJoinRule,
        LLMConvertBondedRule,
        LLMFilterRule,
        NestedLoopsJoinRule,
        NonLLMFilterRule,
        SemanticAggregateRule,
    )

    for rule in _OUR_RULES:
        if rule in opt.IMPLEMENTATION_RULES:
            opt.IMPLEMENTATION_RULES.remove(rule)
    for builtin in (LLMFilterRule, NonLLMFilterRule, NestedLoopsJoinRule, EmbeddingJoinRule,
                    SemanticAggregateRule, LLMConvertBondedRule):
        if builtin not in opt.IMPLEMENTATION_RULES:
            opt.IMPLEMENTATION_RULES.append(builtin)


__all__ = [
    "install",
    "uninstall",
    "CompiledFilter",
    "CompiledFilterRule",
    "CompiledNestedLoopsJoin",
    "CompiledNestedLoopsJoinRule",
    "CompiledMap",
    "CompiledMapRule",
]
