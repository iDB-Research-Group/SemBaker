"""CompiledFilter: PZ-compatible compile-then-execute physical operator.

Pattern: a single LLM call generates a deterministic Python function for the
operator's predicate; the function is cached on the operator instance and
applied to every record locally — no per-record LLM call.

This module subclasses PZ's FilterOp and ImplementationRule so it integrates
with PZ's optimizer/cost model, but it lives outside the PZ source tree.
Register it via `sembaker.backends.pz_ops.install()`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from sembaker.core import build_filter_prompt, compile_artifact, make_key

from palimpzest.constants import (
    NAIVE_EST_FILTER_SELECTIVITY,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats, OperatorCostEstimates
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.logical import FilteredScan
from palimpzest.query.optimizer.primitives import LogicalExpression, PhysicalExpression
from palimpzest.query.optimizer.rules import ImplementationRule


logger = logging.getLogger(__name__)


# Heuristic estimates used to amortize the one-shot compile cost across input
# cardinality in the cost model. Tune from empirical runs.
_COMPILE_COST_USD_ESTIMATE = 0.005
_COMPILE_TIME_SEC_ESTIMATE = 5.0


class CompiledFilter(FilterOp):
    """Compile-then-execute physical implementation of sem_filter.

    Workflow:
      1. On first __call__, issue ONE LLM request that translates the natural-
         language predicate `filter_obj.filter_condition` into a deterministic
         Python function `compiled_filter(row) -> bool`.
      2. Cache the function on the operator instance.
      3. For every subsequent record, evaluate the cached function locally —
         no LLM call in the per-record path.

    Cost model amortizes the compile cost across input cardinality so the
    optimizer prefers this op over LLMFilter once N is more than a handful.
    """

    # gpt-5 family forces temperature=1; older models accept any.
    _COMPILE_TEMPERATURE = 1

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.FILTER,
        reasoning_effort: str = "default",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort

        # Lazily populated on first __call__
        self._compiled_fn: Callable | None = None
        self._compile_code: str | None = None
        self._compile_input_tokens: int = 0
        self._compile_output_tokens: int = 0
        self._compile_cost_usd: float = 0.0
        self._compile_duration_secs: float = 0.0
        # PZ runs filter in a ThreadPoolExecutor; without a lock all threads
        # race past the `if self._compiled_fn is None` check and trigger
        # redundant LLM compile calls.
        self._compile_lock = threading.Lock()

    # ----- PZ bookkeeping plumbing --------------------------------------

    def get_id_params(self):
        id_params = super().get_id_params()
        return {
            "model": None if self.model is None else self.model.value,
            "prompt_strategy": None if self.prompt_strategy is None else self.prompt_strategy.value,
            "reasoning_effort": self.reasoning_effort,
            **id_params,
        }

    def get_op_params(self):
        op_params = super().get_op_params()
        return {
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "reasoning_effort": self.reasoning_effort,
            **op_params,
        }

    def get_model_name(self):
        return None if self.model is None else self.model.value

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates):
        """Per-record exec is ~µs; compile cost is amortized across N records."""
        n = max(1.0, source_op_cost_estimates.cardinality)
        time_per_record_exec = 0.001  # 1 ms upper bound for compiled fn call
        time_per_record = time_per_record_exec + (_COMPILE_TIME_SEC_ESTIMATE / n)
        cost_per_record = _COMPILE_COST_USD_ESTIMATE / n

        cardinality = NAIVE_EST_FILTER_SELECTIVITY * source_op_cost_estimates.cardinality
        # Empirical quality from sem_filter eval: ~0.91 acc with current prompt.
        quality = 0.91

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=cost_per_record,
            quality=quality,
        )

    # ----- compile-then-execute core ------------------------------------

    def _do_compile(self) -> None:
        """One-shot compile via sembaker.core. Populates self._compiled_fn and bookkeeping."""
        # PZ model ids carry a provider prefix (e.g. "openai/gpt-4o-mini-...");
        # the OpenAI Python client expects the bare model name.
        bare_model_id = self.model.value.split("/", 1)[-1] if "/" in self.model.value else self.model.value
        try:
            usd_in = self.model.get_usd_per_input_token()
            usd_out = self.model.get_usd_per_output_token()
        except Exception:
            usd_in = usd_out = 0.0

        # PZ streams records, so the operator has no sample batch here. A warm
        # phase registers sample rows by op kind; the shared dispatcher does
        # refine/validate/cache and the CX_COMPILE method (e2e/ir1/ir2) using
        # THIS op's own intent + columns.
        from sembaker.optimizer.compile_op import compile_operator
        from sembaker.optimizer.refine_gate import lookup_samples

        intent = self.filter_obj.filter_condition
        samples = lookup_samples("filter", intent)
        artifact = compile_operator(
            "filter", intent, bare_model_id,
            cols=self.get_input_fields(), samples=samples,
            usd_per_input_token=usd_in, usd_per_output_token=usd_out, tag="CompiledFilter")
        self._compiled_fn = artifact.fn
        self._compile_code = artifact.code
        self._compile_input_tokens = artifact.input_tokens
        self._compile_output_tokens = artifact.output_tokens
        self._compile_cost_usd = artifact.cost_usd
        self._compile_duration_secs = artifact.duration_secs

    def filter(self, candidate: DataRecord) -> tuple[dict[str, bool], GenerationStats]:
        # First-record path: pay the compile cost and attribute it to this
        # record's stats so PZ accounting captures it. Double-checked locking
        # so concurrent ThreadPoolExecutor workers don't all trigger compile.
        compile_charged = False
        if self._compiled_fn is None:
            with self._compile_lock:
                if self._compiled_fn is None:
                    self._do_compile()
                    compile_charged = True

        if compile_charged:
            compile_input_tokens = self._compile_input_tokens
            compile_output_tokens = self._compile_output_tokens
            compile_cost = self._compile_cost_usd
            compile_llm_time = self._compile_duration_secs
            compile_llm_calls = 1
        else:
            compile_input_tokens = 0
            compile_output_tokens = 0
            compile_cost = 0.0
            compile_llm_time = 0.0
            compile_llm_calls = 0

        exec_start = time.time()
        try:
            passed = bool(self._compiled_fn(candidate.to_dict()))
        except Exception as e:
            if self.verbose:
                print(f"[CompiledFilter exec error] {e}")
            passed = False
        exec_dur = time.time() - exec_start

        stats = GenerationStats(
            fn_call_duration_secs=exec_dur,
            llm_call_duration_secs=compile_llm_time,
            input_text_tokens=compile_input_tokens,
            output_text_tokens=compile_output_tokens,
            total_llm_calls=compile_llm_calls,
            cost_per_record=compile_cost,
        )
        return {"passed_operator": passed}, stats


class CompiledFilterRule(ImplementationRule):
    """Substitute a logical FilteredScan with the compile-then-execute physical filter.

    Matches the same logical pattern as LLMFilterRule (filter_condition is set,
    filter_fn is None) so PZ's optimizer evaluates both and picks based on
    cost/quality estimates.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_match = isinstance(logical_op, FilteredScan) and logical_op.filter.filter_fn is None
        logger.debug(f"CompiledFilterRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting CompiledFilterRule for {logical_expression}")

        models = [
            model for model in runtime_kwargs["available_models"]
            if cls._model_matches_input(model, logical_expression)
        ]
        variable_op_kwargs = []
        for model in models:
            variable_op_kwargs.append({
                "model": model,
                "prompt_strategy": PromptStrategy.FILTER,
                "reasoning_effort": runtime_kwargs["reasoning_effort"],
            })

        return cls._perform_substitution(logical_expression, CompiledFilter, runtime_kwargs, variable_op_kwargs)
