"""CompiledNestedLoopsJoin: PZ-compatible compile-then-execute join operator.

Mirrors PZ's NestedLoopsJoin (which spends 1 LLM call per (left, right) pair)
but instead does ONE compile call to produce `compiled_join(row_left, row_right) -> bool`,
then evaluates that function locally on every pair.

Cost model amortizes the compile across left × right pair count, so it
beats NestedLoopsJoin once M*N ≳ 5–10 pairs.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from sembaker.core import build_join_prompt, compile_artifact, make_key

from palimpzest.constants import (
    NAIVE_EST_JOIN_SELECTIVITY,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.logical import JoinOp as LogicalJoinOp
from palimpzest.query.optimizer.primitives import LogicalExpression, PhysicalExpression
from palimpzest.query.optimizer.rules import ImplementationRule


logger = logging.getLogger(__name__)


_COMPILE_COST_USD_ESTIMATE = 0.005
_COMPILE_TIME_SEC_ESTIMATE = 5.0


class CompiledNestedLoopsJoin(JoinOp):
    """Compile-then-execute physical implementation of sem_join.

    On the first __call__, issues ONE LLM request that translates the
    natural-language `condition` into a deterministic Python function
    `compiled_join(row_left, row_right) -> bool`. The function is cached
    on the operator instance and applied to every (left, right) pair
    locally — no per-pair LLM call.
    """

    _COMPILE_TEMPERATURE = 1
    _N_COMPILE_SAMPLES = 10  # rows per side shown in compile prompt

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.JOIN,
        reasoning_effort: str = "default",
        join_parallelism: int = 64,
        retain_inputs: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            join_parallelism=join_parallelism,
            retain_inputs=retain_inputs,
            **kwargs,
        )
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
        # Concurrent ThreadPoolExecutor in __call__; need lock around compile.
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

    def naive_cost_estimates(
        self,
        left_source_op_cost_estimates: OperatorCostEstimates,
        right_source_op_cost_estimates: OperatorCostEstimates,
    ):
        """Per-pair exec is ~µs; compile cost is amortized across M*N pairs."""
        m = max(1.0, left_source_op_cost_estimates.cardinality)
        n = max(1.0, right_source_op_cost_estimates.cardinality)
        n_pairs = m * n

        time_per_record_exec = 0.001  # 1ms per pair upper bound
        time_per_record = time_per_record_exec + (_COMPILE_TIME_SEC_ESTIMATE / n_pairs)
        cost_per_record = _COMPILE_COST_USD_ESTIMATE / n_pairs

        cardinality = NAIVE_EST_JOIN_SELECTIVITY * n_pairs
        # Empirical from sem_join eval: ~0.96 acc with current prompt
        quality = 0.96

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=cost_per_record,
            quality=quality,
        )

    # ----- compile-then-execute core ------------------------------------

    def _do_compile(self, samples_left: list[dict], samples_right: list[dict]) -> None:
        """One-shot compile via sembaker.core. Populates self._compiled_fn and bookkeeping.

        See sembaker.core.build_join_prompt for why we pass K sample rows per side
        (the compiler needs value-distribution overlap to identify join keys)."""
        bare_model_id = self.model.value.split("/", 1)[-1] if "/" in self.model.value else self.model.value
        try:
            usd_in = self.model.get_usd_per_input_token()
            usd_out = self.model.get_usd_per_output_token()
        except Exception:
            usd_in = usd_out = 0.0

        # PZ join HAS samples here; the shared dispatcher does refine/validate/
        # cache and the CX_COMPILE method (e2e/ir1/ir2).
        from sembaker.optimizer.compile_op import compile_operator

        artifact = compile_operator(
            "join", self.condition, bare_model_id,
            samples_left=samples_left, samples_right=samples_right,
            usd_per_input_token=usd_in, usd_per_output_token=usd_out,
            tag="CompiledNestedLoopsJoin")
        self._compiled_fn = artifact.fn
        self._compile_code = artifact.code
        self._compile_input_tokens = artifact.input_tokens
        self._compile_output_tokens = artifact.output_tokens
        self._compile_cost_usd = artifact.cost_usd
        self._compile_duration_secs = artifact.duration_secs

    def _process_pair(
        self,
        left_candidate: DataRecord,
        right_candidate: DataRecord,
        compile_charge: dict | None,
    ) -> tuple[DataRecord, RecordOpStats]:
        """Evaluate the cached compiled_join on one pair and emit DataRecord + stats.

        compile_charge: if non-None, attribute the one-shot compile cost/time
        to THIS pair's stats (only on the very first pair).
        """
        start = time.time()

        try:
            passed = bool(self._compiled_fn(left_candidate.to_dict(), right_candidate.to_dict()))
        except Exception as e:
            if self.verbose:
                print(f"[CompiledNestedLoopsJoin exec error] {e}")
            passed = False
        exec_dur = time.time() - start

        # handle different join types (mirror PZ NestedLoopsJoin behaviour)
        if self.how == "left" and passed:
            self._left_joined_record_ids.add(left_candidate._id)
        elif self.how == "right" and passed:
            self._right_joined_record_ids.add(right_candidate._id)
        elif self.how == "outer" and passed:
            self._left_joined_record_ids.add(left_candidate._id)
            self._right_joined_record_ids.add(right_candidate._id)

        join_dr = DataRecord.from_join_parents(self.output_schema, left_candidate, right_candidate)
        join_dr._passed_operator = passed

        if compile_charge is not None:
            input_tok = compile_charge["input_tokens"]
            output_tok = compile_charge["output_tokens"]
            cost = compile_charge["cost"]
            llm_time = compile_charge["llm_time"]
            total_calls = 1
        else:
            input_tok = output_tok = total_calls = 0
            cost = 0.0
            llm_time = 0.0

        record_op_stats = RecordOpStats(
            record_id=join_dr._id,
            record_parent_ids=join_dr._parent_ids,
            record_source_indices=join_dr._source_indices,
            record_state=join_dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=exec_dur + llm_time,
            cost_per_record=cost,
            model_name=self.get_model_name(),
            join_condition=self.condition,
            input_text_tokens=input_tok,
            output_text_tokens=output_tok,
            llm_call_duration_secs=llm_time,
            fn_call_duration_secs=exec_dur,
            total_llm_calls=total_calls,
            answer={"passed_operator": passed},
            passed_operator=passed,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )
        return join_dr, record_op_stats

    def __call__(
        self,
        left_candidates: list[DataRecord],
        right_candidates: list[DataRecord],
        final: bool = False,
    ) -> tuple[DataRecordSet, int]:
        # Build the full set of (left, right) candidate pairs (mirrors PZ NestedLoopsJoin)
        join_candidates = []
        for candidate in left_candidates:
            for right_candidate in right_candidates:
                join_candidates.append((candidate, right_candidate))
            for right_candidate in self._right_input_records:
                join_candidates.append((candidate, right_candidate))
        for candidate in self._left_input_records:
            for right_candidate in right_candidates:
                join_candidates.append((candidate, right_candidate))

        # First call: compile (double-checked under lock) using several
        # distinct sample rows from each side as schema+value hints. One
        # row per side is not enough — the LLM compiler needs to see
        # value-distribution overlap to identify join keys; if the chosen
        # pair happens to be about different entities (the common case for
        # cross-table joins, especially when tables are sorted differently)
        # the LLM concludes wrongly that no shared key exists. We RANDOMLY
        # sample K distinct rows from each side so the LLM gets a
        # representative slice of value distributions and can spot overlap
        # in join-key columns. Without the lock PZ's ThreadPoolExecutor
        # would race past the None check.
        import random as _random
        compile_charge = None
        if self._compiled_fn is None and join_candidates:
            with self._compile_lock:
                if self._compiled_fn is None:
                    K = self._N_COMPILE_SAMPLES
                    distinct_left, distinct_right = {}, {}
                    for (lc, rc) in join_candidates:
                        distinct_left.setdefault(id(lc), lc)
                        distinct_right.setdefault(id(rc), rc)
                    left_pool = list(distinct_left.values())
                    right_pool = list(distinct_right.values())
                    rng = _random.Random(42)  # deterministic for reproducibility
                    rng.shuffle(left_pool)
                    rng.shuffle(right_pool)
                    left_seen = [c.to_dict() for c in left_pool[:K]]
                    right_seen = [c.to_dict() for c in right_pool[:K]]
                    self._do_compile(left_seen, right_seen)
                    compile_charge = {
                        "input_tokens": self._compile_input_tokens,
                        "output_tokens": self._compile_output_tokens,
                        "cost": self._compile_cost_usd,
                        "llm_time": self._compile_duration_secs,
                    }

        # Evaluate each pair locally; charge compile cost to the first pair only
        output_records, output_record_op_stats = [], []
        if join_candidates:
            # Sequential first-pair charge so compile-charge attribution is deterministic;
            # remaining pairs run in parallel for fairness with PZ NestedLoopsJoin.
            first_pair = join_candidates[0]
            first_dr, first_stats = self._process_pair(first_pair[0], first_pair[1], compile_charge)
            output_records.append(first_dr)
            output_record_op_stats.append(first_stats)
            self.join_idx += 1

            with ThreadPoolExecutor(max_workers=self.join_parallelism) as ex:
                futures = [
                    ex.submit(self._process_pair, lc, rc, None)
                    for lc, rc in join_candidates[1:]
                ]
                for fut in as_completed(futures):
                    self.join_idx += 1
                    dr, stats = fut.result()
                    output_records.append(dr)
                    output_record_op_stats.append(stats)

        num_inputs_processed = len(join_candidates)

        if self.retain_inputs:
            self._left_input_records.extend(left_candidates)
            self._right_input_records.extend(right_candidates)

        if final:
            return self._compute_unmatched_records(), 0

        if not output_records:
            return DataRecordSet([], []), num_inputs_processed
        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed


class CompiledNestedLoopsJoinRule(ImplementationRule):
    """Substitute a logical JoinOp (with semantic condition) with the
    compile-then-execute physical join. Matches the same logical pattern
    as NestedLoopsJoinRule so they compete via the optimizer.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = (
            isinstance(logical_expression.operator, LogicalJoinOp)
            and logical_expression.operator.condition != ""
        )
        logger.debug(f"CompiledNestedLoopsJoinRule matches_pattern: {is_match}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting CompiledNestedLoopsJoinRule for {logical_expression}")

        models = [
            model for model in runtime_kwargs["available_models"]
            if cls._model_matches_input(model, logical_expression)
        ]
        variable_op_kwargs = []
        for model in models:
            variable_op_kwargs.append({
                "model": model,
                "prompt_strategy": PromptStrategy.JOIN,
                "join_parallelism": runtime_kwargs["join_parallelism"],
                "reasoning_effort": runtime_kwargs["reasoning_effort"],
                "retain_inputs": not runtime_kwargs["is_validation"],
            })

        return cls._perform_substitution(
            logical_expression, CompiledNestedLoopsJoin, runtime_kwargs, variable_op_kwargs
        )
