"""CompiledMap: PZ-compatible compile-then-execute map (ConvertScan) operator.

Mirrors PZ's LLMConvertBonded (which spends 1 LLM call PER RECORD to produce
the new column[s]) but instead does ONE compile call to produce
`compiled_map(row) -> value`, then evaluates that function locally on every
record — no per-record LLM call.

Subclasses PZ's ConvertOp so it integrates with the optimizer/cost model and
competes with LLMConvertBondedRule under the same ConvertScan logical pattern.
Register via `sembaker.backends.pz_ops.install()`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from sembaker.core import build_map_prompt, compile_artifact, make_key

from palimpzest.constants import Cardinality, Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats, OperatorCostEstimates
from palimpzest.query.operators.convert import ConvertOp
from palimpzest.query.operators.logical import ConvertScan
from palimpzest.query.optimizer.primitives import LogicalExpression, PhysicalExpression
from palimpzest.query.optimizer.rules import ImplementationRule


logger = logging.getLogger(__name__)

_COMPILE_COST_USD_ESTIMATE = 0.005
_COMPILE_TIME_SEC_ESTIMATE = 5.0


class CompiledMap(ConvertOp):
    """Compile-then-execute physical implementation of sem_map / ConvertScan.

    On the first record, issues ONE LLM request that translates the
    natural-language transformation (the generated field's description) into
    a deterministic Python function `compiled_map(row) -> value`. The
    function is cached on the operator instance and applied to every
    subsequent record locally — no LLM call in the per-record path.
    """

    _COMPILE_TEMPERATURE = 1

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.MAP,
        reasoning_effort: str = "default",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort

        self._compiled_fn: Callable | None = None
        self._compile_code: str | None = None
        self._compile_input_tokens: int = 0
        self._compile_output_tokens: int = 0
        self._compile_cost_usd: float = 0.0
        self._compile_duration_secs: float = 0.0
        # PZ runs convert per-record, possibly across a thread pool; lock so
        # only one worker triggers the one-shot compile.
        self._compile_lock = threading.Lock()

    # ----- PZ bookkeeping plumbing --------------------------------------

    def __str__(self):
        op = super().__str__()
        op += f"    Prompt Strategy: {self.prompt_strategy}\n"
        op += f"    Reasoning Effort: {self.reasoning_effort}\n"
        return op

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
        """Per-record exec is ~µs; compile cost amortized across N records."""
        n = max(1.0, source_op_cost_estimates.cardinality)
        time_per_record = 0.001 + (_COMPILE_TIME_SEC_ESTIMATE / n)
        cost_per_record = _COMPILE_COST_USD_ESTIMATE / n
        # sem_map is ONE_TO_ONE here -> selectivity 1.0
        cardinality = source_op_cost_estimates.cardinality
        # Empirical from standalone sem_map eval: normalized acc ~0.91.
        quality = 0.91
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=cost_per_record,
            quality=quality,
        )

    # ----- compile-then-execute core ------------------------------------

    def _instruction(self, fields: dict) -> str:
        """Recover the natural-language transformation. Prefer the generated
        field's description; fall back to the operator-level desc."""
        parts = []
        for fname, finfo in fields.items():
            d = getattr(finfo, "description", None)
            parts.append(f"{fname}: {d}" if d else fname)
        instr = "; ".join(parts)
        if self.desc:
            instr = f"{instr}  (operation: {self.desc})" if instr else str(self.desc)
        return instr or "transform the row"

    def _do_compile(self, fields: dict) -> None:
        """One-shot compile via sembaker.core. Populates self._compiled_fn and bookkeeping."""
        bare_model_id = self.model.value.split("/", 1)[-1] if "/" in self.model.value else self.model.value
        try:
            usd_in = self.model.get_usd_per_input_token()
            usd_out = self.model.get_usd_per_output_token()
        except Exception:
            usd_in = usd_out = 0.0

        instr = self._instruction(fields)
        # PZ streams records (no sample batch here). A warm phase registers
        # sample rows by op kind; the shared dispatcher does refine/validate/
        # cache and the CX_COMPILE method using THIS op's instruction + columns.
        from sembaker.optimizer.compile_op import compile_operator
        from sembaker.optimizer.refine_gate import lookup_samples

        samples = lookup_samples("map", instr)
        artifact = compile_operator(
            "map", instr, bare_model_id,
            cols=self.get_input_fields(), samples=samples,
            usd_per_input_token=usd_in, usd_per_output_token=usd_out, tag="CompiledMap")
        self._compiled_fn = artifact.fn
        self._compile_code = artifact.code
        self._compile_input_tokens = artifact.input_tokens
        self._compile_output_tokens = artifact.output_tokens
        self._compile_cost_usd = artifact.cost_usd
        self._compile_duration_secs = artifact.duration_secs

    def convert(self, candidate: DataRecord, fields: dict) -> tuple[dict[str, list], GenerationStats]:
        # First-record path: pay the compile cost (double-checked under lock)
        # and attribute it to this record's stats.
        compile_charged = False
        if self._compiled_fn is None:
            with self._compile_lock:
                if self._compiled_fn is None:
                    self._do_compile(fields)
                    compile_charged = True

        if compile_charged:
            in_tok = self._compile_input_tokens
            out_tok = self._compile_output_tokens
            cost = self._compile_cost_usd
            llm_time = self._compile_duration_secs
            n_calls = 1
        else:
            in_tok = out_tok = n_calls = 0
            cost = 0.0
            llm_time = 0.0

        exec_start = time.time()
        try:
            value = self._compiled_fn(candidate.to_dict())
            value = "" if value is None else value
        except Exception as e:
            if self.verbose:
                print(f"[CompiledMap exec error] {e}")
            value = ""
        exec_dur = time.time() - exec_start

        # ONE_TO_ONE convert: every generated field gets a singleton list.
        # We produce a single transformed value and assign it to each
        # requested field (our eval always generates exactly one field).
        field_answers = {fname: [value] for fname in fields}

        stats = GenerationStats(
            fn_call_duration_secs=exec_dur,
            llm_call_duration_secs=llm_time,
            input_text_tokens=in_tok,
            output_text_tokens=out_tok,
            total_llm_calls=n_calls,
            cost_per_record=cost,
        )
        return field_answers, stats


class CompiledMapRule(ImplementationRule):
    """Substitute a logical ConvertScan (semantic, no UDF) with the
    compile-then-execute physical map. Matches the same pattern as
    LLMConvertBondedRule so the optimizer evaluates both and picks on cost.
    """

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op = logical_expression.operator
        is_match = isinstance(logical_op, ConvertScan) and logical_op.udf is None
        logger.debug(f"CompiledMapRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting CompiledMapRule for {logical_expression}")
        models = [
            model for model in runtime_kwargs["available_models"]
            if cls._model_matches_input(model, logical_expression)
        ]
        variable_op_kwargs = []
        for model in models:
            variable_op_kwargs.append({
                "model": model,
                "prompt_strategy": PromptStrategy.MAP,
                "reasoning_effort": runtime_kwargs["reasoning_effort"],
            })
        return cls._perform_substitution(logical_expression, CompiledMap, runtime_kwargs, variable_op_kwargs)
