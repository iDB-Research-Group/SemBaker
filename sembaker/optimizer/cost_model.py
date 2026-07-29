"""Cost model for the external compile-vs-native decision.

All numbers are EMPIRICAL, measured on gpt-5-mini in this repo's A/B runs
(SemBench official 10-query A/B + the sembench_movie chain smoke):

  - one compile call:      ~11-49s wall, ~$0.002-0.008
  - one native per-row call: ~2-4s wall, ~$0.0003
  - compiled fn execution:  ~µs/row (negligible)

The break-even N (records for filter/map, candidate pairs for join) follows
the Compiled-AI-style amortization formula:

    n* = compile_cost / (native_per_record_cost - exec_per_record_cost)
       ≈ compile_cost / native_per_record_cost          (exec ≈ 0)

In wall-clock terms with gpt-5-mini (compile ~25s, native ~2.5s/call):
n* ≈ 10. We round per op type from what the A/B data actually showed:
small-N queries (N=4, Q2/Q3/Q4/Q8) lost on wall-clock; N>=100 (Q1/Q10,
join Q5-Q7) won. Calls/cost always win for N >= 2, so callers optimizing
for $ may pass objective="cost" to get a lower threshold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpCostProfile:
    """Empirical cost profile for one operator type."""

    compile_secs: float          # one-shot compile wall time
    compile_usd: float           # one-shot compile cost
    native_secs_per_unit: float  # native LLM call wall time per record/pair
    native_usd_per_unit: float   # native LLM call cost per record/pair

    def crossover_n(self, objective: str = "wall") -> int:
        """Smallest N at which compile-execute beats native."""
        if objective == "cost":
            denom = self.native_usd_per_unit
            numer = self.compile_usd
        else:  # wall-clock
            denom = self.native_secs_per_unit
            numer = self.compile_secs
        if denom <= 0:
            return 1
        return max(1, int(numer / denom) + 1)


# Measured on gpt-5-mini, sequential (max_workers=1). Update when the
# compile model changes — these drive every decide() call.
PROFILES: dict[str, OpCostProfile] = {
    "filter": OpCostProfile(compile_secs=25.0, compile_usd=0.005,
                            native_secs_per_unit=2.5, native_usd_per_unit=0.0003),
    "map":    OpCostProfile(compile_secs=25.0, compile_usd=0.004,
                            native_secs_per_unit=2.5, native_usd_per_unit=0.0003),
    # join units are PAIRS; native joins often run with internal parallelism
    # (PZ join_parallelism=64), which divides the effective per-pair wall
    # time — reflected in a lower native_secs_per_unit.
    "join":   OpCostProfile(compile_secs=28.0, compile_usd=0.005,
                            native_secs_per_unit=0.15, native_usd_per_unit=0.0003),
}


def crossover(op_type: str, objective: str = "wall") -> int:
    return PROFILES[op_type].crossover_n(objective)
