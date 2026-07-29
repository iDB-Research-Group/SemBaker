"""sembaker — compile-then-execute semantic operators, as an external optimizer.

Semantic-operator systems (Palimpzest, LOTUS, Nirvana, DocETL, ...) execute
natural-language operators by calling an LLM per row / pair / document. sembaker
instead calls the LLM ONCE to compile the operator into a deterministic Python
function, then runs that function locally over the whole dataset:

    interpretation:  N items -> N LLM calls
    compilation:     N items -> 1 compile call + N local function calls (~0)

Package layout:

    sembaker.core        the compiler itself (backend-agnostic): compile,
                      refine, validate, cache, IR v1/v2, scheduler
    sembaker.optimizer   compile-vs-native decision layer + the unified,
                      ablation-aware compile entry point (compile_operator)
    sembaker.backends    engine adapters — each requires its engine installed:
                        sembaker.backends.pz       pip install sembaker[pz]
                        sembaker.backends.lotus    pip install sembaker[lotus]
                        sembaker.backends.nirvana  pip install sembaker[nirvana]
                        sembaker.backends.docetl   pip install sembaker[docetl]

Your own engine: see docs/writing_a_backend.md — the compile core is public
API and an adapter is typically ~100 lines.
"""

__version__ = "0.1.0"

from sembaker.core import (
    CompiledArtifact,
    build_filter_prompt,
    build_join_prompt,
    build_map_prompt,
    clear_cache,
    compile_artifact,
    make_key,
    refine_predicate,
)
from sembaker.optimizer import PROFILES, Decision, OpCostProfile, crossover, decide

__all__ = [
    "__version__",
    "CompiledArtifact",
    "build_filter_prompt",
    "build_map_prompt",
    "build_join_prompt",
    "compile_artifact",
    "refine_predicate",
    "make_key",
    "clear_cache",
    "decide",
    "Decision",
    "crossover",
    "PROFILES",
    "OpCostProfile",
]
