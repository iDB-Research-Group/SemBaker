"""sembaker.core: system-agnostic compile-then-execute core.

One LLM call compiles a natural-language operator intent into a deterministic
Python function; backends (Palimpzest / Lotus / Nirvana / ...) wrap the result
in their own operator plumbing. No semantic-data-system imports here.
"""

from sembaker.core.cache import clear as clear_cache
from sembaker.core.cache import make_key
from sembaker.core.compiler import (
    CompiledArtifact,
    build_filter_prompt,
    build_join_prompt,
    build_map_prompt,
    compile_artifact,
)
from sembaker.core.refine import refine_predicate

__all__ = [
    "CompiledArtifact",
    "build_filter_prompt",
    "build_map_prompt",
    "build_join_prompt",
    "compile_artifact",
    "refine_predicate",
    "make_key",
    "clear_cache",
]
