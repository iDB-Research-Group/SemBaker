"""System-agnostic compile-then-execute core.

This module owns everything about turning a natural-language operator intent
(filter predicate / map instruction / join condition) into a deterministic
Python function via ONE LLM call. It has NO dependency on any semantic-data
system (Palimpzest / Lotus / Nirvana) — backends wrap these functions in
their own operator plumbing.

The prompts here are byte-identical to the ones previously embedded in
sembaker.backends.pz_ops/compiled_{filter,map,join}.py, so results produced through this module
are directly comparable to all prior A/B runs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI


@dataclass
class CompiledArtifact:
    """The result of one compile call: an executable function + bookkeeping."""

    fn: Callable
    code: str
    fn_name: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_secs: float = 0.0
    from_cache: bool = False


def _exec_code(code: str, fn_name: str) -> Callable:
    """Exec generated code and return the required function."""
    ns: dict[str, Any] = {}
    exec(code, ns)
    if fn_name not in ns:
        raise RuntimeError(f"compile output did not define `{fn_name}`")
    return ns[fn_name]


# ---- prompt builders --------------------------------------------------------


def build_filter_prompt(predicate: str, field_names: list[str]) -> str:
    """Prompt that compiles a boolean per-row predicate.

    Output contract: `compiled_filter(row: dict) -> bool`.
    """
    schema = ", ".join(field_names)
    return (
        "You are an advanced database query optimizer.\n"
        "Translate the following semantic predicate into a highly optimized,\n"
        "deterministic Python function.\n\n"
        f"Predicate Intent: \"{predicate}\"\n\n"
        "Rules:\n"
        "1. The function MUST be named exactly `compiled_filter`.\n"
        "2. It must take a single argument `row`, which is a dictionary\n"
        f"   containing the following keys: {schema}.\n"
        "3. It must return a boolean (True/False).\n"
        "4. Output ONLY valid Python code. Do not include markdown formatting\n"
        "   or explanations.\n"
    )


def build_map_prompt(instruction: str, field_names: list[str]) -> str:
    """Prompt that compiles a per-row transformation.

    Output contract: `compiled_map(row: dict) -> value`.
    """
    schema = ", ".join(field_names)
    return (
        "You are an advanced database query optimizer.\n"
        "Translate the following per-row transformation instruction into a\n"
        "highly optimized, deterministic Python function.\n\n"
        f'Transformation Intent: "{instruction}"\n\n'
        "Rules:\n"
        "1. The function MUST be named exactly `compiled_map`.\n"
        "2. It must take a single argument `row`, which is a dictionary\n"
        f"   containing the following keys: {schema}.\n"
        "3. It must return a single value (string, number, or simple\n"
        "   structured type). Stringify the result if appropriate.\n"
        "4. Output ONLY valid Python code. Do not include markdown\n"
        "   formatting or explanations.\n"
    )


def build_join_prompt(
    condition: str,
    samples_left: list[dict],
    samples_right: list[dict],
) -> str:
    """Prompt that compiles a boolean per-pair join predicate.

    Output contract: `compiled_join(row_left: dict, row_right: dict) -> bool`.

    We pass BOTH the schema (column names) AND SEVERAL real sample rows from
    each side. A per-pair LLM join sees the FULL row values on every pair;
    without enough sample values the compiler is information-starved and
    cannot tell which columns are comparable across tables. ONE sample per
    side is not enough: if those two rows are about different entities the
    compiler may conclude (wrongly) that no shared key exists. Showing K
    distinct rows from each side lets the LLM observe value-distribution
    overlap (e.g. realize that some left-side ids literally appear among
    right-side ids) and pick the right join key.
    """

    def _trunc(v, n=120):
        s = "" if v is None else str(v)
        return s if len(s) <= n else s[: n - 1] + "…"

    def _fmt(samples):
        return "\n".join(
            "  " + json.dumps({k: _trunc(v) for k, v in s.items()},
                              ensure_ascii=False, default=str)
            for s in samples
        )

    schema_left = ", ".join(samples_left[0].keys()) if samples_left else ""
    schema_right = ", ".join(samples_right[0].keys()) if samples_right else ""
    return (
        "You are an advanced database query optimizer.\n"
        "Translate the following semantic JOIN condition into a highly\n"
        "optimized, deterministic Python function.\n\n"
        f"Join Intent: \"{condition}\"\n\n"
        "Left-side schema (columns):\n"
        f"  {schema_left}\n"
        f"Left-side sample rows ({len(samples_left)} shown so you can observe\n"
        "what values each column actually contains and decide which columns\n"
        "are comparable across tables — look for value-distribution overlap):\n"
        f"{_fmt(samples_left)}\n\n"
        "Right-side schema (columns):\n"
        f"  {schema_right}\n"
        f"Right-side sample rows ({len(samples_right)} shown):\n"
        f"{_fmt(samples_right)}\n\n"
        "Rules:\n"
        "1. The function MUST be named exactly `compiled_join`.\n"
        "2. It must take exactly two arguments:\n"
        "   - `row_left`: a dictionary with the left-side keys above.\n"
        "   - `row_right`: a dictionary with the right-side keys above.\n"
        "3. It must return a boolean (True if the rows should be joined, False otherwise).\n"
        "4. Handle potential missing or badly formatted data gracefully\n"
        "   (e.g. using string normalization like `.lower().strip()`).\n"
        "5. Output ONLY valid Python code. Do not include markdown formatting\n"
        "   or explanations.\n"
    )


# ---- compile ---------------------------------------------------------------


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```python"):
        s = s.split("```python", 1)[1].split("```", 1)[0].strip()
    elif s.startswith("```"):
        s = s.split("```", 1)[1].split("```", 1)[0].strip()
    return s


def compile_artifact(
    prompt: str,
    fn_name: str,
    model_id: str,
    *,
    temperature: float = 1,
    usd_per_input_token: float = 0.0,
    usd_per_output_token: float = 0.0,
    api_key: str | None = None,
    tag: str | None = None,
    print_code: bool = True,
    cache_key: str | None = None,
    validator: Callable[[Callable], float] | None = None,
    min_score: float = 0.8,
    max_attempts: int = 3,
) -> CompiledArtifact:
    """LLM call(s): prompt -> Python source -> executable function.

    Args:
        prompt: full compile prompt (use the build_*_prompt helpers).
        fn_name: required function name in the generated code.
        model_id: bare model name for the OpenAI client.
        temperature: gpt-5 family forces 1; older models accept any.
        usd_per_input_token / usd_per_output_token: pricing for bookkeeping.
        api_key: defaults to OPENAI_API_KEY env var.
        tag: label used in diagnostics (defaults to fn_name).
        print_code: print the compiled code for run-log transparency.
        cache_key: if set, reuse/store the artifact under this key.
        validator: optional fn(compiled_fn) -> score in [0,1]. When given,
            compile is retried up to max_attempts; the highest-scoring draw
            is kept (early-exit once a draw reaches min_score). This is the
            pre-cache validation gate that tames temperature=1 variance:
            only a validated draw is cached/returned. None = old behavior
            (accept the first draw).
        min_score: early-exit threshold for the validator.
        max_attempts: max compile draws when validating.

    Raises:
        RuntimeError: missing API key, or generated code never defines fn_name.
    """
    # cache fast-path: reuse a previously generated (validated) artifact.
    if cache_key is not None:
        from sembaker.core import cache as _cache

        hit = _cache.get(cache_key)
        if hit is not None:
            if print_code:
                print(f"[{tag or fn_name}] cache HIT ({cache_key}) -> 0 LLM calls")
            return hit

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (sembaker.core compiles via the OpenAI API)")
    client = OpenAI(api_key=key)
    label = tag or fn_name

    # Broken generations (syntax errors / missing fn) are objective failures —
    # retry those up to max_attempts even WITHOUT a validator; the first draw
    # that execs is accepted in that case (validator-less behavior otherwise
    # unchanged: no best-of-N scoring, just resilience to bad draws).
    attempts = max_attempts
    best: CompiledArtifact | None = None
    best_score = -1.0
    last_err: Exception | None = None

    for attempt in range(attempts):
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
            )
            code = _strip_code_fences(resp.choices[0].message.content)
            fn = _exec_code(code, fn_name)
        except Exception as e:  # broken generation OR transient API error -> failed draw, retry
            last_err = e
            if print_code:
                print(f"[{label}] draw {attempt+1}/{attempts} failed: {type(e).__name__}")
            continue
        duration = time.time() - start

        usage = getattr(resp, "usage", None)
        in_tok = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        out_tok = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        art = CompiledArtifact(
            fn=fn, code=code, fn_name=fn_name, model_id=model_id,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=usd_per_input_token * in_tok + usd_per_output_token * out_tok,
            duration_secs=duration,
        )

        if validator is None:
            best = art
            break

        try:
            score = float(validator(fn))
        except Exception:
            score = 0.0
        if print_code:
            print(f"[{label}] draw {attempt+1}/{attempts} validate score={score:.3f}")
        if score > best_score:
            best, best_score = art, score
        if score >= min_score:
            break

    if best is None:
        raise RuntimeError(
            f"compile produced no usable `{fn_name}` in {attempts} attempt(s)"
            + (f": {type(last_err).__name__}: {last_err}" if last_err else "")
        )

    if print_code:
        print(
            f"[{label}] model={model_id}  compile={best.duration_secs:.2f}s  "
            f"in={best.input_tokens}tok  out={best.output_tokens}tok  cost=${best.cost_usd:.4f}"
            + (f"  best_score={best_score:.3f}" if validator is not None else "")
        )
        print("------ compiled code ------")
        try:
            print(best.code)
        except UnicodeEncodeError:
            # Non-UTF-8 console (e.g. GBK on Chinese Windows) meeting emoji or
            # other exotic chars the LLM put in comments — degrade, never crash.
            print(best.code.encode("ascii", "backslashreplace").decode("ascii"))
        print("---------------------------")

    if cache_key is not None:
        from sembaker.core import cache as _cache

        _cache.put(cache_key, best)

    return best
