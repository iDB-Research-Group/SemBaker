"""IR-decoupled compilation: NL operator -> logic IR (LLM) -> code (transpiler).

Step A (here, the only LLM call): semantic analysis of the NL operator into a
small structured IR (sembaker.core.ir). Step B: deterministic transpile (sembaker.core.ir.
transpile) — no LLM. All temperature=1 variance is thus confined to the IR, a
small structured space, instead of free-form code.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from sembaker.core import ir as _ir
from sembaker.core.compiler import CompiledArtifact, _exec_code

_PRIMITIVES = """Feature kinds (the ONLY primitives you may use):
  keyword_any   args.words=[...]   -> True if any word (lowercased substring) appears in the field
  keyword_count args.words=[...]   -> integer count of those words present
  column                            -> the field's value (stripped, UPPERCASED), for comparisons
  column_equals args.value="X"     -> True if field value == X (case-insensitive)
  regex_present args.pattern="..." -> True if regex matches the (lowercased) field
  regex_extract args.pattern="..." -> first capture group of the regex, else '' (for map)"""

_FORMAT = {
    "filter": """Output JSON ONLY, no prose:
{"op":"filter",
 "features":[{"id":"f1","kind":"<primitive>","field":"<column>","args":{...}}, ...],
 "decision":{"return":"<python boolean expression over the feature ids, e.g. f1 and not f2>"}}""",
    "map": """Output JSON ONLY, no prose:
{"op":"map",
 "features":[{"id":"m1","kind":"<primitive>","field":"<column>","args":{...}}, ...],
 "decision":{"return":"<python expression over feature ids producing the output value>"}}""",
    "join": """Output JSON ONLY, no prose:
{"op":"join",
 "left_features":[{"id":"L1","kind":"<primitive>","field":"<column>","args":{...}}, ...],
 "right_features":[{"id":"R1","kind":"<primitive>","field":"<column>","args":{...}}, ...],
 "decision":{"return":"<python boolean expression over L*/R* ids, e.g. L1 != R1>"}}""",
}


def _samples_block(samples, samples_left, samples_right, kind):
    def fmt(rows):
        return "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in (rows or [])[:6])
    if kind == "join":
        return f"LEFT sample rows:\n{fmt(samples_left)}\nRIGHT sample rows:\n{fmt(samples_right)}"
    return f"Sample rows:\n{fmt(samples)}"


def build_ir_prompt(kind, intent, cols, samples=None, samples_left=None, samples_right=None):
    return (
        "You are the FRONT-END of a semantic-operator compiler. Translate the "
        f"natural-language {kind} into a structured LOGIC IR — describe HOW to decide "
        "using ONLY the primitives below; do not write Python.\n\n"
        f"{kind} predicate: \"{intent}\"\n"
        f"Columns available: {cols}\n"
        f"{_samples_block(samples, samples_left, samples_right, kind)}\n\n"
        f"{_PRIMITIVES}\n\n"
        "Prefer keying on explicit structured columns (e.g. a sentiment/label column) "
        "when one exists and is reliable. Keep the IR minimal.\n\n"
        f"{_FORMAT[kind]}"
    )


def _parse_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1])


def nl_to_ir(kind, intent, model_id, *, cols=None, samples=None,
             samples_left=None, samples_right=None, api_key=None, temperature=1):
    """One LLM call: NL operator -> IR dict."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    prompt = build_ir_prompt(kind, intent, cols, samples, samples_left, samples_right)
    resp = OpenAI(api_key=key).chat.completions.create(
        model=model_id, temperature=temperature,
        messages=[{"role": "system", "content": prompt}])
    return _parse_json(resp.choices[0].message.content)


def compile_via_ir(kind, intent, model_id, *, cols=None, samples=None,
                   samples_left=None, samples_right=None, api_key=None, temperature=1):
    """Returns (ir, code, compiled_fn). Step A = LLM->IR; Step B = transpile (no LLM)."""
    ir = nl_to_ir(kind, intent, model_id, cols=cols, samples=samples,
                  samples_left=samples_left, samples_right=samples_right,
                  api_key=api_key, temperature=temperature)
    code, fn_name = _ir.transpile(ir)         # deterministic, no LLM
    fn = _exec_code(code, fn_name)
    return ir, code, fn
