"""IR v2 front-end: NL operator -> expression-tree IR via GRAMMAR PROMPTING.

One LLM call: the node grammar + two worked examples go in the prompt, the model
emits a JSON expression tree, then sembaker.core.ir2.transpile turns it into code with
no further LLM call. Same latency budget as v1 (a single call) but a much richer
IR (nesting, comparisons, arithmetic, fallback chains, branches).
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from sembaker.core import ir2 as _ir2
from sembaker.core.compiler import _exec_code

_GRAMMAR = """Use ONLY these node types (a node is a JSON object):
  Leaves (read one field):
    {"op":"keyword_any","field":F,"words":[...]}      -> True if any word (lowercased substring) is in the field
    {"op":"keyword_count","field":F,"words":[...]}    -> integer count of those words
    {"op":"regex_present","field":F,"pattern":P}      -> True if the regex matches the (lowercased) field
    {"op":"regex_extract","field":F,"pattern":P}      -> the regex's first capture group, else ""
    {"op":"column","field":F}                          -> the field's value (stripped, UPPERCASED)
    {"op":"lit","value":V}                             -> a constant
  Combinators (args are nodes):
    {"op":"and","args":[...]}   {"op":"or","args":[...]}   {"op":"not","args":[X]}
    {"op":"eq"|"ne"|"gt"|"lt"|"ge"|"le","args":[A,B]}
    {"op":"to_int","args":[X]}                          -> int(X), or "" if not a number
    {"op":"first_nonempty","args":[...]}                -> first non-empty value among args
    {"op":"ifelse","cond":C,"then":T,"else":E}
  For a JOIN, a leaf that reads the RIGHT row adds  "side":"right"  (default is "left")."""

_EXAMPLES = """Examples:
  filter "clearly positive review":
    {"kind":"filter","expr":{"op":"and","args":[
      {"op":"keyword_any","field":"reviewText","words":["masterpiece","loved","brilliant","excellent"]},
      {"op":"not","args":[{"op":"keyword_any","field":"reviewText","words":["boring","waste","terrible"]}]}]}}
  map "extract the patient's age as an integer":
    {"kind":"map","expr":{"op":"to_int","args":[{"op":"first_nonempty","args":[
      {"op":"regex_extract","field":"src","pattern":"(\\\\d{1,3})\\\\s*-?\\\\s*year[\\\\s-]*old"},
      {"op":"regex_extract","field":"src","pattern":"age[:\\\\s]+(\\\\d{1,3})"}]}]}}
  join "opposite sentiment":
    {"kind":"join","expr":{"op":"ne","args":[
      {"op":"column","field":"scoreSentiment","side":"left"},
      {"op":"column","field":"scoreSentiment","side":"right"}]}}"""


def _samples_block(kind, samples, samples_left, samples_right):
    def f(rows):
        return "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in (rows or [])[:6])
    if kind == "join":
        return f"LEFT samples:\n{f(samples_left)}\nRIGHT samples:\n{f(samples_right)}"
    return f"Samples:\n{f(samples)}"


def build_prompt(kind, intent, cols, samples=None, samples_left=None, samples_right=None):
    return (
        "You are the FRONT-END of a semantic-operator compiler. Translate the "
        f"natural-language {kind} into a LOGIC EXPRESSION TREE (JSON) — do NOT write Python.\n\n"
        f"{_GRAMMAR}\n\n"
        "Output JSON ONLY:  {\"kind\":\"" + kind + "\", \"expr\": <node>}\n"
        "filter expr must be boolean; map expr is the output value; join expr is boolean over both rows.\n"
        "Prefer keying on an explicit structured column (e.g. a sentiment/label column) when one exists "
        "and is reliable. Keep the tree minimal.\n\n"
        f"{_EXAMPLES}\n\n"
        f"Now compile:\n  {kind} predicate: \"{intent}\"\n  Columns: {cols}\n"
        f"  {_samples_block(kind, samples, samples_left, samples_right)}"
    )


def _parse_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1])


def nl_to_tree(kind, intent, model_id, *, cols=None, samples=None,
               samples_left=None, samples_right=None, api_key=None, temperature=1):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    prompt = build_prompt(kind, intent, cols, samples, samples_left, samples_right)
    resp = OpenAI(api_key=key).chat.completions.create(
        model=model_id, temperature=temperature,
        messages=[{"role": "system", "content": prompt}])
    return _parse_json(resp.choices[0].message.content)


def compile_via_tree(kind, intent, model_id, *, cols=None, samples=None,
                     samples_left=None, samples_right=None, api_key=None, temperature=1):
    """Returns (ir, code, fn). One LLM call (NL->tree) + deterministic transpile."""
    ir = nl_to_tree(kind, intent, model_id, cols=cols, samples=samples,
                    samples_left=samples_left, samples_right=samples_right,
                    api_key=api_key, temperature=temperature)
    code, fn_name = _ir2.transpile(ir)
    return ir, code, _exec_code(code, fn_name)
