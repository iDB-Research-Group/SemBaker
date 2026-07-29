"""Operator logic IR + a DETERMINISTIC transpiler (IR -> Python, no LLM).

This is the "after-AST" half of the IR-decoupled compilation idea: once the
logic is fixed as a small structured IR, rendering it to code is a mechanical,
deterministic transpile — the LLM's temperature=1 variance is confined to the
"before-AST" step (producing the IR), not the code.

IR shape (JSON):
  filter:
    {"op":"filter",
     "features":[{"id":"f1","kind":"keyword_any","field":"reviewText",
                  "args":{"words":["masterpiece","loved",...]}}, ...],
     "decision":{"return":"f1 and not f2"}}
  join:
    {"op":"join",
     "left_features":[{"id":"L1","kind":"column","field":"scoreSentiment"}],
     "right_features":[{"id":"R1","kind":"column","field":"scoreSentiment"}],
     "decision":{"return":"L1 != R1"}}
  map:
    {"op":"map",
     "features":[{"id":"m1","kind":"regex_extract","field":"reviewText",
                  "args":{"pattern":"(\\d+)\\s*-?\\s*year"}}],
     "decision":{"return":"m1"}}

Feature kinds (the fixed primitive vocabulary the transpiler knows):
  keyword_any   args.words -> True if any word (substring, lowercased) in field
  keyword_count args.words -> count of words present in field
  column        -> the field's value, stripped+uppercased (for comparisons)
  column_equals args.value -> field value == value (case-insensitive)
  regex_present args.pattern -> bool(re.search(pattern, field_lower))
  regex_extract args.pattern -> first group of re.search, or '' (for map)
"""
from __future__ import annotations

import re

_KINDS = {"keyword_any", "keyword_count", "column", "column_equals",
          "regex_present", "regex_extract"}

# identifiers allowed in a decision expression besides the feature ids
_DEC_KEYWORDS = {"and", "or", "not", "in", "True", "False", "None", "if", "else"}
# safe, side-effect-free builtins the LLM legitimately uses to combine features
# (e.g. int(m1) to turn an extracted age string into a number)
_DEC_BUILTINS = {"int", "float", "str", "len", "bool", "abs", "min", "max", "round", "sum"}
_IDENT = re.compile(r"[A-Za-z_]\w*")
_FORBIDDEN = ("__", "import", "exec", "eval", "open", "lambda", "os.", "sys.")


def _text(field: str, rowvar: str) -> str:
    return f"(str({rowvar}.get({field!r}, '') or '').lower())"


def _render_feature(f: dict, rowvar: str) -> str:
    kind = f["kind"]
    field = f["field"]
    a = f.get("args", {}) or {}
    if kind == "keyword_any":
        words = [str(w).lower() for w in a.get("words", [])]
        return f"any(w in {_text(field, rowvar)} for w in {words!r})"
    if kind == "keyword_count":
        words = [str(w).lower() for w in a.get("words", [])]
        return f"sum(1 for w in {words!r} if w in {_text(field, rowvar)})"
    if kind == "column":
        return f"str({rowvar}.get({field!r}, '')).strip().upper()"
    if kind == "column_equals":
        val = str(a.get("value", "")).upper()
        return f"(str({rowvar}.get({field!r}, '')).strip().upper() == {val!r})"
    if kind == "regex_present":
        pat = str(a.get("pattern", ""))
        return f"bool(re.search({pat!r}, {_text(field, rowvar)}))"
    if kind == "regex_extract":
        pat = str(a.get("pattern", ""))
        return (f"((re.search({pat!r}, {_text(field, rowvar)}) or _NO).group(1) "
                f"if re.search({pat!r}, {_text(field, rowvar)}) else '')")
    raise ValueError(f"unknown feature kind {kind!r}")


def _check_decision(expr: str, ids: set[str]) -> None:
    if any(tok in expr for tok in _FORBIDDEN):
        raise ValueError(f"forbidden token in decision: {expr!r}")
    for ident in _IDENT.findall(expr):
        if ident not in ids and ident not in _DEC_KEYWORDS and ident not in _DEC_BUILTINS:
            raise ValueError(f"decision references unknown id {ident!r} (allowed: {sorted(ids)})")


def _validate_features(feats: list[dict]) -> None:
    for f in feats:
        if f.get("kind") not in _KINDS:
            raise ValueError(f"bad feature kind: {f.get('kind')!r}")
        if not f.get("id") or not f.get("field"):
            raise ValueError(f"feature missing id/field: {f}")


def transpile(ir: dict) -> tuple[str, str]:
    """IR -> (python_source, fn_name). Pure & deterministic."""
    op = ir.get("op")
    dec = (ir.get("decision") or {}).get("return", "")
    if not isinstance(dec, str) or not dec.strip():
        raise ValueError("missing decision.return")

    if op == "join":
        lf = ir.get("left_features", []) or []
        rf = ir.get("right_features", []) or []
        _validate_features(lf); _validate_features(rf)
        ids = {f["id"] for f in lf} | {f["id"] for f in rf}
        _check_decision(dec, ids)
        lines = ["import re", "", "class _NO: ",
                 "    @staticmethod", "    def group(*a): return ''", "",
                 "def compiled_join(row_left, row_right):"]
        for f in lf:
            lines.append(f"    {f['id']} = {_render_feature(f, 'row_left')}")
        for f in rf:
            lines.append(f"    {f['id']} = {_render_feature(f, 'row_right')}")
        lines.append(f"    return bool({dec})")
        return "\n".join(lines), "compiled_join"

    # filter / map
    feats = ir.get("features", []) or []
    _validate_features(feats)
    ids = {f["id"] for f in feats}
    _check_decision(dec, ids)
    fn = "compiled_filter" if op == "filter" else "compiled_map"
    ret = f"bool({dec})" if op == "filter" else f"({dec})"
    lines = ["import re", "", "_NO = None", "", f"def {fn}(row):"]
    for f in feats:
        lines.append(f"    {f['id']} = {_render_feature(f, 'row')}")
    lines.append(f"    return {ret}")
    return "\n".join(lines), fn


def ir_signature(ir: dict) -> str:
    """Logic 'shape' ignoring exact word/pattern contents — used to measure how
    STABLE the logic is across LLM draws (two draws with the same structure but
    different keyword lists count as the same logic)."""
    def feats(key):
        return "|".join(f"{f.get('kind')}:{f.get('field')}"
                        for f in (ir.get(key) or []))
    dec = re.sub(r"\s+", "", (ir.get("decision") or {}).get("return", ""))
    if ir.get("op") == "join":
        return f"join[{feats('left_features')}#{feats('right_features')}]=>{dec}"
    return f"{ir.get('op')}[{feats('features')}]=>{dec}"
