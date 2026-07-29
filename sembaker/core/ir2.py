"""IR v2: a small EXPRESSION-TREE (AST) logic IR + a deterministic transpiler.

v1 (sembaker.core.ir) was a flat "feature list + one decision string". v2 generalizes
it to a nestable typed expression tree — booleans, comparisons, arithmetic,
fallback chains, branches — while STILL being produced in a single LLM call
(grammar prompting; see sembaker.core.compile_ir2) and transpiled with NO LLM here.

Because the tree only contains a fixed set of node ops (no free-form decision
string), it is also safer than v1: the only free content is literals / regexes /
word lists, all emitted via repr().

Node grammar (JSON):
  leaves (read a field):
    {"op":"keyword_any","field":F,"words":[...]}
    {"op":"keyword_count","field":F,"words":[...]}
    {"op":"regex_present","field":F,"pattern":P}
    {"op":"regex_extract","field":F,"pattern":P}   -> first group or ""
    {"op":"column","field":F}                       -> value, stripped+UPPER
    {"op":"lit","value":V}
  combinators:
    {"op":"and"|"or","args":[...]}  {"op":"not","args":[X]}
    {"op":"eq"|"ne"|"gt"|"lt"|"ge"|"le","args":[A,B]}
    {"op":"to_int","args":[X]}                       -> int or ""
    {"op":"first_nonempty","args":[...]}
    {"op":"ifelse","cond":C,"then":T,"else":E}
  For JOIN, a leaf reading the RIGHT row adds "side":"right" (default "left").

Top: {"kind":"filter"|"map"|"join", "expr": <node>}
"""
from __future__ import annotations

_CMP = {"eq": "==", "ne": "!=", "gt": ">", "lt": "<", "ge": ">=", "le": "<="}
_LEAVES = {"keyword_any", "keyword_count", "regex_present", "regex_extract", "column"}


def _text(field, rowvar):
    return f"(str({rowvar}.get({field!r}, '') or '').lower())"


def _rowvar(node, is_join):
    if is_join:
        return "row_right" if node.get("side", "left") == "right" else "row_left"
    return "row"


def _render(node, is_join, depth=0):
    if not isinstance(node, dict) or "op" not in node:
        raise ValueError(f"bad node: {node!r}")
    if depth > 40:
        raise ValueError("expression tree too deep")
    op = node["op"]

    if op == "lit":
        return repr(node.get("value"))
    if op in ("and", "or"):
        if not node.get("args"):
            raise ValueError(f"{op} needs args")
        return "(" + f" {op} ".join(_render(a, is_join, depth + 1) for a in node["args"]) + ")"
    if op == "not":
        return f"(not {_render(node['args'][0], is_join, depth + 1)})"
    if op in _CMP:
        a, b = node["args"]
        return f"({_render(a, is_join, depth + 1)} {_CMP[op]} {_render(b, is_join, depth + 1)})"
    if op == "to_int":
        return f"_to_int({_render(node['args'][0], is_join, depth + 1)})"
    if op == "first_nonempty":
        items = "[" + ", ".join(_render(a, is_join, depth + 1) for a in node["args"]) + "]"
        return f"next((x for x in {items} if x not in ('', None)), '')"
    if op == "ifelse":
        return (f"({_render(node['then'], is_join, depth + 1)} if "
                f"{_render(node['cond'], is_join, depth + 1)} else "
                f"{_render(node['else'], is_join, depth + 1)})")

    # leaves
    if op in _LEAVES:
        rv = _rowvar(node, is_join)
        field = node["field"]
        if op == "keyword_any":
            words = [str(w).lower() for w in node.get("words", [])]
            return f"any(w in {_text(field, rv)} for w in {words!r})"
        if op == "keyword_count":
            words = [str(w).lower() for w in node.get("words", [])]
            return f"sum(1 for w in {words!r} if w in {_text(field, rv)})"
        if op == "regex_present":
            return f"bool(re.search({node['pattern']!r}, {_text(field, rv)}))"
        if op == "regex_extract":
            pat = node["pattern"]
            t = _text(field, rv)
            return (f"((re.search({pat!r}, {t}) or _N).group(1) "
                    f"if re.search({pat!r}, {t}) else '')")
        if op == "column":
            return f"str({rv}.get({field!r}, '')).strip().upper()"

    raise ValueError(f"unknown op {op!r}")


_HEADER = (
    "import re\n"
    "class _N:\n"
    "    @staticmethod\n"
    "    def group(*a):\n"
    "        return ''\n"
    "def _to_int(x):\n"
    "    try:\n"
    "        return int(str(x).strip())\n"
    "    except Exception:\n"
    "        return ''\n\n"
)


def transpile(ir: dict) -> tuple[str, str]:
    """IR tree -> (python_source, fn_name). Pure & deterministic, no LLM."""
    kind = ir.get("kind")
    if kind not in ("filter", "map", "join"):
        raise ValueError(f"bad kind {kind!r}")
    expr = _render(ir["expr"], is_join=(kind == "join"))
    if kind == "filter":
        body = f"def compiled_filter(row):\n    return bool({expr})\n"
        return _HEADER + body, "compiled_filter"
    if kind == "map":
        body = f"def compiled_map(row):\n    return ({expr})\n"
        return _HEADER + body, "compiled_map"
    body = f"def compiled_join(row_left, row_right):\n    return bool({expr})\n"
    return _HEADER + body, "compiled_join"


def ir_signature(ir: dict) -> str:
    """Structural shape (ops + fields, ignoring literal/word/regex contents)."""
    def sig(n):
        if not isinstance(n, dict):
            return "?"
        op = n.get("op")
        if op in _LEAVES:
            return f"{op}:{n.get('field')}"
        if op == "lit":
            return "lit"
        kids = []
        for k in ("args", "cond", "then", "else"):
            v = n.get(k)
            if isinstance(v, list):
                kids += [sig(x) for x in v]
            elif isinstance(v, dict):
                kids.append(sig(v))
        return f"{op}(" + ",".join(kids) + ")"
    return f"{ir.get('kind')}:{sig(ir.get('expr', {}))}"
