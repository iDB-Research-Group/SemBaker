"""Directed pair features for the semantic cache, at lookup time.

The classifier shipped in models/semcache_head.joblib was trained on exactly
these features. The FEATURE DEFINITIONS must stay bit-identical to the ones it
was trained on or the fitted trees read the wrong column, so do not change the
lexical resources or the arithmetic below without retraining.

Column contract, in order:

    emb  768   e_x * e_y  ++  e_x - e_y          (MiniLM, normalized)
    cmp   44   compact_dir(x, y, cos, l2)
    nli    8   nli_dir_feats                     (nli-deberta-v3-base)
    xenc   8   stsb/quora roberta base ++ large
    ----------
          828

Convention: `x` is the candidate BROADER side, `y` the candidate NARROWER side.
The head answers "does result_set(x) contain result_set(y)"; the cache calls it
in both orders and composes.

If you retrain with a different feature set, bump `FEATURE_VERSION` and record
it in the joblib bundle — `semcache.py` refuses to load a bundle whose feature
count does not match, which is cheaper than debugging a silently misaligned
828-column vector.
"""

from __future__ import annotations

import re

import numpy as np

FEATURE_VERSION = 1
N_EMB = 768
N_CMP = 44
N_NLI = 8
N_XENC = 8
N_FEATURES = N_EMB + N_CMP + N_NLI + N_XENC        # 828

# --- lexical resources -------------------------------------------------------

NEG = re.compile(r"\b(not|no|never|neither|nor|exclude|excluding|drop|remove|"
                 r"without|non|other than|isn't|aren't|don't|doesn't)\b")
GE = re.compile(r"\b(at least|or higher|or more|or above|above|over|greater|"
                r"minimum|\d\+|maximum score|perfect)\b")
LE = re.compile(r"\b(at most|or lower|or fewer|or less|below|under|fewer|"
                r"less than|lower)\b")
MAP_CUE = re.compile(r"\b(summarize|summary|condense|label|assign|classify|"
                     r"extract|identify|pull|predict|guess|infer|list|"
                     r"enumerate|translate|render|convert|categorize|"
                     r"determine|write)\b")
FILTER_CUE = re.compile(r"\b(keep|select|retain|filter|drop|exclude|include|"
                        r"return only)\b")
VERBS = re.compile(r"\b(keep|select|retain|filter to|filter down to|return|"
                   r"include|take)\b")
FILLER = re.compile(r"\b(only|just|the|a|an|every|all|that are|which are|that|"
                    r"which|reviews?|customers?|is|are|was|were|with|in|of|to|"
                    r"for|and|or|this|it|they|them)\b")
WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
CONJ = re.compile(r"\b(and|both|as well as|who also|that also|which also|from)\b")
EXACT = re.compile(r"\b(only|exactly|perfect|maximum|lowest|highest|single|"
                   r"precisely|solely)\b")
SPECIFIC = re.compile(r"\b(within|first|during|specifically|particular|"
                      r"spanish|french|german|darker|lighter|late|delayed|"
                      r"damaged|refused|denied|rude|overpriced|counterfeit)\b")


def _toks(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", t.lower()))


def _char3(t: str) -> set[str]:
    s = re.sub(r"\s+", " ", t.lower())
    return {s[i:i + 3] for i in range(max(0, len(s) - 2))}


def _content(t: str) -> set[str]:
    s = VERBS.sub(" KEEP ", t.lower())
    s = FILLER.sub(" ", s)
    return _toks(s)


def _nums(t: str) -> set[str]:
    s = t.lower()
    for w, d in WORDNUM.items():
        s = re.sub(rf"\b{w}\b", d, s)
    return set(re.findall(r"\d+", s))


def _jac(x: set, y: set) -> float:
    return len(x & y) / len(x | y) if (x or y) else 0.0


# --- the 44 directed lexical/structural features ----------------------------

DIR_NAMES = [
    "cosine", "l2", "tok_jaccard", "char3_jaccard", "content_jaccard",
    "len_ratio", "num_both", "num_equal", "num_disjoint", "cmp_conflict",
    "optype_conflict",
    "neg_x", "neg_y", "ge_x", "ge_y", "le_x", "le_y",
    "map_x", "map_y", "fil_x", "fil_y",
    "conj_x", "conj_y", "exact_x", "exact_y", "spec_x", "spec_y",
    "cov_x_in_y", "cov_y_in_x", "surplus_y", "surplus_x", "len_diff_signed",
    "conj_y_only", "conj_x_only", "exact_y_only", "exact_x_only",
    "spec_y_only", "spec_x_only", "neg_y_only", "neg_x_only",
    "num_x", "num_y", "num_y_tighter", "num_x_tighter",
]
assert len(DIR_NAMES) == N_CMP


def compact_dir(x: str, y: str, cos: float, l2: float) -> list[float]:
    """44 directed features for the ordered pair (broad candidate, narrow candidate)."""
    xl, yl = x.lower(), y.lower()
    tx, ty = _toks(x), _toks(y)
    cx, cy = _content(x), _content(y)
    nx, ny = _nums(x), _nums(y)
    lx, ly = len(tx), len(ty)

    neg_x, neg_y = bool(NEG.search(xl)), bool(NEG.search(yl))
    ge_x, ge_y = bool(GE.search(xl)), bool(GE.search(yl))
    le_x, le_y = bool(LE.search(xl)), bool(LE.search(yl))
    map_x, map_y = bool(MAP_CUE.search(xl)), bool(MAP_CUE.search(yl))
    fil_x, fil_y = bool(FILTER_CUE.search(xl)), bool(FILTER_CUE.search(yl))
    conj_x, conj_y = bool(CONJ.search(xl)), bool(CONJ.search(yl))
    exact_x, exact_y = bool(EXACT.search(xl)), bool(EXACT.search(yl))
    spec_x, spec_y = bool(SPECIFIC.search(xl)), bool(SPECIFIC.search(yl))

    inter = len(cx & cy)
    mx = max((int(v) for v in nx), default=0)
    my = max((int(v) for v in ny), default=0)
    same_dir = (ge_x and ge_y) or (le_x and le_y)
    y_tighter = same_dir and mx and my and ((my > mx) if ge_x else (my < mx))
    x_tighter = same_dir and mx and my and ((mx > my) if ge_x else (mx < my))

    return [
        cos, l2, _jac(tx, ty), _jac(_char3(x), _char3(y)), _jac(cx, cy),
        min(lx, ly) / max(lx, ly) if max(lx, ly) else 0.0,
        float(bool(nx) and bool(ny)),
        float(bool(nx) and nx == ny),
        float(bool(nx) and bool(ny) and not (nx & ny)),
        float((ge_x and le_y) or (le_x and ge_y)),
        float((map_x and fil_y and not map_y) or (map_y and fil_x and not map_x)),
        float(neg_x), float(neg_y), float(ge_x), float(ge_y),
        float(le_x), float(le_y), float(map_x), float(map_y),
        float(fil_x), float(fil_y), float(conj_x), float(conj_y),
        float(exact_x), float(exact_y), float(spec_x), float(spec_y),
        inter / len(cx) if cx else 0.0,
        inter / len(cy) if cy else 0.0,
        (len(cy) - inter) / max(1, len(cy)),
        (len(cx) - inter) / max(1, len(cx)),
        (ly - lx) / 20.0,
        float(conj_y and not conj_x), float(conj_x and not conj_y),
        float(exact_y and not exact_x), float(exact_x and not exact_y),
        float(spec_y and not spec_x), float(spec_x and not spec_y),
        float(neg_y and not neg_x), float(neg_x and not neg_y),
        mx / 5.0, my / 5.0, float(y_tighter), float(x_tighter),
    ]


# --- cross-encoder derived blocks -------------------------------------------

def nli_dir_feats(raw: np.ndarray, flip: bool) -> np.ndarray:
    """raw columns: e_ab n_ab c_ab e_ba n_ba c_ba -> 8 directed features.

    The informative direction for "does x contain y" is y |= x: if the narrower
    predicate entails the broader one, the broader one's result set contains it.
    High entailment in BOTH directions means equivalence; in one, containment.
    """
    e_ab, n_ab, c_ab, e_ba, n_ba, c_ba = [raw[:, i] for i in range(6)]
    if flip:
        e_yx, n_yx, c_yx, e_xy, n_xy, c_xy = e_ab, n_ab, c_ab, e_ba, n_ba, c_ba
    else:
        e_yx, n_yx, c_yx, e_xy, n_xy, c_xy = e_ba, n_ba, c_ba, e_ab, n_ab, c_ab
    return np.stack([e_yx, n_yx, c_yx, e_xy, n_xy, c_xy,
                     e_yx - e_xy, e_yx * (1.0 - e_xy)], axis=1).astype(np.float32)


def xenc_dir_feats(raw: np.ndarray, flip: bool) -> np.ndarray:
    """raw columns: sts_ab sts_ba dup_ab dup_ba -> 4 directed features."""
    s_ab, s_ba, d_ab, d_ba = [raw[:, i] for i in range(4)]
    s_xy, s_yx = (s_ba, s_ab) if flip else (s_ab, s_ba)
    d_xy, d_yx = (d_ba, d_ab) if flip else (d_ab, d_ba)
    return np.stack([(s_xy + s_yx) / 2, s_xy - s_yx,
                     (d_xy + d_yx) / 2, d_xy - d_yx], axis=1).astype(np.float32)


def emb_dir(ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """Signed difference carries the direction; product carries the similarity."""
    return np.hstack([ex * ey, ex - ey]).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def assemble(emb_block, cmp_block, nli_block, xenc_block) -> np.ndarray:
    """Concatenate the four blocks in the trained column order, with a width check."""
    X = np.hstack([emb_block, cmp_block, nli_block, xenc_block]).astype(np.float32)
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"feature width {X.shape[1]} != trained {N_FEATURES}; "
                         "the saved head and semfeatures.py have diverged")
    return X
