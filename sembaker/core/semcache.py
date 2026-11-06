"""Semantic lookup in front of the exact artifact cache.

`sembaker.core.cache` is keyed on a hash of the canonicalized predicate, so it
has recall 0 on a reworded predicate by construction: "keep reviews rated 4
stars or higher" and "retain reviews with a rating of at least 4" compile twice.
This layer catches that case. It sits between the exact-cache miss and the
compile, and either names an existing artifact key or reports a miss.

Four stages, cheapest first:

  0. EXACT              the hash cache (the caller has already tried it).
  1. STRUCTURAL GUARD   free, hard. An entry is a candidate only if operator
                        kind, schema columns and model_id match exactly. A veto,
                        never a score: a filter cannot serve a map, and a
                        function compiled against different columns reads the
                        wrong field. No classifier confidence overrides it.
  2. BLOCKING           one MiniLM encode of the query, cosine against the
                        surviving entries, keep those above `block_tau`. Pure
                        scaling: its job is recall, not precision.
  3. ADJUDICATION       the round-5 directed containment head, called in BOTH
                        argument orders on each shortlisted pair, composed into
                        a relation. Only `equivalent` is acted on.

The composition is the point of the directed formulation:

    contains(a,b) and contains(b,a)  -> equivalent    reuse the artifact
    exactly one                      -> containment   (recorded, NOT acted on)
    neither                          -> different     compile fresh

`equivalent` therefore needs TWO independent positives, which is why a false
`equivalent` -- the only error that silently returns a function computing
something else -- is rarer than a single head's false-positive rate. The
measured out-of-fold rate is 0.0167 of all tested pairs at the accuracy-optimal
tau, and only 0.23% of pairs are `different` misread as `equivalent`; most of
that 1.67% is `containment` misread as `equivalent`, i.e. reusing a function
that is right on a superset of the rows.

CONTAINMENT IS DELIBERATELY NOT ACTED ON. The verdict and its direction are
recorded on the Decision so a caller can use them, but this layer never serves a
broader artifact in place of a narrower request. Narrowing would need the
residual predicate compiled, and in SemBaker a compile is one LLM call whether
it is the residual or the whole thing, so the saving is not obviously there.
That is a separate experiment, not a default.

Cost: stage 3 runs five cross-encoders (nli-deberta-v3-base, stsb-roberta
base+large, quora-roberta base+large) over 2k ordered pairs, which dominates
lookup latency (~0.2-0.3 s per lookup with candidates, on an Apple M3 Pro).

Enable with CX_SEMCACHE=1 (needs `pip install sembaker[semcache]`). Off by
default: it loads ~1.5GB of encoders. The classifier ships with the package at
sembaker/core/models/semcache_head.joblib; CX_SEMCACHE_MODEL overrides it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sembaker.core import semfeatures as F
from sembaker.core.cache import make_key
from sembaker.core.redact import describe

BLOCK_TAU = 0.50
_PACKAGED_MODEL = Path(__file__).resolve().parent / "models" / "semcache_head.joblib"


def _model_path() -> Path:
    p = os.environ.get("CX_SEMCACHE_MODEL")
    return Path(p) if p else _PACKAGED_MODEL


def enabled() -> bool:
    return os.environ.get("CX_SEMCACHE") == "1"


@dataclass
class Entry:
    key: str
    kind: str
    predicate: str
    columns: tuple[str, ...]
    model_id: str
    emb: np.ndarray = field(repr=False, default=None)

    @property
    def guard(self) -> tuple:
        return (self.kind, tuple(sorted(self.columns)), self.model_id)


@dataclass
class Decision:
    hit: bool
    key: str | None = None
    entry: Entry | None = None
    verdict: str = "different"       # different | containment | equivalent
    score: float = 0.0               # min(p_fwd, p_rev) of the winner
    p_fwd: float = 0.0               # P(query contains entry)
    p_rev: float = 0.0               # P(entry contains query)
    broader: str | None = None       # "query" | "entry", when containment
    stage: str = "miss"              # exact | semantic | miss
    candidates: int = 0
    shortlisted: int = 0


class SemanticCache:
    """Directed-containment semantic cache over compiled-artifact keys."""

    def __init__(self, model_path: Path | None = None, tau: float | None = None,
                 block_tau: float = BLOCK_TAU, safe: bool = True,
                 device: str | None = None):
        import joblib

        path = Path(model_path) if model_path else _model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"no semantic-cache classifier at {path}; "
                "point CX_SEMCACHE_MODEL at one.")
        b = joblib.load(path)
        self.bundle_name = b.get("name", "?")
        if b.get("n_features", F.N_FEATURES) != F.N_FEATURES:
            raise ValueError(
                f"{path} was trained on {b['n_features']} features but "
                f"semfeatures.py emits {F.N_FEATURES}; refusing to load.")
        self._model = b["model"]
        self._pos = b.get("pos_index", 1)
        self._blocks = b.get("blocks", ["emb", "cmp", "nli", "xenc"])
        # `safe` picks the operating point: tau_safe is the smallest threshold
        # whose out-of-fold `equivalent` precision clears 0.95. tau is the
        # accuracy-optimal one, which is the right number to quote and the
        # wrong one to serve from.
        self.tau = float(tau if tau is not None
                         else b["tau_safe"] if safe else b["tau"])
        self.tau_acc, self.tau_safe = float(b["tau"]), float(b["tau_safe"])
        self.block_tau = block_tau
        self.cv = b.get("cv", {})

        self._enc_names = (b["encoder"], b["nli_model"], b["xenc_models"])
        self._device = device
        self._st = None
        self._nli = None
        self._nli_idx = None
        self._xenc = None

        self._by_key: dict[str, Entry] = {}
        self._by_guard: dict[tuple, list[Entry]] = {}
        # outcome counters, so a caller can attribute a hit without having to
        # re-derive it: an artifact returned via a semantic hit carries
        # from_cache=True exactly like an exact hit, and is indistinguishable
        # downstream.
        self.n_exact = 0
        self.n_semantic = 0
        self.n_miss = 0

    # -- lazy encoder loading -------------------------------------------------

    def _encoder(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self._enc_names[0], device=self._device)
        return self._st

    def _nli_model(self):
        if self._nli is None:
            from sentence_transformers import CrossEncoder
            self._nli = CrossEncoder(self._enc_names[1], device=self._device)
            idx = {v.lower(): k for k, v in self._nli.config.id2label.items()}
            self._nli_idx = (idx["entailment"], idx["neutral"], idx["contradiction"])
        return self._nli

    def _xenc_models(self):
        if self._xenc is None:
            from sentence_transformers import CrossEncoder
            self._xenc = [[CrossEncoder(n, device=self._device) for n in pair]
                          for pair in self._enc_names[2]]
        return self._xenc

    def warm(self) -> None:
        """Load every encoder now, so the first lookup is not paying for it."""
        self._encoder()
        if "nli" in self._blocks:
            self._nli_model()
        if "xenc" in self._blocks:
            self._xenc_models()

    # -- storage --------------------------------------------------------------

    def insert(self, kind, predicate, columns, model_id, key=None) -> Entry:
        key = key or make_key(kind, predicate, list(columns), model_id)
        e = Entry(key=key, kind=kind, predicate=predicate,
                  columns=tuple(columns), model_id=model_id,
                  emb=self._encode(predicate))
        self._by_key[key] = e
        self._by_guard.setdefault(e.guard, []).append(e)
        return e

    def _encode(self, text: str) -> np.ndarray:
        return self._encoder().encode([text], normalize_embeddings=True,
                                      show_progress_bar=False)[0]

    def __len__(self):
        return len(self._by_key)

    # -- lookup ---------------------------------------------------------------

    def lookup(self, kind, predicate, columns, model_id) -> Decision:
        key = make_key(kind, predicate, list(columns), model_id)
        if key in self._by_key:                                   # stage 0
            self.n_exact += 1
            return Decision(True, key, self._by_key[key], "equivalent", 1.0,
                            1.0, 1.0, stage="exact")

        cands = self._by_guard.get((kind, tuple(sorted(columns)), model_id), [])
        if not cands:                                             # stage 1
            self.n_miss += 1
            return Decision(False)

        q = self._encode(predicate)                               # stage 2
        cos_all = np.array([float(q @ e.emb) for e in cands])
        short = np.where(cos_all >= self.block_tau)[0]
        if len(short) == 0:
            self.n_miss += 1
            return Decision(False, candidates=len(cands))

        picked = [cands[i] for i in short]
        p_fwd, p_rev = self._adjudicate(predicate, q, picked, cos_all[short])  # stage 3

        # equivalent needs both directions; among those, take the most confident
        both = np.minimum(p_fwd, p_rev)
        j = int(np.argmax(both))
        best, f, r = picked[j], float(p_fwd[j]), float(p_rev[j])
        hit = f >= self.tau and r >= self.tau
        if hit:
            verdict = "equivalent"
        elif (f >= self.tau) ^ (r >= self.tau):
            verdict = "containment"
        else:
            verdict = "different"

        if hit:
            self.n_semantic += 1
        else:
            self.n_miss += 1
        return Decision(
            hit=hit,
            key=best.key if hit else None,
            entry=best,
            verdict=verdict,
            score=float(both[j]),
            p_fwd=f, p_rev=r,
            # p_fwd is P(query's result set contains the entry's)
            broader=(None if verdict != "containment"
                     else ("query" if f > r else "entry")),
            stage="semantic" if hit else "miss",
            candidates=len(cands), shortlisted=len(picked))

    # -- stage 3 --------------------------------------------------------------

    def _adjudicate(self, predicate, q, entries, cos):
        """Score both argument orders for every shortlisted candidate."""
        n = len(entries)
        eb = np.stack([e.emb for e in entries])
        qa = np.repeat(q[None, :], n, axis=0)
        l2 = np.linalg.norm(qa - eb, axis=1)

        cmp_f = np.array([F.compact_dir(predicate, e.predicate, c, d)
                          for e, c, d in zip(entries, cos, l2)], dtype=np.float32)
        cmp_r = np.array([F.compact_dir(e.predicate, predicate, c, d)
                          for e, c, d in zip(entries, cos, l2)], dtype=np.float32)

        pairs = [(predicate, e.predicate) for e in entries]
        nli_raw = self._nli_raw(pairs)
        x_base, x_large = self._xenc_raw(pairs)

        nli_f = F.nli_dir_feats(nli_raw, False)
        nli_r = F.nli_dir_feats(nli_raw, True)
        x_f = np.hstack([F.xenc_dir_feats(x_base, False),
                         F.xenc_dir_feats(x_large, False)])
        x_r = np.hstack([F.xenc_dir_feats(x_base, True),
                         F.xenc_dir_feats(x_large, True)])

        Xf = F.assemble(F.emb_dir(qa, eb), cmp_f, nli_f, x_f)
        Xr = F.assemble(F.emb_dir(eb, qa), cmp_r, nli_r, x_r)

        p = self._model.predict_proba(np.vstack([Xf, Xr]))[:, self._pos]
        return p[:n], p[n:]

    def _nli_raw(self, pairs) -> np.ndarray:
        """(n, 6): e n c for (a,b) then for (b,a)."""
        m = self._nli_model()
        ei, ni, ci = self._nli_idx
        ab = F.softmax(np.asarray(m.predict(pairs, batch_size=32,
                                            show_progress_bar=False)))
        ba = F.softmax(np.asarray(m.predict([(b, a) for a, b in pairs],
                                            batch_size=32, show_progress_bar=False)))
        return np.stack([ab[:, ei], ab[:, ni], ab[:, ci],
                         ba[:, ei], ba[:, ni], ba[:, ci]], axis=1).astype(np.float32)

    def _xenc_raw(self, pairs):
        """Two (n, 4) arrays: sts_ab sts_ba dup_ab dup_ba, base then large."""
        rev = [(b, a) for a, b in pairs]
        out = []
        for sts, dup in self._xenc_models():
            out.append(np.stack([
                np.asarray(sts.predict(pairs, batch_size=32, show_progress_bar=False)),
                np.asarray(sts.predict(rev, batch_size=32, show_progress_bar=False)),
                np.asarray(dup.predict(pairs, batch_size=32, show_progress_bar=False)),
                np.asarray(dup.predict(rev, batch_size=32, show_progress_bar=False)),
            ], axis=1).astype(np.float32))
        return out


# --- module-level singleton, for the compile path ----------------------------

_CACHE: SemanticCache | None = None


def get_cache() -> SemanticCache | None:
    """The process-wide semantic cache, or None if disabled / unavailable."""
    global _CACHE
    if not enabled():
        return None
    if _CACHE is None:
        try:
            _CACHE = SemanticCache()
        except Exception as exc:                # never break a compile over this
            print(f"[semcache] disabled: {describe(exc)}")
            os.environ["CX_SEMCACHE"] = "0"
            return None
    return _CACHE


def reset_cache() -> None:
    global _CACHE
    _CACHE = None
