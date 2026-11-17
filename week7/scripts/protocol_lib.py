"""Reference definitions for the Week 7 H&M protocol (pure Python, no CNN).

Every function here mirrors a rule in week7/config/*.json and
week7/outputs/hm/evaluation_protocol.md. Nothing in this module reads
human labels to choose parameters.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

MISSING_TOKENS = {"", "-1", "unknown", "undefined"}


def is_missing(value) -> bool:
    return value is None or str(value).strip().lower() in MISSING_TOKENS


def tie_key(query_id: str, candidate_id: str, seed: int = 42, salt: str = "") -> str:
    return hashlib.sha256(f"{seed}:{salt}{query_id}:{candidate_id}".encode()).hexdigest()


def rank(query_id: str, scores: dict[str, float], k: int = 5, seed: int = 42, ndigits: int | None = None) -> list[str]:
    """Descending score, sha256 tie-break, query excluded."""
    items = [(c, s) for c, s in scores.items() if c != query_id]
    key = (lambda cs: (-round(cs[1], ndigits), tie_key(query_id, cs[0], seed))) if ndigits is not None \
        else (lambda cs: (-cs[1], tie_key(query_id, cs[0], seed)))
    return [c for c, _ in sorted(items, key=key)[:k]]


# ---------- metadata baseline ----------
TOKEN = re.compile(r"[a-z0-9]+")


def tfidf_vectors(docs: dict[str, str]) -> dict[str, dict[str, float]]:
    toks = {i: TOKEN.findall((d or "").lower()) for i, d in docs.items()}
    n = len(docs)
    df = Counter(t for ts in toks.values() for t in set(ts))
    idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
    out = {}
    for i, ts in toks.items():
        v = {t: (1 + math.log(c)) * idf[t] for t, c in Counter(ts).items()}
        norm = math.sqrt(sum(x * x for x in v.values()))
        out[i] = {t: x / norm for t, x in v.items()} if norm else {}
    return out


def cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def metadata_score(q: dict, c: dict, weights: dict[str, float], tq: dict, tc: dict) -> float:
    s = 0.0
    for field, w in weights.items():
        if field == "detail_desc_tfidf":
            s += w * cosine_sparse(tq, tc)
        elif not is_missing(q.get(field)) and not is_missing(c.get(field)) and q[field] == c[field]:
            s += w
    return s


# ---------- label aggregation ----------
def aggregate(a, b, adjudicated=None):
    """Return final label or None (missing). Never returns 0 for a missing label."""
    if a is None and b is None:
        return None
    if a is None or b is None:  # single label: adjudicator if present, else the one available label (flagged in reports)
        return adjudicated if adjudicated is not None else (b if a is None else a)
    if a == b:
        return a
    if adjudicated is not None:
        if adjudicated not in (a, b):
            raise ValueError("adjudicator must choose one of the two labels")
        return adjudicated
    return min(a, b) if abs(a - b) == 1 else None


# ---------- metrics ----------
def _judged(rels):
    return [r for r in rels if r is not None]


def precision_at5(rels):
    j = _judged(rels[:5])
    return sum(r >= 1 for r in j) / len(j) if j else None


def mean_rel_at5(rels):
    j = _judged(rels[:5])
    return sum(j) / len(j) if j else None


def hit_at5(rels):
    r5 = rels[:5]
    if any(r is None for r in r5):
        return None
    return float(any(r == 2 for r in r5))


def pooled_ndcg_at5(rels, pool_labels, gain="linear"):
    r5 = rels[:5]
    if any(r is None for r in r5):
        return None
    g = (lambda r: r) if gain == "linear" else (lambda r: 2 ** r - 1)
    dcg = sum(g(r) / math.log2(k + 2) for k, r in enumerate(r5))
    ideal = sorted(_judged(pool_labels), reverse=True)[:5]
    idcg = sum(g(r) / math.log2(k + 2) for k, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


def weighted_kappa(a, b, categories=(0, 1, 2), weight="linear"):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    w = [[(abs(i - j) / (k - 1)) ** (1 if weight == "linear" else 2) for j in range(k)] for i in range(k)]
    obs = [[0.0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1 / n
    pa = [sum(obs[i]) for i in range(k)]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    do = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    de = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1 - do / de if de else float("nan")


def paired_bootstrap(diffs_by_stratum: dict[str, list[float]], b: int = 10000, seed: int = 42):
    """Stratified paired bootstrap of the mean query difference; percentile 95% CI."""
    import numpy as np
    rng = np.random.default_rng(seed)
    strata = sorted(diffs_by_stratum)
    arrays = [np.asarray(diffs_by_stratum[s], dtype=float) for s in strata]
    n_total = sum(len(a) for a in arrays)
    means = np.zeros(b)
    for a in arrays:
        idx = rng.integers(0, len(a), size=(b, len(a)))
        means += a[idx].sum(axis=1)
    means /= n_total
    point = sum(a.sum() for a in arrays) / n_total
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(point), float(lo), float(hi)
