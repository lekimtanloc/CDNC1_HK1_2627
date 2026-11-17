"""Week 7 evaluator for protocol W7-HM-SIMILAR-PRODUCTS-v1 (frozen: see evaluator_manifest.json).

Implements week7/config/evaluation.json exactly: label aggregation, missing-label rules, Precision@5, MeanRelevance@5,
pooled NDCG@5, Hit@5, stratified paired bootstrap (95% percentile CI), weighted Cohen's kappa with cluster bootstrap.
Every protocol gap is listed in AMBIGUITIES and reported with each result; no rule is silently invented.

Workflow (the only supported order; enforced in code):
    1. validate  : python3 -I -B week7/src/evaluate_recommendations.py lock --a1 A.csv --a2 B.csv [--adjudicator C.csv]
                   -> validates files, writes labels_lock.json with SHA-256 of every label file. The key is NOT read.
    2. evaluate  : python3 -I -B week7/src/evaluate_recommendations.py evaluate
                   -> re-checks the lock, THEN opens the private key, computes everything, writes results.

This module does nothing at import time and never reads labels unless a function is called with explicit paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
SEED = 42
B_DEFAULT = 10000
METHODS = ("visual", "metadata")
CATS = (0, 1, 2)
EVALUATOR_VERSION = "evaluator-v1"

AMBIGUITIES = {
    "A1": "Label 'U' (cannot identify) is treated as 'not labelled'. If annotator X says U and Y gives 0/1/2, the protocol rule 'only one "
          "annotator labelled the pair' is applied literally (single label used, flagged). The protocol does not say whether a U should veto the pair.",
    "A2": "Adjudicator on a pair with only ONE annotator label: the rule says 'adjudicator if available'; the adjudicator's label is used as given "
          "(it cannot be checked against two labels). Pair with NO annotator label stays missing even if the adjudicator labelled it (literal reading).",
    "A3": "RQ1 anchor rate when some anchors are unlabelled is undefined: denominator = judged anchors; a query with 0 judged anchors is excluded from RQ1.",
    "A4": "Pooled IDCG when the query's pool has fewer than 5 judged items: all judged items are used (ideal list shorter than 5).",
    "A5": "Stratum-balanced mean when a stratum has no retained query: the mean is taken over the strata that remain (weights renormalised).",
    "A6": "Scope of the drop rule ('more than 1 missing in a method's Top-5 -> dropped from all paired comparisons'): applied literally to EVERY paired "
          "analysis (all metrics, RQ1, secondary and sensitivity analyses of the same label table), using either method's missingness.",
    "A7": "Kappa is mathematically undefined when expected disagreement is 0 (e.g. both annotators use one single category): reported as null, never as 0 or 1. "
          "Bootstrap replicates with undefined kappa are dropped and counted.",
    "A8": "Bootstrap implementation details unspecified by the protocol: queries ordered by query_id inside each stratum, strata in sorted name order, "
          "one numpy.random.default_rng(42) stream consumed stratum by stratum (index matrix of shape (B, n_stratum)); numpy default linear-interpolation percentiles. "
          "Unstratified sensitivity uses a fresh default_rng(42) over all queries sorted by query_id.",
    "A9": "'Mean label' sensitivity: pairs with two labels use their mean; pairs with one label use it; pairs with none are missing (no adjudication, no min rule). "
          "Only MeanRel@5 and pooled NDCG@5 are reported for it, as specified.",
    "A10": "CI 'excludes 0' means lower bound > 0 or upper bound < 0 (strict); a bound equal to 0 counts as 'includes 0'. RQ1 'entirely > 0' means lower bound > 0.",
    "A11": "Hit@5 is excluded for a query with any missing Top-5 position, even if a 2 is already present among the judged positions (literal rule).",
    "A12": "Catalogue-weighted sensitivity uses the fixed weight stratum_size/stratum_queries from query_manifest.csv for every retained query (not recomputed after exclusions).",
    "A13": "The 'uncertain' flag and notes are reported as counts only; they never change a label (protocol: analysis only).",
}


# ----------------------------------------------------------------------------- labels
def parse_label(raw) -> tuple:
    s = (raw or "").strip().upper()
    if s == "":
        return None, "blank"
    if s == "U":
        return None, "U"
    if s in ("0", "1", "2"):
        return int(s), "ok"
    raise ValueError(f"invalid label {raw!r}")


def read_label_file(path: Path) -> dict:
    """{pair_id: {'label': int|None, 'marker': 'ok'|'U'|'blank', 'uncertain': bool}} ; refuses duplicates and bad labels."""
    out, annots = {}, set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pid = r["pair_id"].strip()
            if pid in out:
                raise ValueError(f"{path}: duplicate pair_id {pid}")
            lab, marker = parse_label(r["label"])
            out[pid] = {"label": lab, "marker": marker, "uncertain": r.get("uncertain", "").strip() == "1"}
            annots.add(r["annotator_id"].strip())
    annots.discard("")
    if len(annots) != 1:
        raise ValueError(f"{path}: need exactly one annotator_id, found {sorted(annots)}")
    out["__annotator__"] = next(iter(annots))
    return out


def aggregate_pair(a, b, adj=None):
    """Protocol label_aggregation. a, b, adj: int in {0,1,2} or None (= not labelled). Returns (label|None, status)."""
    if a is not None and b is not None:
        if a == b:
            return a, "agree"
        if adj is not None:
            if adj not in (a, b):
                raise ValueError(f"adjudicator label {adj} is neither {a} nor {b}")
            return adj, "adjudicated"
        if abs(a - b) == 1:
            return min(a, b), "min_rule"
        return None, "unresolved"
    if a is not None or b is not None:
        if adj is not None:                                   # A2
            return adj, "single_adjudicated"
        return (a if a is not None else b), "single_label"
    return None, "missing"


def aggregate_all(pair_ids, la: dict, lb: dict, ladj: dict | None = None) -> tuple:
    """Return (final {pair: label|None}, status {pair: str}). Inputs: {pair: int|None}."""
    final, status = {}, {}
    for p in pair_ids:
        adj = (ladj or {}).get(p)
        final[p], status[p] = aggregate_pair(la.get(p), lb.get(p), adj)
    return final, status


def mean_label_table(pair_ids, la, lb) -> dict:
    out = {}
    for p in pair_ids:
        a, b = la.get(p), lb.get(p)
        out[p] = (a + b) / 2 if a is not None and b is not None else (a if a is not None else b)
    return out


# ----------------------------------------------------------------------------- per-position metrics
def precision5(rels):
    j = [r for r in rels[:5] if r is not None]
    return sum(r >= 1 for r in j) / len(j) if j else None


def meanrel5(rels):
    j = [r for r in rels[:5] if r is not None]
    return sum(j) / len(j) if j else None


def hit5(rels):
    r5 = rels[:5]
    return None if any(r is None for r in r5) else float(any(r == 2 for r in r5))


def _gain(r, kind):
    return r if kind == "linear" else 2 ** r - 1


def dcg(rels, kind="linear"):
    return sum(_gain(r, kind) / math.log2(k + 2) for k, r in enumerate(rels))


def pooled_ndcg5(rels, pool_labels, kind="linear"):
    """None if any Top-5 position is missing or IDCG == 0. IDCG = DCG of the 5 highest judged labels of the whole pool (A4 if < 5)."""
    r5 = rels[:5]
    if any(r is None for r in r5):
        return None
    ideal = sorted((x for x in pool_labels if x is not None), reverse=True)[:5]
    idcg = dcg(ideal, kind)
    return dcg(r5, kind) / idcg if idcg > 0 else None


# ----------------------------------------------------------------------------- bootstrap
def _bootstrap_means(arrays, B, seed):
    rng = np.random.default_rng(seed)
    n = sum(len(a) for a in arrays)
    tot = np.zeros(B)
    for a in arrays:
        idx = rng.integers(0, len(a), size=(B, len(a)))
        tot += a[idx].sum(axis=1)
    return tot / n, sum(a.sum() for a in arrays) / n


def stratified_bootstrap(vals_by_stratum: dict, B=B_DEFAULT, seed=SEED):
    """Mean over queries (weight 1 each) with percentile 95% CI, resampling queries WITHIN strata. Returns dict or None if empty."""
    arrays = [np.asarray(vals_by_stratum[s], dtype=float) for s in sorted(vals_by_stratum) if len(vals_by_stratum[s])]
    if not arrays:
        return None
    means, point = _bootstrap_means(arrays, B, seed)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(point), "ci_low": float(lo), "ci_high": float(hi), "n": int(sum(len(a) for a in arrays))}


def unstratified_bootstrap(values: list, B=B_DEFAULT, seed=SEED):
    if not values:
        return None
    means, point = _bootstrap_means([np.asarray(values, dtype=float)], B, seed)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(point), "ci_low": float(lo), "ci_high": float(hi), "n": len(values)}


def excludes_zero(ci) -> bool:
    return ci is not None and (ci["ci_low"] > 0 or ci["ci_high"] < 0)   # A10


# ----------------------------------------------------------------------------- kappa
def kappa_from_matrix(M, weight="linear"):
    """M: 3x3 count matrix (rows annotator 1, cols annotator 2). Returns float or None (A7)."""
    M = np.asarray(M, dtype=float)
    n = M.sum()
    if n == 0:
        return None
    k = M.shape[0]
    i, j = np.indices((k, k))
    w = (np.abs(i - j) / (k - 1)) ** (1 if weight == "linear" else 2)
    po = M / n
    pe = np.outer(po.sum(axis=1), po.sum(axis=0))
    de = (w * pe).sum()
    if de == 0:
        return None
    return float(1 - (w * po).sum() / de)


def kappa_cluster_bootstrap(per_query_mats: dict, weight="linear", B=B_DEFAULT, seed=SEED):
    """per_query_mats {query: 3x3 counts}. Cluster (query) bootstrap, percentile CI. Replicates with undefined kappa dropped (A7)."""
    qs = sorted(per_query_mats)
    if not qs:
        return None
    mats = np.stack([np.asarray(per_query_mats[q], dtype=float) for q in qs])
    point = kappa_from_matrix(mats.sum(axis=0), weight)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qs), size=(B, len(qs)))
    S = mats[idx].sum(axis=1)                                   # (B,3,3)
    n = S.sum(axis=(1, 2))
    k = 3
    i, j = np.indices((k, k))
    w = (np.abs(i - j) / (k - 1)) ** (1 if weight == "linear" else 2)
    po = S / np.where(n == 0, 1, n)[:, None, None]
    pe = po.sum(axis=2)[:, :, None] * po.sum(axis=1)[:, None, :]
    de = (w * pe).sum(axis=(1, 2))
    do = (w * po).sum(axis=(1, 2))
    ok = de > 0
    ks = 1 - do[ok] / de[ok]
    if ks.size == 0:
        return {"kappa": point, "ci_low": None, "ci_high": None, "replicates_dropped": int(B)}
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return {"kappa": point, "ci_low": float(lo), "ci_high": float(hi), "replicates_dropped": int(B - ok.sum())}


def agreement_report(pairs: list, B=B_DEFAULT) -> dict:
    """pairs: [(query_id, a, b)] with a, b in {0,1,2} (both labelled, before adjudication)."""
    per_q = defaultdict(lambda: np.zeros((3, 3)))
    M = np.zeros((3, 3))
    for q, a, b in pairs:
        per_q[q][a, b] += 1
        M[a, b] += 1
    n = int(M.sum())
    out = {"n_pairs": n, "queries": len(per_q), "confusion_rows_a1_cols_a2": M.astype(int).tolist(),
           "raw_agreement": float(np.trace(M) / n) if n else None}
    for w in ("linear", "quadratic"):
        r = kappa_cluster_bootstrap(per_q, w, B) if n else None
        out[f"kappa_{w}"] = r
    k = out["kappa_linear"]["kappa"] if out.get("kappa_linear") else None
    out["low_reliability"] = None if k is None else bool(k < 0.40)
    return out


# ----------------------------------------------------------------------------- query tables
def build_query_table(key_rows: list, queries: dict) -> dict:
    """key_rows: rows of annotation_key.csv (main set). queries: {query_id: {'group','stratum_size','stratum_queries'}}.
    Returns {query: {'top5': {method: [pair_id rank1..5]}, 'anchors': [...], 'pool': [...], 'cands': {pair: cand}}}."""
    qt = {}
    for r in key_rows:
        if r["set"] != "main":
            raise ValueError("pilot rows must never reach the evaluator")
        q = r["query_id"]
        if q not in queries:
            raise ValueError(f"key row for unknown query {q}")
        t = qt.setdefault(q, {"top5": {m: {} for m in METHODS}, "anchors": [], "pool": [], "cands": {}})
        if r["pair_id"] in t["cands"] or r["candidate_id"] in t["cands"].values():
            raise ValueError(f"duplicate pair/candidate for query {q}")
        t["pool"].append(r["pair_id"]); t["cands"][r["pair_id"]] = r["candidate_id"]
        if str(r["is_anchor"]) == "True":
            t["anchors"].append(r["pair_id"])
        for m in METHODS:
            if str(r[f"in_{m}_top5"]) == "True":
                rk = int(r[f"{m}_rank"])
                if rk in t["top5"][m]:
                    raise ValueError(f"duplicate rank {rk} ({m}, query {q})")
                t["top5"][m][rk] = r["pair_id"]
    for q, t in qt.items():
        for m in METHODS:
            if sorted(t["top5"][m]) != [1, 2, 3, 4, 5]:
                raise ValueError(f"query {q}: {m} Top-5 ranks incomplete: {sorted(t['top5'][m])}")
            t["top5"][m] = [t["top5"][m][k] for k in (1, 2, 3, 4, 5)]
        if any(a in t["top5"]["visual"] + t["top5"]["metadata"] for a in t["anchors"]):
            raise ValueError(f"query {q}: an anchor is also a Top-5 item")
    missing = set(queries) - set(qt)
    if missing:
        raise ValueError(f"queries without key rows: {sorted(missing)[:3]}")
    return qt


def per_query_values(qt: dict, labels: dict) -> dict:
    """labels: {pair_id: number|None}. Per query/method raw values; no exclusion decisions yet."""
    out = {}
    for q, t in qt.items():
        pool = [labels.get(p) for p in t["pool"]]
        row = {"anchors_judged": [labels.get(p) for p in t["anchors"] if labels.get(p) is not None]}
        for m in METHODS:
            rels = [labels.get(p) for p in t["top5"][m]]
            row[m] = {"rels": rels, "n_missing": sum(r is None for r in rels), "P5": precision5(rels), "MeanRel5": meanrel5(rels),
                      "Hit5": hit5(rels), "NDCG5": pooled_ndcg5(rels, pool, "linear"), "NDCG5_exp": pooled_ndcg5(rels, pool, "exp")}
        out[q] = row
    return out


METRICS = ("P5", "MeanRel5", "NDCG5", "Hit5")


def retained_sets(vals: dict) -> tuple:
    """Query sets per analysis + exclusion reasons (protocol missing_labels rules)."""
    dropped = sorted(q for q, v in vals.items() if v["visual"]["n_missing"] > 1 or v["metadata"]["n_missing"] > 1)        # A6
    core = [q for q in sorted(vals) if q not in set(dropped)]
    sets = {"P5": core, "MeanRel5": core}
    sets["Hit5"] = [q for q in core if vals[q]["visual"]["n_missing"] == 0 and vals[q]["metadata"]["n_missing"] == 0]
    sets["NDCG5"] = [q for q in sets["Hit5"] if vals[q]["visual"]["NDCG5"] is not None]
    sets["RQ1"] = [q for q in core if vals[q]["anchors_judged"]]
    excl = {"dropped_more_than_1_missing_in_a_Top5": dropped,
            "Hit5_NDCG5_excluded_missing_position": sorted(set(core) - set(sets["Hit5"])),
            "NDCG5_excluded_IDCG_zero": sorted(set(sets["Hit5"]) - set(sets["NDCG5"])),
            "RQ1_excluded_no_judged_anchor": sorted(set(core) - set(sets["RQ1"]))}
    return sets, excl


def _block(qs, getter_v, getter_m, groups, weights, B):
    dv = {q: getter_v(q) for q in qs}
    dm = {q: getter_m(q) for q in qs}
    diff = {q: dv[q] - dm[q] for q in qs}
    by = lambda d: {g: [d[q] for q in qs if groups[q] == g] for g in sorted(set(groups[q] for q in qs))}  # noqa: E731
    res = {"n_queries": len(qs)}
    if not qs:
        return res
    res["visual"] = stratified_bootstrap(by(dv), B)
    res["metadata"] = stratified_bootstrap(by(dm), B)
    res["diff_visual_minus_metadata"] = stratified_bootstrap(by(diff), B)
    res["diff_unstratified_sensitivity"] = unstratified_bootstrap([diff[q] for q in qs], B)
    strata = sorted(set(groups[q] for q in qs))
    res["balanced_by_stratum"] = {k: float(np.mean([np.mean([d[q] for q in qs if groups[q] == g]) for g in strata]))
                                  for k, d in (("visual", dv), ("metadata", dm), ("diff", diff))}          # A5
    wsum = sum(weights[q] for q in qs)
    res["catalogue_weighted"] = {k: float(sum(weights[q] * d[q] for q in qs) / wsum)
                                 for k, d in (("visual", dv), ("metadata", dm), ("diff", diff))}           # A12
    res["per_stratum_descriptive"] = {g: {"n": sum(groups[q] == g for q in qs),
                                          "visual": float(np.mean([dv[q] for q in qs if groups[q] == g])),
                                          "metadata": float(np.mean([dm[q] for q in qs if groups[q] == g]))} for g in strata}
    res["verdict_diff_excludes_zero"] = excludes_zero(res["diff_visual_minus_metadata"])
    return res


def analyze(qt: dict, labels: dict, queries: dict, B=B_DEFAULT, metrics=METRICS, with_rq1=True, with_exp=True) -> dict:
    vals = per_query_values(qt, labels)
    sets, excl = retained_sets(vals)
    groups = {q: queries[q]["group"] for q in qt}
    weights = {q: queries[q]["stratum_size"] / queries[q]["stratum_queries"] for q in qt}
    out = {"retained": {k: len(v) for k, v in sets.items()}, "exclusions": excl, "metrics": {}}
    for m in metrics:
        out["metrics"][m] = _block(sets[m], lambda q, m=m: vals[q]["visual"][m], lambda q, m=m: vals[q]["metadata"][m], groups, weights, B)
    if with_exp and "NDCG5" in metrics:
        out["metrics"]["NDCG5_exp_gain_sensitivity"] = _block(sets["NDCG5"], lambda q: vals[q]["visual"]["NDCG5_exp"],
                                                              lambda q: vals[q]["metadata"]["NDCG5_exp"], groups, weights, B)
    if with_rq1:
        qs = sets["RQ1"]
        arate = {q: float(np.mean([a >= 1 for a in vals[q]["anchors_judged"]])) for q in qs}                  # A3
        diff = {q: vals[q]["visual"]["P5"] - arate[q] for q in qs}
        by = lambda d: {g: [d[q] for q in qs if groups[q] == g] for g in sorted(set(groups[q] for q in qs))}  # noqa: E731
        ci = stratified_bootstrap(by(diff), B) if qs else None
        out["RQ1"] = {"n_queries": len(qs), "visual_P5_mean": float(np.mean([vals[q]["visual"]["P5"] for q in qs])) if qs else None,
                      "anchor_rate_mean": float(np.mean(list(arate.values()))) if qs else None, "delta": ci,
                      "delta_unstratified_sensitivity": unstratified_bootstrap(list(diff.values()), B),
                      "verdict_visual_above_chance": bool(ci is not None and ci["ci_low"] > 0)}              # A10
    return {"summary": out, "_vals": vals}


def verdicts(summary: dict, kappa_low: bool | None) -> dict:
    rq2 = summary["metrics"]["MeanRel5"]
    d = rq2.get("diff_visual_minus_metadata")
    if d is None:
        v2 = "undetermined (no retained queries)"
    elif excludes_zero(d):
        v2 = "difference detected: visual %s metadata on MeanRel@5" % ("higher than" if d["mean"] > 0 else "lower than")
    else:
        v2 = "no difference detected (this is NOT evidence of equivalence)"
    rq1 = summary.get("RQ1", {})
    v1 = ("visual above the random-anchor rate (CI entirely > 0)" if rq1.get("verdict_visual_above_chance")
          else "visual not shown to be above the random-anchor rate")
    return {"RQ2_primary": v2, "RQ1_primary": v1, "reliability": "TENTATIVE: kappa_w < 0.40 (low reliability)" if kappa_low else
            ("reliability not flagged" if kappa_low is False else "reliability undetermined (kappa undefined)"),
            "note": "mechanical verdicts under the frozen rules; secondary metrics are descriptive only"}


# ----------------------------------------------------------------------------- full run on in-memory tables
def run_evaluation(key_rows, queries, la: dict, lb: dict, ladj: dict | None, B=B_DEFAULT, markers=None) -> dict:
    """la/lb/ladj: {pair_id: int|None}. markers: optional {'a':{pair:marker},'b':..} for U/uncertain counts."""
    qt = build_query_table(key_rows, queries)
    pair_ids = [p for t in qt.values() for p in t["pool"]]
    final, status = aggregate_all(pair_ids, la, lb, ladj)
    primary = analyze(qt, final, queries, B)
    sens = {}
    for name, tab in (("annotator_A_only", {p: la.get(p) for p in pair_ids}), ("annotator_B_only", {p: lb.get(p) for p in pair_ids})):
        sens[name] = analyze(qt, tab, queries, B, with_rq1=False)["summary"]
    ml = analyze(qt, mean_label_table(pair_ids, la, lb), queries, B, metrics=("MeanRel5", "NDCG5"), with_rq1=False, with_exp=False)["summary"]
    sens["mean_label"] = ml
    # agreement (before adjudication): pairs labelled by BOTH annotators, overall / per method Top-5 / anchors
    members = {"all_pool_pairs": set(pair_ids)}
    for m in METHODS:
        members[f"{m}_top5_pairs"] = {p for t in qt.values() for p in t["top5"][m]}
    members["anchor_pairs"] = {p for t in qt.values() for p in t["anchors"]}
    pq = {p: q for q, t in qt.items() for p in t["pool"]}
    agree = {}
    for name, S in members.items():
        agree[name] = agreement_report([(pq[p], la[p], lb[p]) for p in sorted(S) if la.get(p) is not None and lb.get(p) is not None], B)
    kl = agree["all_pool_pairs"]["low_reliability"]
    st = Counter(status.values())
    return {"evaluator_version": EVALUATOR_VERSION, "bootstrap": {"B": B, "seed": SEED}, "ambiguities_applied": AMBIGUITIES,
            "label_aggregation": {"status_counts": dict(st), "pairs": len(pair_ids),
                                  "pairs_final_missing": sum(v is None for v in final.values()),
                                  "adjudicator_used": ladj is not None},
            "primary": primary["summary"], "verdicts": verdicts(primary["summary"], kl),
            "sensitivity": sens, "agreement": agree,
            "marker_counts": markers or {}}, primary["_vals"], final


# ----------------------------------------------------------------------------- IO / lock / CLI
def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_queries(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["article_id"]: {"group": r["product_group_name"], "stratum_size": int(r["stratum_size"]),
                                  "stratum_queries": int(r["stratum_queries"])} for r in csv.DictReader(f)}


def verify_frozen(manifest_path: Path, files_root: Path = ROOT) -> None:
    """Abort if the evaluator, its tests/spec or the protocol changed after freezing."""
    man = json.loads(Path(manifest_path).read_text())
    bad = [f for f, v in man["files"].items() if sha_file(files_root / f) != v["sha256"]]
    if bad:
        raise RuntimeError(f"evaluator/protocol changed after freeze -> BLOCKED: {bad}")


def cmd_lock(a, evaldir: Path, outdir: Path):
    sys.path.insert(0, str(Path(__file__).parent))
    import validate_annotations as VA
    tasks = VA.load_tasks(evaldir)
    files = {"annotator_1": a.a1, "annotator_2": a.a2, **({"adjudicator": a.adjudicator} if a.adjudicator else {})}
    rep = VA.validate_many(list(files.values()), "main", tasks, require_complete=True)
    if rep["status"] != "COMPLETE":
        raise SystemExit("labels not valid/complete -> no lock:\n" + json.dumps(rep["errors"] or rep["files"], indent=1, ensure_ascii=False)[:3000])
    outdir.mkdir(parents=True, exist_ok=True)
    lock = {"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evaluator_version": EVALUATOR_VERSION,
            "files": {k: {"path": str(Path(p).resolve()), "sha256": sha_file(p), "annotator_id": read_label_file(Path(p))["__annotator__"]}
                      for k, p in files.items()}, "private_key_opened": False}
    ids = [v["annotator_id"] for v in lock["files"].values()]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"annotators not distinct: {ids}")
    (outdir / "labels_lock.json").write_text(json.dumps(lock, indent=1))
    print(json.dumps(lock, indent=1))


def cmd_evaluate(a, evaldir: Path, privdir: Path, hmdir: Path, outdir: Path, frozen_manifest: Path | None):
    if frozen_manifest is not None:
        verify_frozen(frozen_manifest)
    lock_p = outdir / "labels_lock.json"
    if not lock_p.exists():
        raise SystemExit("no labels_lock.json: run 'lock' first (labels must be hashed before the key is opened)")
    lock = json.loads(lock_p.read_text())
    for k, v in lock["files"].items():
        if sha_file(v["path"]) != v["sha256"]:
            raise SystemExit(f"{k}: label file changed after lock -> BLOCKED")
    raw = {k: read_label_file(Path(v["path"])) for k, v in lock["files"].items()}
    lab = {k: {p: r["label"] for p, r in d.items() if p != "__annotator__"} for k, d in raw.items()}
    markers = {k: dict(Counter(r["marker"] for p, r in d.items() if p != "__annotator__")) for k, d in raw.items()}
    markers["uncertain_flags"] = {k: sum(r["uncertain"] for p, r in d.items() if p != "__annotator__") for k, d in raw.items()}
    # ---- only now the key is opened
    pm = json.loads((privdir / "pool_manifest.json").read_text())
    if sha_file(privdir / "annotation_key.csv") != pm["annotation_key_sha256"]:
        raise SystemExit("annotation_key.csv does not match pool_manifest.json")
    with open(privdir / "annotation_key.csv", newline="", encoding="utf-8") as f:
        key_rows = [r for r in csv.DictReader(f) if r["set"] == "main"]
    queries = load_queries(hmdir / "query_manifest.csv")
    res, vals, final = run_evaluation(key_rows, queries, lab["annotator_1"], lab["annotator_2"], lab.get("adjudicator"), a.B, markers)
    res["lock"] = lock
    res["evaluator_frozen_verified"] = frozen_manifest is not None
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    with open(outdir / "per_query_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "group", "method", "n_missing_top5", "P5", "MeanRel5", "Hit5", "NDCG5", "NDCG5_exp"])
        for q in sorted(vals):
            for m in METHODS:
                v = vals[q][m]
                w.writerow([q, queries[q]["group"], m, v["n_missing"], *[("" if v[k] is None else f"{v[k]:.6f}") for k in ("P5", "MeanRel5", "Hit5", "NDCG5", "NDCG5_exp")]])
    print(json.dumps({"verdicts": res["verdicts"], "retained": res["primary"]["retained"]}, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lock", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--evaldir", type=Path, default=W7 / "outputs/evaluation")
        p.add_argument("--outdir", type=Path, default=W7 / "outputs/evaluation_results")
        if name == "lock":
            p.add_argument("--a1", type=Path, required=True); p.add_argument("--a2", type=Path, required=True)
            p.add_argument("--adjudicator", type=Path)
        else:
            p.add_argument("--privdir", type=Path, default=W7 / "outputs/evaluation_private")
            p.add_argument("--hmdir", type=Path, default=W7 / "outputs/hm")
            p.add_argument("--B", type=int, default=B_DEFAULT)
            p.add_argument("--frozen-manifest", type=Path, default=W7 / "outputs/evaluation/evaluator_manifest.json")
            p.add_argument("--skip-freeze-check", action="store_true", help="development only; recorded in the results")
    a = ap.parse_args(argv)
    if a.cmd == "lock":
        cmd_lock(a, a.evaldir, a.outdir)
    else:
        cmd_evaluate(a, a.evaldir, a.privdir, a.hmdir, a.outdir, None if a.skip_freeze_check else a.frozen_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
