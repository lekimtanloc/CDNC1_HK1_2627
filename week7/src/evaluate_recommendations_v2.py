"""Week 7 evaluator v2 = evaluator v1 (frozen, imported unchanged) + methodology amendment v1.

Amendment v1 (week7/outputs/methodology/methodology_amendment_v1.md) resolves the four result-affecting ambiguities:
  A1  'U'/blank = abstention (missing). A lone numeric label is NOT primary ground truth when two independent judgments are required.
  A2  An adjudicator resolves a disagreement between >= 2 independent numeric labels. With only one numeric label, an additional
      numeric label is a SECOND JUDGMENT (applies the two-label rules), never an overriding adjudication.
  A3  RQ1 uses only queries whose Visual Top-5 AND both random anchors are fully labelled.
  A6  RQ1 and RQ2 have separate query-exclusion rules. RQ2 needs all 10 Top-5 positions (both lists) labelled. Within each comparison both
      arms use exactly the same query set.
Everything else (metrics, bootstrap, kappa, lock-before-key workflow) is the frozen v1 code. The v1 rules are kept ONLY as labelled
sensitivity analyses so that the effect of the amendment is visible.

Workflow (enforced):
    lock  --a1 A --a2 B                 -> validates + hashes the two primary files (key not opened)
    plan                                -> lists pairs needing a second judgment / adjudication (labels only, key not opened)
    lock  --a1 A --a2 B --third C       -> optional: appends the supplementary/adjudication file (primary files must be unchanged)
    evaluate                            -> checks frozen hashes + lock, THEN opens the private key, writes results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate_recommendations as E1  # noqa: E402  (frozen v1; hash recorded in evaluator_v2_manifest.json)

ROOT, W7 = E1.ROOT, E1.W7
METHODS = E1.METHODS
EVALUATOR_VERSION = "evaluator-v2"
AMENDMENT = W7 / "outputs/methodology/methodology_amendment_v1.md"
V2_MANIFEST = W7 / "outputs/methodology/evaluator_v2_manifest.json"

AMBIGUITY_REGISTER = {
    "A1": {"status": "RESOLVED_BY_AMENDMENT", "rule": "U/blank = abstention. A single numeric label is not primary ground truth; the pair is missing unless an independent second numeric judgment exists."},
    "A2": {"status": "RESOLVED_BY_AMENDMENT", "rule": "Adjudicator only between >= 2 numeric labels. One numeric label + one additional numeric label = second judgment (two-label rules), not an overriding adjudication."},
    "A3": {"status": "RESOLVED_BY_AMENDMENT", "rule": "RQ1 query needs the 5 Visual Top-5 labels and both anchor labels."},
    "A4": {"status": "MOOT_BY_AMENDMENT", "rule": "A complete Top-5 puts >= 5 judged items in the pool, so a pool with < 5 judged items cannot reach IDCG."},
    "A5": {"status": "OPEN", "rule": "Stratum-balanced mean over the strata that still have retained queries."},
    "A6": {"status": "RESOLVED_BY_AMENDMENT", "rule": "Separate RQ1 / RQ2 exclusion; RQ2 needs both Top-5 lists fully labelled; both arms share one query set inside each comparison."},
    "A7": {"status": "OPEN", "rule": "kappa undefined (null) when expected disagreement is 0; such bootstrap replicates are dropped and counted."},
    "A8": {"status": "OPEN", "rule": "Bootstrap ordering/RNG/percentile conventions of evaluator v1 (queries sorted by id inside strata, default_rng(42), linear percentiles)."},
    "A9": {"status": "REVISED_BY_AMENDMENT", "rule": "Mean-label sensitivity needs >= 2 independent numeric judgments (a and b, or the single one of a/b plus the third file); an adjudication choice is not an independent rating."},
    "A10": {"status": "OPEN", "rule": "CI excludes 0 only if lower bound > 0 or upper bound < 0 (strict); bounds within 1e-12 of 0 count as 0 (floating-point hygiene, added in v2 after a fixture showed 5e-17 noise)."},
    "A11": {"status": "MOOT_BY_AMENDMENT", "rule": "Hit@5 is only computed on complete lists, so the 'a 2 is present but a position is missing' case cannot occur."},
    "A12": {"status": "OPEN", "rule": "Catalogue weights stratum_size/stratum_queries are fixed, not recomputed after exclusions."},
    "A13": {"status": "OPEN", "rule": "'uncertain' flags and notes are reported as counts only."},
    "N1": {"status": "REVIEW_REQUIRED", "rule": "Operational details of the supplementary-judgment procedure (who, how many raters, a single third file, no further adjudicator for second-judgment disagreements) are specified by the amendment but not yet approved."},
    "N2": {"status": "REVIEW_REQUIRED", "rule": "No interface exists that shows the two labels to an adjudicator (protocol section 6.2 step 3); the supplementary task list is produced by 'plan' but the annotation page labels the full set blind."},
    "N3": {"status": "REVIEW_REQUIRED", "rule": "No minimum number of retained queries is defined for confirmatory claims. Strict complete-list rules can shrink RQ2 / RQ1 samples (and may bias toward easily identified products)."},
    "N4": {"status": "REVIEW_REQUIRED", "rule": "RQ1 and RQ2 use different query populations; their results are not pooled or compared with each other."},
    "N5": {"status": "OPEN", "rule": "Metric-specific IDCG = 0 exclusion for pooled NDCG@5 is kept from the frozen protocol, so NDCG@5 may use fewer queries than the other RQ2 metrics (both arms always share the same set)."},
}


# ----------------------------------------------------------------------------- numeric hygiene (A10)
TOL = 1e-12


def excludes_zero_v2(ci) -> bool:
    """Strict rule of A10, with a floating-point tolerance: a bound within 1e-12 of 0 counts as 0 (e.g. 0.3 + 0.3 - 0.6 = 5e-17)."""
    return ci is not None and (ci["ci_low"] > TOL or ci["ci_high"] < -TOL)


def verdicts_v2(summary, kappa_low):
    d = summary["metrics"]["MeanRel5"].get("diff_visual_minus_metadata")
    if d is None:
        v2 = "undetermined (no retained queries)"
    elif excludes_zero_v2(d):
        v2 = "difference detected: visual %s metadata on MeanRel@5" % ("higher than" if d["mean"] > 0 else "lower than")
    else:
        v2 = "no difference detected (this is NOT evidence of equivalence)"
    v1 = ("visual above the random-anchor rate (CI entirely > 0)" if summary.get("RQ1", {}).get("verdict_visual_above_chance")
          else "visual not shown to be above the random-anchor rate")
    return {"RQ2_primary": v2, "RQ1_primary": v1,
            "reliability": "TENTATIVE: kappa_w < 0.40 (low reliability)" if kappa_low else
            ("reliability not flagged" if kappa_low is False else "reliability undetermined (kappa undefined)"),
            "note": "mechanical verdicts under the frozen rules + amendment v1; secondary metrics are descriptive only"}


# ----------------------------------------------------------------------------- aggregation (A1, A2)
def aggregate_pair_v2(a, b, c=None):
    """a, b: primary annotators; c: third file. Each int in {0,1,2} or None (abstention). Returns (label|None, status)."""
    num = [x for x in (a, b) if x is not None]
    if len(num) == 2:
        if a == b:
            return a, "agree"
        if c is not None:                                         # adjudication: only between two numeric labels
            if c not in (a, b):
                raise ValueError(f"adjudicator label {c} is neither {a} nor {b}")
            return c, "adjudicated"
        if abs(a - b) == 1:
            return min(a, b), "min_rule"
        return None, "unresolved"
    if len(num) == 1:
        x = num[0]
        if c is None:
            return None, "single_numeric_not_ground_truth"      # A1
        if c == x:                                                # A2: second judgment, never an override
            return x, "second_judgment_agree"
        if abs(c - x) == 1:
            return min(c, x), "second_judgment_min_rule"
        return None, "second_judgment_unresolved"
    if c is not None:
        return None, "single_numeric_not_ground_truth_third_only"
    return None, "missing"


def aggregate_all_v2(pair_ids, la, lb, lc=None):
    final, status = {}, {}
    for p in pair_ids:
        final[p], status[p] = aggregate_pair_v2(la.get(p), lb.get(p), (lc or {}).get(p))
    return final, status


def mean_label_table_v2(pair_ids, la, lb, lc=None):
    out = {}
    for p in pair_ids:
        a, b, c = la.get(p), lb.get(p), (lc or {}).get(p)
        if a is not None and b is not None:
            out[p] = (a + b) / 2
        elif (a is not None) != (b is not None) and c is not None:
            out[p] = ((a if a is not None else b) + c) / 2
        else:
            out[p] = None
    return out


# ----------------------------------------------------------------------------- query sets (A3, A6)
def build_query_table(key_rows, queries):
    qt = E1.build_query_table(key_rows, queries)
    for q, t in qt.items():
        if len(t["anchors"]) != 2:
            raise ValueError(f"query {q}: protocol requires exactly 2 anchors, found {len(t['anchors'])}")
    return qt


def retained_sets_v2(vals):
    """RQ2: both Top-5 lists fully labelled. RQ1: Visual Top-5 + both anchors fully labelled. Independent exclusion rules."""
    rq2 = [q for q in sorted(vals) if vals[q]["visual"]["n_missing"] == 0 and vals[q]["metadata"]["n_missing"] == 0]
    ndcg = [q for q in rq2 if vals[q]["visual"]["NDCG5"] is not None]                    # IDCG = 0 rule kept (N5)
    rq1_visual = [q for q in sorted(vals) if vals[q]["visual"]["n_missing"] == 0]
    rq1 = [q for q in rq1_visual if len(vals[q]["anchors_judged"]) == 2]
    sets = {"P5": rq2, "MeanRel5": rq2, "Hit5": rq2, "NDCG5": ndcg, "RQ1": rq1}
    excl = {"RQ2_excluded_incomplete_Top5": {q: {"visual_missing": vals[q]["visual"]["n_missing"], "metadata_missing": vals[q]["metadata"]["n_missing"]}
                                             for q in sorted(vals) if q not in set(rq2)},
            "RQ2_NDCG5_excluded_IDCG_zero": sorted(set(rq2) - set(ndcg)),
            "RQ1_excluded_visual_Top5_incomplete": sorted(set(vals) - set(rq1_visual)),
            "RQ1_excluded_anchor_unlabelled": sorted(set(rq1_visual) - set(rq1))}
    return sets, excl


def analyze_v2(qt, labels, queries, B=E1.B_DEFAULT, metrics=E1.METRICS, with_rq1=True, with_exp=True):
    vals = E1.per_query_values(qt, labels)
    sets, excl = retained_sets_v2(vals)
    groups = {q: queries[q]["group"] for q in qt}
    weights = {q: queries[q]["stratum_size"] / queries[q]["stratum_queries"] for q in qt}
    by_stratum = lambda qs: dict(Counter(groups[q] for q in qs))  # noqa: E731
    out = {"retained": {k: len(v) for k, v in sets.items()}, "retained_by_stratum": {k: by_stratum(v) for k, v in sets.items()},
           "exclusions": excl, "metrics": {}}
    for m in metrics:
        out["metrics"][m] = E1._block(sets[m], lambda q, m=m: vals[q]["visual"][m], lambda q, m=m: vals[q]["metadata"][m], groups, weights, B)
        if "diff_visual_minus_metadata" in out["metrics"][m]:
            out["metrics"][m]["verdict_diff_excludes_zero"] = excludes_zero_v2(out["metrics"][m]["diff_visual_minus_metadata"])
    if with_exp and "NDCG5" in metrics:
        out["metrics"]["NDCG5_exp_gain_sensitivity"] = E1._block(sets["NDCG5"], lambda q: vals[q]["visual"]["NDCG5_exp"],
                                                                 lambda q: vals[q]["metadata"]["NDCG5_exp"], groups, weights, B)
    if with_rq1:
        qs = sets["RQ1"]
        arate = {q: float(np.mean([a >= 1 for a in vals[q]["anchors_judged"]])) for q in qs}
        diff = {q: vals[q]["visual"]["P5"] - arate[q] for q in qs}
        by = {g: [diff[q] for q in qs if groups[q] == g] for g in sorted({groups[q] for q in qs})}
        ci = E1.stratified_bootstrap(by, B) if qs else None
        out["RQ1"] = {"n_queries": len(qs), "visual_P5_mean": float(np.mean([vals[q]["visual"]["P5"] for q in qs])) if qs else None,
                      "anchor_rate_mean": float(np.mean(list(arate.values()))) if qs else None, "delta": ci,
                      "delta_unstratified_sensitivity": E1.unstratified_bootstrap(list(diff.values()), B),
                      "verdict_visual_above_chance": bool(ci is not None and ci["ci_low"] > TOL)}
    return {"summary": out, "_vals": vals}


# ----------------------------------------------------------------------------- full run
def _agreement(qt, la, lb, B):
    pair_ids = [p for t in qt.values() for p in t["pool"]]
    members = {"all_pool_pairs": set(pair_ids)}
    for m in METHODS:
        members[f"{m}_top5_pairs"] = {p for t in qt.values() for p in t["top5"][m]}
    members["anchor_pairs"] = {p for t in qt.values() for p in t["anchors"]}
    pq = {p: q for q, t in qt.items() for p in t["pool"]}
    return {name: E1.agreement_report([(pq[p], la[p], lb[p]) for p in sorted(S) if la.get(p) is not None and lb.get(p) is not None], B)
            for name, S in members.items()}


def run_evaluation_v2(key_rows, queries, la, lb, lc=None, B=E1.B_DEFAULT, markers=None):
    qt = build_query_table(key_rows, queries)
    pair_ids = [p for t in qt.values() for p in t["pool"]]
    final, status = aggregate_all_v2(pair_ids, la, lb, lc)
    primary = analyze_v2(qt, final, queries, B)
    sens = {}
    for name, tab in (("annotator_A_only", {p: la.get(p) for p in pair_ids}), ("annotator_B_only", {p: lb.get(p) for p in pair_ids})):
        sens[name] = analyze_v2(qt, tab, queries, B, with_rq1=False)["summary"]
    sens["mean_label"] = analyze_v2(qt, mean_label_table_v2(pair_ids, la, lb, lc), queries, B, metrics=("MeanRel5", "NDCG5"), with_rq1=False, with_exp=False)["summary"]
    # transparency: effect of the amendment (SUPERSEDED rules, never used for verdicts)
    final_v1, _ = E1.aggregate_all(pair_ids, la, lb, lc)
    sens["superseded_A1_single_label_fallback_only"] = analyze_v2(qt, final_v1, queries, B)["summary"]
    sens["superseded_protocol_v1_all_rules"] = E1.analyze(qt, final_v1, queries, B)["summary"]
    st = Counter(status.values())
    both = {p for p in pair_ids if la.get(p) is not None and lb.get(p) is not None}
    awaiting = sorted(p for p in pair_ids if status[p] == "single_numeric_not_ground_truth")
    res = {"evaluator_version": EVALUATOR_VERSION, "supersedes": "evaluator-v1", "bootstrap": {"B": B, "seed": E1.SEED},
           "ambiguity_register": AMBIGUITY_REGISTER,
           "label_aggregation": {"status_counts": dict(st), "pairs": len(pair_ids), "pairs_final_missing": sum(v is None for v in final.values()),
                                 "pairs_both_primary_numeric": len(both), "pairs_single_numeric_awaiting_second_judgment": len(awaiting),
                                 "third_file_used": lc is not None,
                                 "third_labels_ignored_because_primary_agree": sum(1 for p in pair_ids if (lc or {}).get(p) is not None and status[p] == "agree"
                                                                                  and (lc or {}).get(p) != final[p])},
           "primary": primary["summary"], "verdicts": None, "sensitivity": sens, "agreement": _agreement(qt, la, lb, B),
           "marker_counts": markers or {}}
    res["verdicts"] = verdicts_v2(primary["summary"], res["agreement"]["all_pool_pairs"]["low_reliability"])
    return res, primary["_vals"], final


# ----------------------------------------------------------------------------- lock / plan / evaluate
def _stage(a, evaldir, outdir):
    import validate_annotations as VA
    tasks = VA.load_tasks(evaldir)
    lock_p = outdir / "labels_lock.json"
    existing = json.loads(lock_p.read_text()) if lock_p.exists() else None
    primary = {"annotator_1": a.a1, "annotator_2": a.a2}
    rep = VA.validate_many(list(primary.values()), "main", tasks, require_complete=True)
    if rep["status"] != "COMPLETE":
        raise SystemExit("primary labels not valid/complete -> no lock:\n" + json.dumps(rep["errors"] or rep["files"], indent=1, ensure_ascii=False)[:3000])
    files = dict(primary)
    if a.third:
        if existing is None:
            raise SystemExit("lock the two primary files first (published order: primary lock -> plan -> third file)")
        if "annotator_3" in existing["files"]:
            raise SystemExit("third file already locked")
        rep3 = VA.validate_many([a.third], "main", tasks, require_complete=False)
        if rep3["status"] == "INVALID":
            raise SystemExit("third file invalid -> no lock:\n" + json.dumps(rep3["errors"], indent=1, ensure_ascii=False)[:2000])
        files["annotator_3"] = a.third
    if existing:
        for k in primary:
            if E1.sha_file(primary[k]) != existing["files"][k]["sha256"]:
                raise SystemExit(f"{k} differs from the existing lock -> BLOCKED")
    return tasks, existing, files, lock_p


def cmd_lock(a, evaldir, outdir):
    tasks, existing, files, lock_p = _stage(a, evaldir, outdir)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = {k: {"path": str(Path(p).resolve()), "sha256": E1.sha_file(p), "annotator_id": E1.read_label_file(Path(p))["__annotator__"]} for k, p in files.items()}
    all_entries = {**(existing["files"] if existing else {}), **entries}
    ids = [v["annotator_id"] for v in all_entries.values()]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"annotators not distinct: {ids}")
    lock = {"evaluator_version": EVALUATOR_VERSION, "private_key_opened": False, "files": all_entries,
            "history": (existing["history"] if existing else []) + [{"utc": now, "added": sorted(set(entries) - set(existing["files"] if existing else {}))}],
            "locked_utc": existing["locked_utc"] if existing else now}
    outdir.mkdir(parents=True, exist_ok=True)
    lock_p.write_text(json.dumps(lock, indent=1))
    print(json.dumps({"locked": sorted(all_entries), "history": lock["history"]}, indent=1))


def cmd_plan(outdir):
    lock_p = outdir / "labels_lock.json"
    if not lock_p.exists():
        raise SystemExit("no labels_lock.json: lock the primary files first")
    lock = json.loads(lock_p.read_text())
    for k, v in lock["files"].items():
        if E1.sha_file(v["path"]) != v["sha256"]:
            raise SystemExit(f"{k}: label file changed after lock -> BLOCKED")
    ra, rb = (E1.read_label_file(Path(lock["files"][k]["path"])) for k in ("annotator_1", "annotator_2"))
    pids = sorted(p for p in ra if p != "__annotator__")
    second, adj = [], []
    for p in pids:
        a, b = ra[p]["label"], rb[p]["label"]
        if (a is None) != (b is None):
            second.append([p, "single_numeric"])
        elif a is not None and a != b:
            adj.append([p, a, b])
    for name, head, rows in (("second_judgment_needed.csv", ["pair_id", "reason"], second), ("adjudication_needed.csv", ["pair_id", "label_annotator_1", "label_annotator_2"], adj)):
        with open(outdir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(head); w.writerows(rows)
    summary = {"second_judgment_needed": len(second), "adjudication_needed": len(adj),
               "both_abstained_not_supplementable": sum(1 for p in pids if ra[p]["label"] is None and rb[p]["label"] is None),
               "note": "No key opened. Second-judgment raters must label BLIND (no labels shown); adjudication shows both labels (REVIEW_REQUIRED N2)."}
    (outdir / "plan_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def cmd_evaluate(a, evaldir, privdir, hmdir, outdir, frozen_manifest):
    if frozen_manifest is not None:
        E1.verify_frozen(frozen_manifest)
    lock_p = outdir / "labels_lock.json"
    if not lock_p.exists():
        raise SystemExit("no labels_lock.json: run 'lock' first (labels must be hashed before the key is opened)")
    lock = json.loads(lock_p.read_text())
    for k, v in lock["files"].items():
        if E1.sha_file(v["path"]) != v["sha256"]:
            raise SystemExit(f"{k}: label file changed after lock -> BLOCKED")
    raw = {k: E1.read_label_file(Path(v["path"])) for k, v in lock["files"].items()}
    lab = {k: {p: r["label"] for p, r in d.items() if p != "__annotator__"} for k, d in raw.items()}
    markers = {k: dict(Counter(r["marker"] for p, r in d.items() if p != "__annotator__")) for k, d in raw.items()}
    markers["uncertain_flags"] = {k: sum(r["uncertain"] for p, r in d.items() if p != "__annotator__") for k, d in raw.items()}
    pm = json.loads((privdir / "pool_manifest.json").read_text())                 # key is opened only now
    if E1.sha_file(privdir / "annotation_key.csv") != pm["annotation_key_sha256"]:
        raise SystemExit("annotation_key.csv does not match pool_manifest.json")
    with open(privdir / "annotation_key.csv", newline="", encoding="utf-8") as f:
        key_rows = [r for r in csv.DictReader(f) if r["set"] == "main"]
    queries = E1.load_queries(hmdir / "query_manifest.csv")
    res, vals, final = run_evaluation_v2(key_rows, queries, lab["annotator_1"], lab["annotator_2"], lab.get("annotator_3"), a.B, markers)
    res["lock"] = lock
    res["evaluator_frozen_verified"] = frozen_manifest is not None
    res["methodology_amendment_sha256"] = E1.sha_file(AMENDMENT) if AMENDMENT.exists() else None
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
    for name in ("lock", "plan", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--evaldir", type=Path, default=W7 / "outputs/evaluation")
        p.add_argument("--outdir", type=Path, default=W7 / "outputs/evaluation_results")
        if name == "lock":
            p.add_argument("--a1", type=Path, required=True); p.add_argument("--a2", type=Path, required=True)
            p.add_argument("--third", type=Path, help="supplementary second-judgment / adjudication file (after the primary lock)")
        if name == "evaluate":
            p.add_argument("--privdir", type=Path, default=W7 / "outputs/evaluation_private")
            p.add_argument("--hmdir", type=Path, default=W7 / "outputs/hm")
            p.add_argument("--B", type=int, default=E1.B_DEFAULT)
            p.add_argument("--frozen-manifest", type=Path, default=V2_MANIFEST)
            p.add_argument("--skip-freeze-check", action="store_true", help="development only; recorded in the results")
    a = ap.parse_args(argv)
    if a.cmd == "lock":
        cmd_lock(a, a.evaldir, a.outdir)
    elif a.cmd == "plan":
        cmd_plan(a.outdir)
    else:
        cmd_evaluate(a, a.evaldir, a.privdir, a.hmdir, a.outdir, None if a.skip_freeze_check else a.frozen_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
