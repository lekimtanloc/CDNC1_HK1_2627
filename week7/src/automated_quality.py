"""Week 7 — automated quality PROXIES for the frozen Top-5 lists (visual V-R50-FULLPAD vs metadata M-ATTR-TFIDF).

These are attribute-agreement proxies computed from H&M catalogue metadata. They are NOT human relevance labels and NOT
ground truth. Several attributes are used directly by the metadata baseline's ranking score, which gives that baseline a
structural (circular) advantage on those proxies; every metric carries the weight the baseline gives to its attribute.

Reads only frozen inputs (product universe, query manifest, both Top-5 files, articles.csv). Writes only to
week7/outputs/automated_quality/. Never opens the private key, never reads/creates annotation labels, does not re-run ResNet-50.

Run (repo root):  python3 -I -B week7/src/automated_quality.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
HM = W7 / "outputs/hm"
OUT = W7 / "outputs/automated_quality"
sys.path.insert(0, str(Path(__file__).parent))
import evaluate_recommendations as E1  # noqa: E402  (frozen v1: only its stratified bootstrap is reused, unchanged)

B, SEED, K = 10000, 42, 5
METHODS = ("visual", "metadata")
# metric -> (attribute, weight of that attribute in the metadata baseline score, short note)
MATCH = {
    "ProductTypeMatch@5": ("product_type_no", 0.30, "used by metadata ranking"),
    "ProductGroupMatch@5": ("product_group_name", 0.10, "used by metadata ranking"),
    "GarmentGroupMatch@5": ("garment_group_no", 0.10, "used by metadata ranking"),
    "ColourMatch@5": ("colour_group_code", 0.15, "used by metadata ranking"),
    "PerceivedColourMasterMatch@5": ("perceived_colour_master_id", 0.05, "used by metadata ranking (secondary colour proxy)"),
    "GraphicalAppearanceMatch@5": ("graphical_appearance_no", 0.10, "used by metadata ranking"),
    "DepartmentMatch@5": ("department_no", 0.0, "NOT used by metadata ranking (organisational field)"),
    "SectionMatch@5": ("section_no", 0.0, "NOT used by metadata ranking (organisational field)"),
}
DIVERSITY_ATTRS = ("product_type_no", "colour_group_code", "graphical_appearance_no")
METRICS = list(MATCH) + ["CategoryMismatch@5", "UniqueProducts@5", "AttributeDiversity@5"]
GROUPS = ["Garment Upper body", "Garment Lower body", "Shoes", "Garment Full body", "Accessories", "Bags"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def verify_inputs() -> dict:
    pm = json.loads((HM / "protocol_manifest.json").read_text())
    bad = [f for f, v in pm["files"].items() if sha(ROOT / f) != v["sha256"]]
    if bad:
        raise SystemExit(f"frozen protocol files changed -> BLOCKED: {bad}")
    vlog = json.loads((HM / "visual_run_log.json").read_text())["outputs_sha256"]
    if sha(HM / "visual_top5.csv") != vlog["visual_top5.csv"]:
        raise SystemExit("visual_top5.csv differs from visual_run_log.json")
    mlog = json.loads((HM / "metadata_run_log.json").read_text())["output_sha256"]
    if sha(HM / "metadata_top5.csv") != mlog["metadata_top5.csv"]:
        raise SystemExit("metadata_top5.csv differs from metadata_run_log.json")
    return {"protocol_files_verified": len(pm["files"]), "visual_top5_sha256": vlog["visual_top5.csv"], "metadata_top5_sha256": mlog["metadata_top5.csv"]}


def load():
    arts = {r["article_id"]: r for r in read(ROOT / "dataset_sources/hm-dataset/articles.csv")}
    U = read(HM / "product_universe.csv")
    prod = {}
    for u in U:
        a = arts[u["article_id"]]
        assert a["product_code"] == u["product_code"]
        prod[u["article_id"]] = {**u, "department_no": a["department_no"], "section_no": a["section_no"]}
    Q = read(HM / "query_manifest.csv")
    tops = {}
    for m in METHODS:
        d = defaultdict(list)
        for r in read(HM / f"{m}_top5.csv"):
            d[r["query_id"]].append((int(r["rank"]), r["candidate_id"]))
        tops[m] = {q: [c for _, c in sorted(v)] for q, v in d.items()}
    # validity
    qids = [q["article_id"] for q in Q]
    assert len(qids) == 50 and len(set(qids)) == 50
    for m in METHODS:
        assert set(tops[m]) == set(qids), m
        for q, lst in tops[m].items():
            assert len(lst) == K and len(set(lst)) == K and q not in lst and all(c in prod for c in lst), (m, q)
            assert prod[q]["product_code"] not in {prod[c]["product_code"] for c in lst}
    return prod, Q, tops


def missing(v) -> bool:
    return str(v).strip().lower() in ("", "-1", "unknown", "undefined")


def list_metrics(q: dict, cands: list[dict]) -> dict:
    out = {}
    for name, (attr, _, _) in MATCH.items():
        ok = [c for c in cands if not missing(q[attr]) and not missing(c[attr])]
        out[name] = sum(c[attr] == q[attr] for c in ok) / len(ok) if ok else None
    out["CategoryMismatch@5"] = 1 - out["ProductGroupMatch@5"]
    out["UniqueProducts@5"] = len({c["product_code"] for c in cands}) / len(cands)
    out["AttributeDiversity@5"] = float(np.mean([(len({c[a] for c in cands}) - 1) / (len(cands) - 1) for a in DIVERSITY_ATTRS]))
    return out


def random_expectation(q: dict, pool: list[dict]) -> dict:
    """Exact expectation for 5 candidates drawn uniformly without replacement from the 144-product pool."""
    n = len(pool)
    out = {}
    for name, (attr, _, _) in MATCH.items():
        out[name] = sum(c[attr] == q[attr] for c in pool) / n
    out["CategoryMismatch@5"] = 1 - out["ProductGroupMatch@5"]
    out["UniqueProducts@5"] = 1.0                                   # one representative per product_code: always distinct
    div = []
    for a in DIVERSITY_ATTRS:
        cnt = Counter(c[a] for c in pool)
        e_distinct = sum(1 - math.comb(n - k, K) / math.comb(n, K) for k in cnt.values())
        div.append((e_distinct - 1) / (K - 1))
    out["AttributeDiversity@5"] = float(np.mean(div))
    return out


def ceiling(q: dict, pool: list[dict]) -> dict:
    """Best achievable value for any Top-5 from this pool: min(5, #pool products sharing the attribute)/5 (floor for mismatch)."""
    out = {name: min(K, sum(c[attr] == q[attr] for c in pool)) / K for name, (attr, _, _) in MATCH.items()}
    out["CategoryMismatch@5"] = 1 - out["ProductGroupMatch@5"]
    out["UniqueProducts@5"] = None
    out["AttributeDiversity@5"] = None
    return out


def fmt(x):
    return "" if x is None else f"{x:.6f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs()
    prod, Q, tops = load()
    ids = list(prod)
    groups = {q["article_id"]: q["product_group_name"] for q in Q}
    per = {}                                                          # (query, method) -> metrics
    for q in Q:
        qid = q["article_id"]
        for m in METHODS:
            per[(qid, m)] = list_metrics(prod[qid], [prod[c] for c in tops[m][qid]])
        per[(qid, "random_expected")] = random_expectation(prod[qid], [prod[c] for c in ids if c != qid])
        per[(qid, "ceiling")] = ceiling(prod[qid], [prod[c] for c in ids if c != qid])
    # ---- per-query file
    with open(OUT / "quality_metrics_per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "product_group", "query_product_type", "method", *METRICS])
        for q in Q:
            for m in (*METHODS, "random_expected", "ceiling"):
                w.writerow([q["article_id"], q["product_group_name"], q["product_type_name"], m, *[fmt(per[(q["article_id"], m)][k]) for k in METRICS]])
    # ---- overall comparison with paired stratified bootstrap (proxy only)
    qids = [q["article_id"] for q in Q]
    by_stratum = lambda vals: {g: [vals[x] for x in qids if groups[x] == g] for g in sorted(set(groups.values()))}  # noqa: E731
    comp = []
    for k in METRICS:
        v = {x: per[(x, "visual")][k] for x in qids}
        mt = {x: per[(x, "metadata")][k] for x in qids}
        rnd = {x: per[(x, "random_expected")][k] for x in qids}
        cei = [per[(x, "ceiling")][k] for x in qids if per[(x, "ceiling")][k] is not None]
        d_vm = {x: v[x] - mt[x] for x in qids}
        d_vr = {x: v[x] - rnd[x] for x in qids}
        d_mr = {x: mt[x] - rnd[x] for x in qids}
        ci_vm, ci_vr, ci_mr = (E1.stratified_bootstrap(by_stratum(d), B, SEED) for d in (d_vm, d_vr, d_mr))
        attr, wgt, note = MATCH.get(k, (None, None, None))
        if k == "CategoryMismatch@5":
            attr, wgt, note = "product_group_name", 0.10, "1 - ProductGroupMatch@5; used by metadata ranking"
        elif k == "UniqueProducts@5":
            attr, wgt, note = "product_code", None, "validity check (one representative per product_code)"
        elif k == "AttributeDiversity@5":
            attr, wgt, note = "+".join(DIVERSITY_ATTRS), None, "mean over attributes of (distinct values - 1)/4; neither better nor worse per se"
        comp.append({"metric": k, "attribute": attr, "metadata_baseline_weight": "" if wgt is None else wgt, "note": note, "n_queries": len(qids),
                     "visual_mean": fmt(np.mean(list(v.values()))), "metadata_mean": fmt(np.mean(list(mt.values()))), "random_expected_mean": fmt(np.mean(list(rnd.values()))),
                     "ceiling_mean": fmt(float(np.mean(cei))) if cei else "",
                     "diff_visual_minus_metadata": fmt(ci_vm["mean"]), "diff_vm_ci_low": fmt(ci_vm["ci_low"]), "diff_vm_ci_high": fmt(ci_vm["ci_high"]),
                     "diff_visual_minus_random": fmt(ci_vr["mean"]), "diff_vr_ci_low": fmt(ci_vr["ci_low"]), "diff_vr_ci_high": fmt(ci_vr["ci_high"]),
                     "diff_metadata_minus_random": fmt(ci_mr["mean"]), "diff_mr_ci_low": fmt(ci_mr["ci_low"]), "diff_mr_ci_high": fmt(ci_mr["ci_high"]),
                     "queries_visual_gt_metadata": sum(d_vm[x] > 1e-12 for x in qids), "queries_equal": sum(abs(d_vm[x]) <= 1e-12 for x in qids),
                     "queries_visual_lt_metadata": sum(d_vm[x] < -1e-12 for x in qids)})
    with open(OUT / "quality_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0])); w.writeheader(); w.writerows(comp)
    # ---- by product group (descriptive)
    with open(OUT / "quality_by_category.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["product_group", "n_queries", "metric", "visual_mean", "metadata_mean", "random_expected_mean", "ceiling_mean", "diff_visual_minus_metadata"])
        for g in GROUPS:
            qs = [x for x in qids if groups[x] == g]
            for k in METRICS:
                vm, mm, rm = (float(np.mean([per[(x, m)][k] for x in qs])) for m in ("visual", "metadata", "random_expected"))
                ce = [per[(x, "ceiling")][k] for x in qs if per[(x, "ceiling")][k] is not None]
                w.writerow([g, len(qs), k, fmt(vm), fmt(mm), fmt(rm), fmt(float(np.mean(ce))) if ce else "", fmt(vm - mm)])
    # ---- candidate-level diagnostics for the visual lists (what kind of 'errors'?)
    diag = {}
    for m in METHODS:
        c_tot = Counter()
        for q in Q:
            qp = prod[q["article_id"]]
            for c in tops[m][q["article_id"]]:
                cp = prod[c]
                same_type = cp["product_type_no"] == qp["product_type_no"]
                same_group = cp["product_group_name"] == qp["product_group_name"]
                c_tot["candidates"] += 1
                c_tot["same_type"] += same_type
                c_tot["same_group_other_type"] += same_group and not same_type
                c_tot["other_group"] += not same_group
                c_tot["same_type_other_colour"] += same_type and cp["colour_group_code"] != qp["colour_group_code"]
                c_tot["same_type_other_pattern"] += same_type and cp["graphical_appearance_no"] != qp["graphical_appearance_no"]
                c_tot["same_type_same_colour"] += same_type and cp["colour_group_code"] == qp["colour_group_code"]
        diag[m] = dict(c_tot)
    cross = Counter()
    for q in Q:
        qp = prod[q["article_id"]]
        for c in tops["visual"][q["article_id"]]:
            cp = prod[c]
            if cp["product_group_name"] != qp["product_group_name"]:
                cross[f"{qp['product_group_name']} -> {cp['product_group_name']}"] += 1
    high_mm = [{"query_id": x, "group": groups[x], "type": prod[x]["product_type_name"], "colour": prod[x]["colour_group_name"],
                "visual_category_mismatch": per[(x, "visual")]["CategoryMismatch@5"],
                "visual_candidate_groups": [prod[c]["product_group_name"] for c in tops["visual"][x]],
                "visual_candidate_types": [prod[c]["product_type_name"] for c in tops["visual"][x]]}
               for x in qids if per[(x, "visual")]["CategoryMismatch@5"] >= 0.4]
    mc = {r["query_id"]: r for r in read(HM / "method_comparison.csv")}
    disjoint = [{"query_id": x, "group": groups[x], "type": prod[x]["product_type_name"], "overlap": int(mc[x]["overlap_count"]),
                 "visual_type_match": per[(x, "visual")]["ProductTypeMatch@5"], "metadata_type_match": per[(x, "metadata")]["ProductTypeMatch@5"]}
                for x in qids if int(mc[x]["overlap_count"]) == 0]
    details = {"candidate_level": diag, "visual_cross_group_flows": dict(cross.most_common()), "queries_visual_category_mismatch_ge_0_4": high_mm,
               "queries_with_disjoint_top5": disjoint}
    (OUT / "quality_details.json").write_text(json.dumps(details, indent=1, ensure_ascii=False))
    # ---- AI judgment step: availability record (no judgment produced)
    (OUT / "ai_judgment_status.json").write_text(json.dumps({
        "label": "AI_JUDGMENT_EXPLORATORY", "performed": False,
        "reason": ["no independent local vision-language model is installed (Hugging Face cache holds only an audio model; no ollama/llama.cpp/mlx-vlm/open_clip)",
                   "the coding agent itself is not an independent judge: it built both methods, wrote the protocol, has access to the private method key and cannot fix seed/decoding",
                   "an external API judge would send catalogue images to a third-party service and needs explicit approval of the research owner",
                   "0/1/2 scores must never be inferred from metadata (would be circular and fake)"],
        "files_created": []}, indent=1))
    outs = ["quality_metrics_per_query.csv", "quality_comparison.csv", "quality_by_category.csv", "quality_details.json", "ai_judgment_status.json"]
    import evaluate_recommendations
    (OUT / "quality_manifest.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "AUTOMATED_PROXY_ONLY",
        "not_ground_truth": True, "human_labels_used": False, "ai_judgments_used": False, "resnet_rerun": False, "rankings_modified": False,
        "inputs": {**inputs, "product_universe.csv": sha(HM / "product_universe.csv"), "query_manifest.csv": sha(HM / "query_manifest.csv"),
                   "articles.csv": sha(ROOT / "dataset_sources/hm-dataset/articles.csv"), "method_comparison.csv": sha(HM / "method_comparison.csv")},
        "script": {"path": "week7/src/automated_quality.py", "sha256": sha(Path(__file__))},
        "bootstrap": {"B": B, "seed": SEED, "stratified_by": "product_group_name", "implementation": "evaluate_recommendations.stratified_bootstrap (frozen v1, unchanged)",
                      "evaluator_v1_sha256": sha(evaluate_recommendations.__file__)},
        "metrics": {k: {"attribute": MATCH[k][0], "metadata_baseline_weight": MATCH[k][1], "note": MATCH[k][2]} for k in MATCH},
        "outputs_sha256": {n: sha(OUT / n) for n in outs},
        "environment": {"python": platform.python_version(), "numpy": np.__version__}}, indent=1))
    print(json.dumps({r["metric"]: [r["visual_mean"], r["metadata_mean"], r["random_expected_mean"], r["diff_visual_minus_metadata"],
                                    r["diff_vm_ci_low"], r["diff_vm_ci_high"]] for r in comp}, indent=0))
    print(json.dumps({"candidate_level": diag, "cross": dict(cross.most_common()), "high_mismatch": len(high_mm), "disjoint": len(disjoint)}, indent=0, ensure_ascii=False))


if __name__ == "__main__":
    main()
