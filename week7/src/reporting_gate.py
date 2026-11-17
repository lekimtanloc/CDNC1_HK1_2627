"""Reporting gate (decisions N3 + N4): is each research question's valid-query set large enough to be reported as a MAIN result?

    python3 -I -B week7/src/reporting_gate.py [--results week7/outputs/evaluation_results/results.json] [--out-dir DIR]

Reads from results.json ONLY query counts, per-group counts and exclusion lists (never a metric value, CI or verdict) plus
query_manifest.csv for the group universe. Thresholds come from research_decisions.json (N3: >= 40 of 50 valid queries per RQ and
>= 4 valid queries in EVERY product group; a pragmatic acceptance threshold, NOT statistical-power evidence).
N4: the RQ1 and RQ2 sets are separate; their sizes, group distributions and overlap are published here and their metrics must not be
compared as if they came from the same sample.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
DECISIONS = W7 / "outputs/methodology/research_decisions.json"
COUNT_KEYS = ("retained", "retained_by_stratum", "exclusions")      # the only parts of results.json this module reads


def load_thresholds(path: Path = DECISIONS) -> dict:
    d = json.loads(path.read_text())
    if d.get("status") != "APPROVED_BY_RESEARCH_OWNER":
        raise SystemExit("research_decisions.json is not APPROVED_BY_RESEARCH_OWNER: thresholds not valid")
    n3 = d["decisions"]["N3"]
    return {"min_valid_queries": int(n3["min_valid_queries_per_rq"]), "min_per_group": int(n3["min_queries_per_product_group"]),
            "decision_status": d["status"]}


def compute_gate(results: dict, query_groups: dict, th: dict) -> dict:
    prim = {k: results["primary"][k] for k in COUNT_KEYS}            # explicit allowlist: no metric field is ever touched
    groups = sorted(set(query_groups.values()))
    total = len(query_groups)
    ret, by, ex = prim["retained"], prim["retained_by_stratum"], prim["exclusions"]
    if not (ret["MeanRel5"] == ret["P5"] == ret["Hit5"]):
        raise SystemExit("inconsistent RQ2 sets in results.json")
    rq2_excl = set(ex["RQ2_excluded_incomplete_Top5"])
    rq1_excl = set(ex["RQ1_excluded_visual_Top5_incomplete"]) | set(ex["RQ1_excluded_anchor_unlabelled"])
    allq = set(query_groups)
    rq2_valid, rq1_valid = allq - rq2_excl, allq - rq1_excl
    if len(rq2_valid) != ret["MeanRel5"] or len(rq1_valid) != ret["RQ1"]:
        raise SystemExit("exclusion lists do not reproduce the retained counts: results.json is inconsistent")
    out = {"decision_status": th["decision_status"], "thresholds": {"min_valid_queries_per_RQ": th["min_valid_queries"], "min_queries_per_product_group": th["min_per_group"]},
           "note_thresholds": "pragmatic acceptance threshold; not evidence of statistical power", "queries_total": total, "metric_values_read": False, "RQ": {}}
    for rq, valid in (("RQ1", rq1_valid), ("RQ2", rq2_valid)):
        per_group = {g: sum(1 for q in valid if query_groups[q] == g) for g in groups}
        reasons = []
        if len(valid) < th["min_valid_queries"]:
            reasons.append(f"valid queries {len(valid)} < {th['min_valid_queries']}")
        low = {g: n for g, n in per_group.items() if n < th["min_per_group"]}
        if low:
            reasons.append(f"groups below {th['min_per_group']} queries: {low}")
        out["RQ"][rq] = {"valid_queries": len(valid), "of": total, "per_group": per_group, "meets_threshold": not reasons, "reasons": reasons,
                         "status": "REPORTABLE_AS_MAIN_RESULT" if not reasons else "BELOW_THRESHOLD_DESCRIPTIVE_ONLY"}
    out["RQ"]["RQ2"]["pooled_NDCG5_valid_queries_informational"] = ret["NDCG5"]
    group_sizes = {g: sum(1 for q in allq if query_groups[q] == g) for g in groups}
    out["composition_N4"] = {"group_sizes": group_sizes, "RQ1_only": len(rq1_valid - rq2_valid), "RQ2_only": len(rq2_valid - rq1_valid),
                             "both": len(rq1_valid & rq2_valid), "neither": len(allq - rq1_valid - rq2_valid),
                             "exclusion_reasons": {"RQ2_incomplete_Top5_queries": len(rq2_excl), "RQ2_NDCG5_IDCG_zero": len(ex["RQ2_NDCG5_excluded_IDCG_zero"]),
                                                   "RQ1_visual_Top5_incomplete": len(ex["RQ1_excluded_visual_Top5_incomplete"]),
                                                   "RQ1_anchor_unlabelled": len(ex["RQ1_excluded_anchor_unlabelled"])},
                             "statement": "RQ1 and RQ2 use separate valid-query sets; their metrics must not be compared as if they came from the same sample."}
    out["overall"] = ("MAIN_RESULTS_REPORTABLE_FOR_BOTH_RQ" if all(v["meets_threshold"] for v in out["RQ"].values())
                      else "AT_LEAST_ONE_RQ_BELOW_THRESHOLD: report that RQ as descriptive/exploratory with its query count and distribution")
    return out


def to_markdown(g: dict) -> str:
    L = ["# Reporting gate — N3 / N4", "", f"Trạng thái quyết định: **{g['decision_status']}**. Ngưỡng: ≥ {g['thresholds']['min_valid_queries_per_RQ']}/{g['queries_total']} query hợp lệ cho từng RQ "
         f"và ≥ {g['thresholds']['min_queries_per_product_group']} query trong mỗi nhóm sản phẩm (ngưỡng nghiệm thu thực dụng, **không phải bằng chứng statistical power**).", "",
         "Tệp này chỉ dùng số lượng query; không đọc giá trị metric nào.", "", "| RQ | query hợp lệ | kết luận |", "|---|---|---|"]
    for rq, v in g["RQ"].items():
        L.append(f"| {rq} | {v['valid_queries']} / {v['of']} | **{v['status']}**" + (f" — {'; '.join(v['reasons'])}" if v["reasons"] else "") + " |")
    groups = sorted(g["composition_N4"]["group_sizes"])
    L += ["", "## Phân bố theo nhóm sản phẩm (N4)", "", "| Nhóm | query được chọn | RQ1 hợp lệ | RQ2 hợp lệ |", "|---|---|---|---|"]
    for gr in groups:
        L.append(f"| {gr} | {g['composition_N4']['group_sizes'][gr]} | {g['RQ']['RQ1']['per_group'][gr]} | {g['RQ']['RQ2']['per_group'][gr]} |")
    c = g["composition_N4"]
    L += ["", f"Giao nhau: cả hai {c['both']}; chỉ RQ1 {c['RQ1_only']}; chỉ RQ2 {c['RQ2_only']}; không RQ nào {c['neither']}.", "",
          f"Lý do loại: {json.dumps(c['exclusion_reasons'], ensure_ascii=False)}.", "", f"**{c['statement']}**", "", f"Tổng: `{g['overall']}`", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=W7 / "outputs/evaluation_results/results.json")
    ap.add_argument("--hmdir", type=Path, default=W7 / "outputs/hm")
    ap.add_argument("--decisions", type=Path, default=DECISIONS)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    with open(a.hmdir / "query_manifest.csv", newline="", encoding="utf-8") as f:
        groups = {r["article_id"]: r["product_group_name"] for r in csv.DictReader(f)}
    g = compute_gate(json.loads(a.results.read_text()), groups, load_thresholds(a.decisions))
    out = a.out_dir or a.results.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "reporting_gate.json").write_text(json.dumps(g, indent=1, ensure_ascii=False))
    (out / "reporting_gate.md").write_text(to_markdown(g))
    print(to_markdown(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
