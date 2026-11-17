"""Tests for week7/src/automated_quality.py (automated attribute PROXIES, not human relevance).

Independent re-computation from the raw CSV files (no import of the module's metric code), exact random expectation checked by
Monte Carlo, ceiling invariants, bootstrap cross-check with the frozen protocol_lib, determinism, and guards that no label,
key or AI judgment is touched or produced.
Run: python3 -I -B -m pytest -q week7/tests/test_automated_quality.py -p no:cacheprovider
"""
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
HM = W7 / "outputs/hm"
AQ = W7 / "outputs/automated_quality"
sys.path.insert(0, str(W7 / "scripts"))
import protocol_lib as PL  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module", autouse=True)
def run_once():
    frozen = {p: sha(p) for p in [HM / "visual_top5.csv", HM / "metadata_top5.csv", HM / "product_universe.csv", HM / "query_manifest.csv"]}
    subprocess.run([sys.executable, "-I", "-B", str(W7 / "src/automated_quality.py")], check=True, capture_output=True, cwd=ROOT)
    yield
    assert {p: sha(p) for p in frozen} == frozen                       # inputs never modified


@pytest.fixture(scope="module")
def data():
    arts = {r["article_id"]: r for r in rd(ROOT / "dataset_sources/hm-dataset/articles.csv")}
    U = {r["article_id"]: {**r, "department_no": arts[r["article_id"]]["department_no"], "section_no": arts[r["article_id"]]["section_no"]}
         for r in rd(HM / "product_universe.csv")}
    Q = rd(HM / "query_manifest.csv")
    tops = {}
    for m in ("visual", "metadata"):
        d = defaultdict(list)
        for r in rd(HM / f"{m}_top5.csv"):
            d[r["query_id"]].append((int(r["rank"]), r["candidate_id"]))
        tops[m] = {q: [c for _, c in sorted(v)] for q, v in d.items()}
    per = {(r["query_id"], r["method"]): r for r in rd(AQ / "quality_metrics_per_query.csv")}
    comp = {r["metric"]: r for r in rd(AQ / "quality_comparison.csv")}
    return U, Q, tops, per, comp


ATTR = {"ProductTypeMatch@5": "product_type_no", "ProductGroupMatch@5": "product_group_name", "GarmentGroupMatch@5": "garment_group_no",
        "ColourMatch@5": "colour_group_code", "PerceivedColourMasterMatch@5": "perceived_colour_master_id",
        "GraphicalAppearanceMatch@5": "graphical_appearance_no", "DepartmentMatch@5": "department_no", "SectionMatch@5": "section_no"}


def test_inputs_valid_50_queries_250_rows_each(data):
    U, Q, tops, per, _ = data
    assert len(Q) == 50
    for m in ("visual", "metadata"):
        assert len(tops[m]) == 50 and sum(len(v) for v in tops[m].values()) == 250
        for q, lst in tops[m].items():
            assert len(set(lst)) == 5 and q not in lst
    assert len(per) == 50 * 4                                            # visual, metadata, random_expected, ceiling


def test_match_metrics_independent_recompute(data):
    U, Q, tops, per, comp = data
    for k, a in ATTR.items():
        for m in ("visual", "metadata"):
            vals = []
            for q in Q:
                qid = q["article_id"]
                v = sum(U[c][a] == U[qid][a] for c in tops[m][qid]) / 5
                assert abs(float(per[(qid, m)][k]) - v) < 1e-6, (k, m, qid)
                vals.append(v)
            assert abs(float(comp[k][f"{m}_mean"]) - np.mean(vals)) < 1e-6


def test_category_mismatch_unique_and_diversity(data):
    U, Q, tops, per, _ = data
    for q in Q:
        qid = q["article_id"]
        for m in ("visual", "metadata"):
            r = per[(qid, m)]
            assert abs(float(r["CategoryMismatch@5"]) - (1 - float(r["ProductGroupMatch@5"]))) < 1e-6
            assert float(r["UniqueProducts@5"]) == 1.0
            c = [U[x] for x in tops[m][qid]]
            div = np.mean([(len({x[a] for x in c}) - 1) / 4 for a in ("product_type_no", "colour_group_code", "graphical_appearance_no")])
            assert abs(float(r["AttributeDiversity@5"]) - div) < 1e-6


def test_random_expectation_exact_and_monte_carlo(data):
    U, Q, _, per, _ = data
    rng = random.Random(7)
    for q in Q[:6]:
        qid = q["article_id"]
        pool = [U[c] for c in U if c != qid]
        exp_type = sum(c["product_type_no"] == U[qid]["product_type_no"] for c in pool) / len(pool)
        assert abs(float(per[(qid, "random_expected")]["ProductTypeMatch@5"]) - exp_type) < 1e-6
        sims = []
        for _ in range(20000):
            s = rng.sample(pool, 5)
            sims.append(np.mean([(len({x[a] for x in s}) - 1) / 4 for a in ("product_type_no", "colour_group_code", "graphical_appearance_no")]))
        assert abs(np.mean(sims) - float(per[(qid, "random_expected")]["AttributeDiversity@5"])) < 0.01


def test_ceiling_bounds_both_methods(data):
    U, Q, tops, per, _ = data
    for q in Q:
        qid = q["article_id"]
        for k, a in ATTR.items():
            n_same = sum(U[c][a] == U[qid][a] for c in U if c != qid)
            cei = min(5, n_same) / 5
            assert abs(float(per[(qid, "ceiling")][k]) - cei) < 1e-6
            for m in ("visual", "metadata"):
                assert float(per[(qid, m)][k]) <= cei + 1e-6, (qid, k, m)
    assert {q["query_product_type"] for (qid, m), q in per.items() if m == "ceiling" and float(q["ProductTypeMatch@5"]) == 0} == {"Dungarees", "Polo shirt"}


def test_paired_difference_and_bootstrap_match_frozen_reference(data):
    U, Q, tops, per, comp = data
    groups = {q["article_id"]: q["product_group_name"] for q in Q}
    for k in ("ProductTypeMatch@5", "CategoryMismatch@5", "DepartmentMatch@5"):
        d = {qid: float(per[(qid, "visual")][k]) - float(per[(qid, "metadata")][k]) for qid in groups}
        by = {g: [d[x] for x in sorted(groups) if groups[x] == g] for g in sorted(set(groups.values()))}
        point, lo, hi = PL.paired_bootstrap({g: [d[x] for x in [q["article_id"] for q in Q] if groups[x] == g] for g in sorted(set(groups.values()))}, b=10000, seed=42)
        r = comp[k]
        assert abs(float(r["diff_visual_minus_metadata"]) - np.mean(list(d.values()))) < 1e-6
        assert abs(float(r["diff_vm_ci_low"]) - lo) < 1e-6 and abs(float(r["diff_vm_ci_high"]) - hi) < 1e-6
        assert float(r["diff_vm_ci_low"]) <= float(r["diff_visual_minus_metadata"]) <= float(r["diff_vm_ci_high"])
        assert int(r["queries_visual_gt_metadata"]) + int(r["queries_equal"]) + int(r["queries_visual_lt_metadata"]) == 50


def test_by_category_consistent_with_per_query(data):
    _, Q, _, per, _ = data
    rows = rd(AQ / "quality_by_category.csv")
    for r in rows:
        qs = [q["article_id"] for q in Q if q["product_group_name"] == r["product_group"]]
        assert int(r["n_queries"]) == len(qs)
        v = np.mean([float(per[(x, "visual")][r["metric"]]) for x in qs])
        assert abs(float(r["visual_mean"]) - v) < 1e-6
    assert sum(int(r["n_queries"]) for r in rows if r["metric"] == "ProductTypeMatch@5") == 50


def test_details_candidate_counts(data):
    U, Q, tops, _, _ = data
    d = json.loads((AQ / "quality_details.json").read_text())
    for m in ("visual", "metadata"):
        cl = d["candidate_level"][m]
        assert cl["candidates"] == 250 and cl["same_type"] + cl["same_group_other_type"] + cl["other_group"] == 250
        same_type = sum(U[c]["product_type_no"] == U[q["article_id"]]["product_type_no"] for q in Q for c in tops[m][q["article_id"]])
        assert cl["same_type"] == same_type
    assert sum(d["visual_cross_group_flows"].values()) == d["candidate_level"]["visual"]["other_group"]


def test_deterministic_rerun():
    names = ["quality_metrics_per_query.csv", "quality_comparison.csv", "quality_by_category.csv", "quality_details.json"]
    before = {n: sha(AQ / n) for n in names}
    subprocess.run([sys.executable, "-I", "-B", str(W7 / "src/automated_quality.py")], check=True, capture_output=True, cwd=ROOT)
    assert {n: sha(AQ / n) for n in names} == before


def test_manifest_and_status_flags():
    man = json.loads((AQ / "quality_manifest.json").read_text())
    assert man["status"] == "AUTOMATED_PROXY_ONLY" and man["not_ground_truth"] and not man["human_labels_used"] and not man["ai_judgments_used"]
    assert not man["resnet_rerun"] and not man["rankings_modified"]
    for n, h in man["outputs_sha256"].items():
        assert sha(AQ / n) == h
    ai = json.loads((AQ / "ai_judgment_status.json").read_text())
    assert ai["label"] == "AI_JUDGMENT_EXPLORATORY" and ai["performed"] is False and ai["files_created"] == []


def test_no_labels_key_or_ai_judgments_touched():
    src = (W7 / "src/automated_quality.py").read_text()
    for forbidden in ("annotation_key", "evaluation_private", "pool_manifest", "annotator_0", "annotation_tasks"):
        assert forbidden not in src
    assert list(W7.rglob("annotator_0*.csv")) == [] and not (W7 / "outputs/evaluation_results").exists()
    assert not [p for p in AQ.iterdir() if "judgment" in p.name and p.name != "ai_judgment_status.json"]


def test_frozen_manifests_unchanged():
    for m in ("outputs/hm/protocol_manifest.json", "outputs/evaluation/evaluator_manifest.json", "outputs/methodology/evaluator_v2_manifest.json",
              "outputs/methodology/methodology_manifest.json", "outputs/methodology/subset_workflow_manifest.json"):
        d = json.loads((W7 / m).read_text())
        assert [f for f, v in d["files"].items() if sha(ROOT / f) != v["sha256"]] == [], m
