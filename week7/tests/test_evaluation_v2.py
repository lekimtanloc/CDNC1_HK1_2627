"""Tests for evaluator v2 (methodology amendment v1: decisions A1, A2, A3, A6).

SYNTHETIC FIXTURES ONLY, hand-computed expectations, plus a differential test against a reference implementation written
independently inside this file (and scikit-learn for NDCG). No real label is read or created; no research metric is computed.
Run: python3 -I -B -m pytest -q week7/tests/test_evaluation_v2.py -p no:cacheprovider
"""
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
sys.path.insert(0, str(W7 / "src"))
import evaluate_recommendations as E1  # noqa: E402
import evaluate_recommendations_v2 as E2  # noqa: E402

B = 600
NONE = None


# ============================================================ hand-designed fixture FX2
def build_fx2():
    key, la, lb = [], {}, {}

    def add(q, vis, meta, anchors, labels):
        for c in dict.fromkeys(vis + meta + anchors):
            pid = f"{q}-{c}"
            key.append({"pair_id": pid, "set": "main", "query_id": q, "candidate_id": c, "in_visual_top5": str(c in vis),
                        "visual_rank": vis.index(c) + 1 if c in vis else "", "in_metadata_top5": str(c in meta),
                        "metadata_rank": meta.index(c) + 1 if c in meta else "", "is_anchor": str(c in anchors)})
            la[pid], lb[pid] = labels.get(c, (None, None))

    add("qA", ["a1", "a2", "a3", "a4", "a5"], ["a1", "m2", "m3", "m4", "m5"], ["x1", "x2"],
        {"a1": (2, 2), "a2": (1, 1), "a3": (0, 0), "a4": (2, 2), "a5": (1, 1), "m2": (1, 1), "m3": (0, 0), "m4": (0, 0), "m5": (2, 2),
         "x1": (0, 0), "x2": (1, 1)})
    add("qB", ["b1", "b2", "b3", "b4", "b5"], ["b1", "b2", "b3", "b4", "b5"], ["y1", "y2"],          # identical lists
        {"b1": (2, 2), "b2": (1, 1), "b3": (1, 1), "b4": (1, 1), "b5": (0, 0), "y1": (1, 1), "y2": (0, 0)})
    add("qC", ["c1", "c2", "c3", "c4", "c5"], ["d1", "d2", "d3", "d4", "d5"], ["w1", "w2"],           # d2: ONE numeric label (A1)
        {"c1": (1, 1), "c2": (1, 1), "c3": (0, 0), "c4": (0, 0), "c5": (0, 0), "d1": (1, 1), "d2": (1, None), "d3": (0, 0), "d4": (0, 0),
         "d5": (2, 2), "w1": (1, 1), "w2": (1, 1)})
    add("qD", ["e1", "e2", "e3", "e4", "e5"], ["f1", "f2", "f3", "f4", "f5"], ["z1", "z2"],           # one anchor unlabelled (A3); all zeros
        {**{c: (0, 0) for c in ["e1", "e2", "e3", "e4", "e5", "f1", "f2", "f3", "f4", "f5", "z2"]}, "z1": (None, None)})
    add("qE", ["g1", "g2", "g3", "g4", "g5"], ["h1", "h2", "h3", "h4", "h5"], ["u1", "u2"],           # visual g5 unlabelled
        {**{c: (1, 1) for c in ["g1", "g2", "g3", "g4", "h1", "h2", "h3", "h4", "h5", "u1", "u2"]}, "g5": (None, None)})
    queries = {"qA": {"group": "G1", "stratum_size": 10, "stratum_queries": 2}, "qB": {"group": "G1", "stratum_size": 10, "stratum_queries": 2},
               "qC": {"group": "G2", "stratum_size": 6, "stratum_queries": 3}, "qD": {"group": "G2", "stratum_size": 6, "stratum_queries": 3},
               "qE": {"group": "G2", "stratum_size": 6, "stratum_queries": 3}}
    return key, queries, la, lb


@pytest.fixture(scope="module")
def fx():
    return build_fx2()


@pytest.fixture(scope="module")
def res(fx):
    key, queries, la, lb = fx
    return E2.run_evaluation_v2(key, queries, la, lb, None, B=B)[0]


# ============================================================ A1 / A2: aggregation
@pytest.mark.parametrize("a,b,c,expected", [
    (2, 2, None, (2, "agree")), (2, 2, 0, (2, "agree")),                                      # third ignored when primaries agree
    (1, 2, None, (1, "min_rule")), (0, 1, None, (0, "min_rule")), (0, 2, None, (None, "unresolved")),
    (0, 2, 2, (2, "adjudicated")), (0, 2, 0, (0, "adjudicated")), (1, 2, 1, (1, "adjudicated")),
    # A1: one numeric label is NOT ground truth
    (1, None, None, (None, "single_numeric_not_ground_truth")), (None, 2, None, (None, "single_numeric_not_ground_truth")),
    (0, None, None, (None, "single_numeric_not_ground_truth")),
    # A2: an additional numeric label is a SECOND JUDGMENT (two-label rules), never an override
    (1, None, 1, (1, "second_judgment_agree")), (None, 2, 2, (2, "second_judgment_agree")),
    (1, None, 0, (0, "second_judgment_min_rule")), (1, None, 2, (1, "second_judgment_min_rule")),
    (None, 0, 2, (None, "second_judgment_unresolved")), (2, None, 0, (None, "second_judgment_unresolved")),
    (None, None, 2, (None, "single_numeric_not_ground_truth_third_only")), (None, None, None, (None, "missing")),
])
def test_aggregate_pair_v2_table(a, b, c, expected):
    assert E2.aggregate_pair_v2(a, b, c) == expected


def test_A1_old_rule_would_have_used_the_single_label_new_rule_does_not():
    assert E1.aggregate_pair(1, None, None) == (1, "single_label")                  # frozen v1 behaviour (superseded)
    assert E2.aggregate_pair_v2(1, None, None)[0] is None


def test_A2_third_label_never_overrides_a_single_numeric_label():
    """x=2, third=0: v1 'adjudicator if available' would return 0; v2 must NOT return the third label, the pair is unresolved."""
    assert E1.aggregate_pair(2, None, 0) == (0, "single_adjudicated")               # superseded behaviour
    assert E2.aggregate_pair_v2(2, None, 0) == (None, "second_judgment_unresolved")
    for x in (0, 1, 2):
        for y in (0, 1, 2):
            lab, st = E2.aggregate_pair_v2(x, None, y)
            assert st.startswith("second_judgment") and lab in (None, min(x, y), x)
            if lab is not None:
                assert lab <= max(x, y) and lab == (x if x == y else min(x, y))


def test_A2_adjudication_must_choose_one_of_the_two_labels():
    with pytest.raises(ValueError):
        E2.aggregate_pair_v2(0, 2, 1)
    assert E2.aggregate_pair_v2(0, 2, 2) == (2, "adjudicated")
    assert E2.aggregate_pair_v2(None, 2, 0)[1] == "second_judgment_unresolved"       # not an adjudication: no error either


def test_abstention_never_becomes_zero():
    for combo in [(None, None, None), (1, None, None), (None, 0, None), (None, None, 1), (0, 2, None)]:
        lab, _ = E2.aggregate_pair_v2(*combo)
        assert lab is None


def test_mean_label_v2_needs_two_independent_judgments():
    t = E2.mean_label_table_v2(["p1", "p2", "p3", "p4", "p5"], {"p1": 0, "p2": 2, "p3": None, "p4": 1, "p5": None},
                               {"p1": 2, "p2": None, "p3": None, "p4": None, "p5": None}, {"p2": 1, "p3": 2, "p5": 0})
    assert t == {"p1": 1.0, "p2": 1.5, "p3": None, "p4": None, "p5": None}


# ============================================================ A3 / A6: query sets
def test_query_sets_are_separate_for_RQ1_and_RQ2(res):
    p = res["primary"]
    assert p["retained"] == {"P5": 3, "MeanRel5": 3, "Hit5": 3, "NDCG5": 2, "RQ1": 3}
    assert p["exclusions"]["RQ2_excluded_incomplete_Top5"] == {"qC": {"visual_missing": 0, "metadata_missing": 1},
                                                              "qE": {"visual_missing": 1, "metadata_missing": 0}}
    assert p["exclusions"]["RQ2_NDCG5_excluded_IDCG_zero"] == ["qD"]
    assert p["exclusions"]["RQ1_excluded_visual_Top5_incomplete"] == ["qE"]
    assert p["exclusions"]["RQ1_excluded_anchor_unlabelled"] == ["qD"]
    # the two populations differ: qC only in RQ1 ; qD only in RQ2
    assert p["retained_by_stratum"]["RQ1"] == {"G1": 2, "G2": 1} and p["retained_by_stratum"]["P5"] == {"G1": 2, "G2": 1}


def test_A6_RQ2_requires_both_lists_complete_no_one_missing_tolerance(fx):
    key, queries, la, lb = fx
    qt = E2.build_query_table(key, queries)
    final, _ = E2.aggregate_all_v2([p for t in qt.values() for p in t["pool"]], la, lb)
    vals = E1.per_query_values(qt, final)
    sets, _ = E2.retained_sets_v2(vals)
    assert "qC" not in sets["MeanRel5"] and "qE" not in sets["MeanRel5"]               # exactly one missing position each
    v1_sets, _ = E1.retained_sets(vals)
    assert "qE" in v1_sets["MeanRel5"]                                                 # frozen v1 would have tolerated it
    assert sets["Hit5"] == sets["MeanRel5"] == sets["P5"]


def test_A6_RQ1_ignores_metadata_completeness_and_RQ2_ignores_anchors(fx):
    key, queries, la, lb = fx
    qt = E2.build_query_table(key, queries)
    final, _ = E2.aggregate_all_v2([p for t in qt.values() for p in t["pool"]], la, lb)
    sets, _ = E2.retained_sets_v2(E1.per_query_values(qt, final))
    assert "qC" in sets["RQ1"] and "qC" not in sets["P5"]        # metadata incomplete: still valid for RQ1 (visual + anchors complete)
    assert "qD" not in sets["RQ1"] and "qD" in sets["P5"]        # anchor unlabelled: invalid for RQ1, valid for RQ2


def test_A3_RQ1_needs_all_five_visual_labels_and_both_anchors(fx):
    key, queries, la, lb = fx
    la2, lb2 = dict(la), dict(lb)
    la2["qA-x2"] = lb2["qA-x2"] = None                                                  # one of qA's two anchors becomes unlabelled
    r = E2.run_evaluation_v2(key, queries, la2, lb2, None, B=B)[0]["primary"]
    assert "qA" in r["exclusions"]["RQ1_excluded_anchor_unlabelled"] and r["retained"]["RQ1"] == 2
    assert r["retained"]["P5"] == 3                                                     # RQ2 unaffected by anchors
    la3, lb3 = dict(la), dict(lb)
    la3["qB-b3"] = lb3["qB-b3"] = None                                                  # a visual position unlabelled in qB
    r = E2.run_evaluation_v2(key, queries, la3, lb3, None, B=B)[0]["primary"]
    assert "qB" in r["exclusions"]["RQ1_excluded_visual_Top5_incomplete"] and r["retained"]["RQ1"] == 2 and r["retained"]["P5"] == 2


def test_same_query_set_inside_each_comparison(res):
    for name, b in res["primary"]["metrics"].items():
        assert b["visual"]["n"] == b["metadata"]["n"] == b["diff_visual_minus_metadata"]["n"] == b["n_queries"], name
    for sens in res["sensitivity"].values():
        for name, b in sens["metrics"].items():
            if b["n_queries"]:
                assert b["visual"]["n"] == b["metadata"]["n"] == b["diff_visual_minus_metadata"]["n"] == b["n_queries"], name


def test_anchor_count_is_enforced(fx):
    key, queries, la, lb = fx
    bad = [r for r in key if r["pair_id"] != "qA-x2"]
    with pytest.raises(ValueError, match="exactly 2 anchors"):
        E2.build_query_table(bad, queries)


# ============================================================ hand-computed values
def test_RQ2_values_hand_computed(res):
    m = res["primary"]["metrics"]
    # RQ2 set = {qA, qB, qD}
    assert m["P5"]["visual"]["mean"] == pytest.approx((0.8 + 0.8 + 0) / 3) and m["P5"]["metadata"]["mean"] == pytest.approx((0.6 + 0.8 + 0) / 3)
    assert m["P5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(0.2 / 3)
    assert m["MeanRel5"]["visual"]["mean"] == pytest.approx((1.2 + 1.0 + 0) / 3) and m["MeanRel5"]["metadata"]["mean"] == pytest.approx((1.0 + 1.0 + 0) / 3)
    assert m["MeanRel5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(0.2 / 3)
    assert m["Hit5"]["visual"]["mean"] == pytest.approx(2 / 3) and m["Hit5"]["diff_visual_minus_metadata"]["mean"] == 0
    # pooled NDCG@5 over {qA, qB} (qD: IDCG = 0)
    assert m["NDCG5"]["n_queries"] == 2
    assert m["NDCG5"]["visual"]["mean"] == pytest.approx(0.8328628097580959) and m["NDCG5"]["metadata"]["mean"] == pytest.approx(0.7861544043940374)
    assert m["NDCG5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(0.046708405364058514)


def test_secondary_aggregations_hand_computed(res):
    mr = res["primary"]["metrics"]["MeanRel5"]
    assert mr["balanced_by_stratum"]["diff"] == pytest.approx(0.05)                     # G1 (qA .2, qB 0) -> .1 ; G2 (qD 0) -> 0
    assert mr["catalogue_weighted"]["diff"] == pytest.approx(1.0 / 12)                  # weights qA 5, qB 5, qD 2
    assert mr["per_stratum_descriptive"]["G1"]["n"] == 2 and mr["per_stratum_descriptive"]["G2"]["n"] == 1


def test_RQ1_values_hand_computed(res):
    rq1 = res["primary"]["RQ1"]                                                         # set = {qA, qB, qC}
    assert rq1["n_queries"] == 3
    assert rq1["visual_P5_mean"] == pytest.approx((0.8 + 0.8 + 0.4) / 3) and rq1["anchor_rate_mean"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert rq1["delta"]["mean"] == pytest.approx(0.0, abs=1e-12)                        # deltas .3, .3, -.6
    assert rq1["verdict_visual_above_chance"] is False


def test_label_status_counts_and_awaiting_pairs(res):
    la = res["label_aggregation"]
    assert la["pairs"] == 11 + 7 + 12 + 12 + 12 and la["pairs_single_numeric_awaiting_second_judgment"] == 1
    sc = la["status_counts"]
    assert sc["single_numeric_not_ground_truth"] == 1 and sc["missing"] == 2 and sc["agree"] == la["pairs"] - 3
    assert la["third_file_used"] is False


# ============================================================ second judgment end to end (A1 + A2)
def test_second_judgment_agree_brings_the_query_into_RQ2(fx):
    key, queries, la, lb = fx
    r, _, final = E2.run_evaluation_v2(key, queries, la, lb, {"qC-d2": 1}, B=B)
    assert final["qC-d2"] == 1 and r["label_aggregation"]["status_counts"]["second_judgment_agree"] == 1
    p, m = r["primary"], r["primary"]["metrics"]
    assert p["retained"] == {"P5": 4, "MeanRel5": 4, "Hit5": 4, "NDCG5": 3, "RQ1": 3}
    # qC now: visual [1,1,0,0,0], metadata [1,1,0,0,2]  -> P5 .4/.6 ; MeanRel .4/.8 ; Hit 0/1
    assert m["P5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx((0.2 + 0 - 0.2 + 0) / 4)
    assert m["MeanRel5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx((0.2 + 0 - 0.4 + 0) / 4)
    assert m["Hit5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(-1 / 4)
    assert m["NDCG5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(-0.034178156094464875)


def test_second_judgment_disagreement_is_not_overridden(fx):
    key, queries, la, lb = fx
    la2, lb2 = dict(la), dict(lb)
    la2["qC-d2"], lb2["qC-d2"] = 0, None                      # single numeric 0 ; third says 2 -> |d| = 2 -> unresolved, NOT 2
    r, _, final = E2.run_evaluation_v2(key, queries, la2, lb2, {"qC-d2": 2}, B=B)
    assert final["qC-d2"] is None and r["label_aggregation"]["status_counts"]["second_judgment_unresolved"] == 1
    assert r["primary"]["retained"]["P5"] == 3


def test_adjudication_resolves_a_disagreement_between_two_numeric_labels(fx):
    key, queries, la, lb = fx
    la2, lb2 = dict(la), dict(lb)
    la2["qB-b2"], lb2["qB-b2"] = 2, 0                         # disagreement by 2 -> unresolved without adjudicator -> qB leaves RQ2/RQ1
    r0 = E2.run_evaluation_v2(key, queries, la2, lb2, None, B=B)[0]["primary"]
    assert r0["retained"]["P5"] == 2 and "qB" in r0["exclusions"]["RQ2_excluded_incomplete_Top5"]
    r1, _, final = E2.run_evaluation_v2(key, queries, la2, lb2, {"qB-b2": 0}, B=B)
    assert final["qB-b2"] == 0 and r1["primary"]["retained"]["P5"] == 3


def test_third_label_is_ignored_when_primaries_agree_and_is_counted(fx):
    key, queries, la, lb = fx
    r, _, final = E2.run_evaluation_v2(key, queries, la, lb, {"qA-a1": 0}, B=B)
    assert final["qA-a1"] == 2 and r["label_aggregation"]["third_labels_ignored_because_primary_agree"] == 1


# ============================================================ amendment effect is visible (superseded rules kept as sensitivity)
def test_superseded_rules_are_reported_only_as_sensitivity(res):
    s = res["sensitivity"]
    assert {"superseded_A1_single_label_fallback_only", "superseded_protocol_v1_all_rules"} <= set(s)
    # v1 rules: qC counts (single label used) and qE (1 missing tolerated) -> larger sets than the amended primary
    v1 = s["superseded_protocol_v1_all_rules"]["retained"]
    assert v1["P5"] == 5 and res["primary"]["retained"]["P5"] == 3
    only_a1 = s["superseded_A1_single_label_fallback_only"]["retained"]
    assert only_a1["P5"] == 4                                                           # qC enters, qE still excluded (amended A6)
    assert res["verdicts"]["RQ2_primary"] and "ambiguity_register" in res
    assert res["ambiguity_register"]["A1"]["status"] == "RESOLVED_BY_AMENDMENT"


def test_register_lists_resolved_open_and_review_required():
    reg = E2.AMBIGUITY_REGISTER
    assert {k for k, v in reg.items() if v["status"] == "RESOLVED_BY_AMENDMENT"} == {"A1", "A2", "A3", "A6"}
    assert {k for k, v in reg.items() if v["status"] == "REVIEW_REQUIRED"} == {"N1", "N2", "N3", "N4"}
    assert {f"A{i}" for i in range(1, 14)} <= set(reg)


# ============================================================ differential test vs an independent reference
def ref_final(a, b, c):
    if a is not None and b is not None:
        if a == b:
            return a
        if c is not None:
            assert c in (a, b)
            return c
        return min(a, b) if abs(a - b) == 1 else None
    x = a if a is not None else b
    if x is not None and c is not None:
        return x if c == x else (min(x, c) if abs(x - c) == 1 else None)
    return None


def random_fixture(seed):
    rng = random.Random(seed)
    key, la, lb, lc, queries = [], {}, {}, {}, {}
    groups = ["G1", "G2", "G3"]
    for i in range(14):
        q = f"q{i}"
        g = groups[i % 3]
        queries[q] = {"group": g, "stratum_size": 20 + (i % 3) * 5, "stratum_queries": 4 + (i % 2)}
        vis = [f"v{k}" for k in range(5)]
        meta = vis[:rng.randint(0, 3)] + [f"m{k}" for k in range(5)]
        meta = meta[:5] if len(meta) >= 5 else meta + [f"m{k}" for k in range(5 - len(meta))]
        meta = list(dict.fromkeys(meta))[:5]
        while len(meta) < 5:
            meta.append(f"mm{len(meta)}")
        anc = ["x1", "x2"]
        for c in dict.fromkeys(vis + meta + anc + [f"o{k}" for k in range(rng.randint(0, 2))]):
            pid = f"{q}-{c}"
            key.append({"pair_id": pid, "set": "main", "query_id": q, "candidate_id": c, "in_visual_top5": str(c in vis),
                        "visual_rank": vis.index(c) + 1 if c in vis else "", "in_metadata_top5": str(c in meta),
                        "metadata_rank": meta.index(c) + 1 if c in meta else "", "is_anchor": str(c in anc)})
            base = rng.choice([0, 1, 2])
            a = None if rng.random() < 0.12 else (base if rng.random() < 0.75 else rng.choice([0, 1, 2]))
            b = None if rng.random() < 0.12 else (base if rng.random() < 0.75 else rng.choice([0, 1, 2]))
            la[pid], lb[pid] = a, b
            if a is not None and b is not None and a != b and rng.random() < 0.6:
                lc[pid] = rng.choice([a, b])                              # valid adjudication
            elif (a is None) != (b is None) and rng.random() < 0.6:
                lc[pid] = rng.choice([0, 1, 2])                           # second judgment
            elif a == b and rng.random() < 0.1:
                lc[pid] = rng.choice([0, 1, 2])                           # ignored third label
    return key, queries, la, lb, lc


def ref_results(key, queries, la, lb, lc):
    by_q = {}
    for r in key:
        by_q.setdefault(r["query_id"], []).append(r)
    final = {r["pair_id"]: ref_final(la[r["pair_id"]], lb[r["pair_id"]], lc.get(r["pair_id"])) for r in key}
    out = {"P5": {}, "MeanRel5": {}, "Hit5": {}, "NDCG5": {}, "RQ1": {}}
    for q, rows in by_q.items():
        top = {m: [r["pair_id"] for r in sorted((r for r in rows if r[f"in_{m}_top5"] == "True"), key=lambda r: int(r[f"{m}_rank"]))] for m in ("visual", "metadata")}
        rel = {m: [final[p] for p in top[m]] for m in top}
        anchors = [final[r["pair_id"]] for r in rows if r["is_anchor"] == "True"]
        if all(x is not None for x in rel["visual"]) and all(x is not None for x in rel["metadata"]):
            for name, f in (("P5", lambda r: sum(x >= 1 for x in r) / 5), ("MeanRel5", lambda r: sum(r) / 5), ("Hit5", lambda r: float(2 in r))):
                out[name][q] = (f(rel["visual"]), f(rel["metadata"]))
            pool = [final[r["pair_id"]] for r in rows if final[r["pair_id"]] is not None]
            if sum(pool) > 0:
                scores = {}
                for m in rel:
                    sc = np.zeros(len(pool))
                    idx = {r["pair_id"]: i for i, r in enumerate([r for r in rows if final[r["pair_id"]] is not None])}
                    for rank, p in enumerate(top[m]):
                        sc[idx[p]] = 10 - rank
                    scores[m] = ndcg_score([pool], [sc], k=5)
                out["NDCG5"][q] = (scores["visual"], scores["metadata"])
        if all(x is not None for x in rel["visual"]) and all(x is not None for x in anchors):
            out["RQ1"][q] = sum(x >= 1 for x in rel["visual"]) / 5 - sum(x >= 1 for x in anchors) / 2
    return out


@pytest.mark.parametrize("seed", range(25))
def test_differential_against_independent_reference(seed):
    key, queries, la, lb, lc = random_fixture(seed)
    ref = ref_results(key, queries, la, lb, lc)
    r, _, _ = E2.run_evaluation_v2(key, queries, la, lb, lc, B=60)
    m = r["primary"]["metrics"]
    for name in ("P5", "MeanRel5", "Hit5", "NDCG5"):
        vals = ref[name]
        assert m[name]["n_queries"] == len(vals), (name, seed)
        if vals:
            assert m[name]["visual"]["mean"] == pytest.approx(np.mean([v[0] for v in vals.values()]), abs=1e-9)
            assert m[name]["metadata"]["mean"] == pytest.approx(np.mean([v[1] for v in vals.values()]), abs=1e-9)
            assert m[name]["diff_visual_minus_metadata"]["mean"] == pytest.approx(np.mean([v[0] - v[1] for v in vals.values()]), abs=1e-9)
    rq1 = r["primary"]["RQ1"]
    assert rq1["n_queries"] == len(ref["RQ1"])
    if ref["RQ1"]:
        assert rq1["delta"]["mean"] == pytest.approx(np.mean(list(ref["RQ1"].values())), abs=1e-9)


# ============================================================ lock / plan / evaluate flow (synthetic package, tmp only)
@pytest.fixture()
def pkg(tmp_path, fx):
    key, queries, la, lb = fx
    ev, priv, hm = tmp_path / "evaluation", tmp_path / "private", tmp_path / "hm"
    for d in (ev, priv, hm):
        d.mkdir()
    pids = sorted(r["pair_id"] for r in key)
    with open(ev / "annotation_tasks.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair_id", "set", "query_image", "candidate_image", "o1", "o2", "o3"])
        for i, p in enumerate(pids):
            w.writerow([p, "main", "i", "c", i, i, i])
    (ev / "annotation_manifest.json").write_text(json.dumps({"hashes": {"annotation_tasks.csv": E1.sha_file(ev / "annotation_tasks.csv")}}))
    with open(priv / "annotation_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    (priv / "pool_manifest.json").write_text(json.dumps({"annotation_key_sha256": E1.sha_file(priv / "annotation_key.csv")}))
    with open(hm / "query_manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["article_id", "product_group_name", "stratum_size", "stratum_queries"])
        for q, v in queries.items():
            w.writerow([q, v["group"], v["stratum_size"], v["stratum_queries"]])

    def labfile(name, annot, labels, only=None, blank_missing=False):
        p = tmp_path / name
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc"])
            for pid in pids:
                if only is not None and pid not in only:
                    continue
                v = labels.get(pid)
                w.writerow([pid, "main", annot, "U" if v is None else v, "", "", ""])
        return p
    return {"ev": ev, "priv": priv, "hm": hm, "out": tmp_path / "results", "a1": labfile("annotator_01.csv", "annotator_01", la),
            "a2": labfile("annotator_02.csv", "annotator_02", lb), "labfile": labfile, "tmp": tmp_path}


def ns(**kw):
    return type("A", (), kw)()


def test_flow_lock_plan_third_evaluate(pkg):
    E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=None), pkg["ev"], pkg["out"])
    lock = json.loads((pkg["out"] / "labels_lock.json").read_text())
    assert lock["private_key_opened"] is False and set(lock["files"]) == {"annotator_1", "annotator_2"}
    E2.cmd_plan(pkg["out"])
    need = list(csv.DictReader(open(pkg["out"] / "second_judgment_needed.csv")))
    assert [r["pair_id"] for r in need] == ["qC-d2"] and list(csv.DictReader(open(pkg["out"] / "adjudication_needed.csv"))) == []
    plan = json.loads((pkg["out"] / "plan_summary.json").read_text())
    assert plan["second_judgment_needed"] == 1 and plan["adjudication_needed"] == 0 and plan["both_abstained_not_supplementable"] == 2
    third = pkg["labfile"]("annotator_03.csv", "annotator_03", {"qC-d2": 1}, only={"qC-d2"})
    E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=third), pkg["ev"], pkg["out"])
    lock = json.loads((pkg["out"] / "labels_lock.json").read_text())
    assert set(lock["files"]) == {"annotator_1", "annotator_2", "annotator_3"} and len(lock["history"]) == 2
    E2.cmd_evaluate(ns(B=300), pkg["ev"], pkg["priv"], pkg["hm"], pkg["out"], None)
    r = json.loads((pkg["out"] / "results.json").read_text())
    assert r["evaluator_version"] == "evaluator-v2" and r["label_aggregation"]["third_file_used"] is True
    assert r["primary"]["retained"] == {"P5": 4, "MeanRel5": 4, "Hit5": 4, "NDCG5": 3, "RQ1": 3}
    assert r["primary"]["metrics"]["MeanRel5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(-0.05)
    assert r["evaluator_frozen_verified"] is False and "ambiguity_register" in r
    assert len(list(csv.DictReader(open(pkg["out"] / "per_query_metrics.csv")))) == 10


def test_evaluate_without_third_file_matches_in_memory(pkg, fx):
    E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=None), pkg["ev"], pkg["out"])
    E2.cmd_evaluate(ns(B=300), pkg["ev"], pkg["priv"], pkg["hm"], pkg["out"], None)
    r = json.loads((pkg["out"] / "results.json").read_text())
    assert r["primary"]["retained"] == {"P5": 3, "MeanRel5": 3, "Hit5": 3, "NDCG5": 2, "RQ1": 3}
    assert r["label_aggregation"]["pairs_single_numeric_awaiting_second_judgment"] == 1


def test_third_file_cannot_be_added_before_primary_lock_or_after_change(pkg):
    third = pkg["labfile"]("annotator_03.csv", "annotator_03", {"qC-d2": 1}, only={"qC-d2"})
    with pytest.raises(SystemExit, match="primary files first"):
        E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=third), pkg["ev"], pkg["out"])
    E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=None), pkg["ev"], pkg["out"])
    pkg["a1"].write_text(pkg["a1"].read_text().replace(",2,,,", ",1,,,", 1))
    with pytest.raises(SystemExit, match="differs from the existing lock"):
        E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=third), pkg["ev"], pkg["out"])


def test_third_file_with_bad_label_or_same_annotator_is_rejected(pkg):
    E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=None), pkg["ev"], pkg["out"])
    bad = pkg["labfile"]("annotator_03.csv", "annotator_03", {"qC-d2": 7}, only={"qC-d2"})
    with pytest.raises(SystemExit, match="third file invalid"):
        E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=bad), pkg["ev"], pkg["out"])
    same = pkg["labfile"]("annotator_03b.csv", "annotator_01", {"qC-d2": 1}, only={"qC-d2"})
    with pytest.raises(SystemExit, match="not distinct"):
        E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=same), pkg["ev"], pkg["out"])


def test_evaluate_blocks_without_lock_after_label_change_and_on_key_tamper(pkg):
    with pytest.raises(SystemExit, match="lock"):
        E2.cmd_evaluate(ns(B=100), pkg["ev"], pkg["priv"], pkg["hm"], pkg["out"], None)
    E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=None), pkg["ev"], pkg["out"])
    (pkg["priv"] / "annotation_key.csv").write_text((pkg["priv"] / "annotation_key.csv").read_text() + "\n")
    with pytest.raises(SystemExit, match="pool_manifest"):
        E2.cmd_evaluate(ns(B=100), pkg["ev"], pkg["priv"], pkg["hm"], pkg["out"], None)
    pkg["a2"].write_text(pkg["a2"].read_text().replace(",1,,,", ",0,,,", 1))
    with pytest.raises(SystemExit, match="changed after lock"):
        E2.cmd_evaluate(ns(B=100), pkg["ev"], pkg["priv"], pkg["hm"], pkg["out"], None)


def test_plan_lists_disagreements_for_adjudication(pkg, fx):
    key, queries, la, lb = fx
    la2 = dict(la); la2["qB-b2"] = 2; lb2 = dict(lb); lb2["qB-b2"] = 0
    a1 = pkg["labfile"]("annotator_01.csv", "annotator_01", la2); a2 = pkg["labfile"]("annotator_02.csv", "annotator_02", lb2)
    E2.cmd_lock(ns(a1=a1, a2=a2, third=None), pkg["ev"], pkg["out"])
    E2.cmd_plan(pkg["out"])
    adj = list(csv.DictReader(open(pkg["out"] / "adjudication_needed.csv")))
    assert adj == [{"pair_id": "qB-b2", "label_annotator_1": "2", "label_annotator_2": "0"}]


def test_plan_and_lock_never_touch_the_key(pkg):
    (pkg["priv"] / "annotation_key.csv").unlink()
    E2.cmd_lock(ns(a1=pkg["a1"], a2=pkg["a2"], third=None), pkg["ev"], pkg["out"])
    E2.cmd_plan(pkg["out"])
    src = (W7 / "src/evaluate_recommendations_v2.py").read_text()
    before_eval = src.split("def cmd_evaluate")[0]
    assert "annotation_key" not in before_eval.replace("annotation_key.csv does not match", "")


# ============================================================ repository guards
def test_v1_evaluator_and_manifest_untouched_and_no_real_labels():
    man = json.loads((W7 / "outputs/evaluation/evaluator_manifest.json").read_text())
    assert E1.sha_file(W7 / "src/evaluate_recommendations.py") == man["files"]["week7/src/evaluate_recommendations.py"]["sha256"]
    E1.verify_frozen(W7 / "outputs/evaluation/evaluator_manifest.json")                  # all v1 frozen files unchanged
    assert list(W7.rglob("annotator_0*.csv")) == [] and not (W7 / "outputs/evaluation_results").exists()


def test_excludes_zero_tolerance_handles_float_noise():
    assert E2.excludes_zero_v2({"ci_low": 5e-17, "ci_high": 0.2}) is False          # 0.3 + 0.3 - 0.6 noise must not create a 'detection'
    assert E2.excludes_zero_v2({"ci_low": -0.2, "ci_high": -3e-13}) is False
    assert E2.excludes_zero_v2({"ci_low": 1e-6, "ci_high": 0.2}) and E2.excludes_zero_v2({"ci_low": -0.2, "ci_high": -1e-6})
    assert E2.excludes_zero_v2(None) is False
