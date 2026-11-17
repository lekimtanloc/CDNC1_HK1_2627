"""Tests for week7/src/evaluate_recommendations.py.

All data here are SYNTHETIC FIXTURES with hand-computed expectations. No real annotation file is read or created, and
nothing computed from fixtures is a research result (outputs go to pytest tmp dirs only).
Independent cross-checks: scikit-learn (ndcg_score, cohen_kappa_score) and the frozen week7/scripts/protocol_lib.py.
Run: python3 -I -B -m pytest -q week7/tests/test_evaluation.py -p no:cacheprovider
"""
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, ndcg_score

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
sys.path.insert(0, str(W7 / "src"))
sys.path.insert(0, str(W7 / "scripts"))
import evaluate_recommendations as E  # noqa: E402
import protocol_lib as PL  # noqa: E402
import validate_annotations as VA  # noqa: E402

B = 2000  # small B for speed; B=10000 is the default and is exercised once below


# ============================================================ fixture
def build_fixture():
    """4 queries, hand-designed (see comments). Returns key_rows, queries, la, lb."""
    key, la, lb = [], {}, {}

    def add(q, vis, meta, anchors, labels):
        """labels: {cand: (a, b)}; absent cand -> (None, None)."""
        cands = list(dict.fromkeys(vis + meta + anchors))
        for c in cands:
            pid = f"{q}-{c}"
            key.append({"pair_id": pid, "set": "main", "query_id": q, "candidate_id": c,
                        "in_visual_top5": str(c in vis), "visual_rank": vis.index(c) + 1 if c in vis else "",
                        "in_metadata_top5": str(c in meta), "metadata_rank": meta.index(c) + 1 if c in meta else "",
                        "is_anchor": str(c in anchors)})
            a, b = labels.get(c, (None, None))
            la[pid], lb[pid] = a, b

    # qA: shared candidate a1 in both Top-5; one 'U' (a5: annotator B has no label)
    add("qA", ["a1", "a2", "a3", "a4", "a5"], ["a1", "m2", "m3", "m4", "m5"], ["x1", "x2"],
        {"a1": (2, 2), "a2": (1, 1), "a3": (0, 0), "a4": (2, 2), "a5": (1, None), "m2": (1, 1), "m3": (0, 0), "m4": (0, 0),
         "m5": (2, 2), "x1": (0, 0), "x2": (1, 1)})
    # qB: IDENTICAL Top-5 for both methods; b2 differs by 1 (min rule); b3 differs by 2 (unresolved -> missing)
    add("qB", ["b1", "b2", "b3", "b4", "b5"], ["b1", "b2", "b3", "b4", "b5"], ["y1", "y2"],
        {"b1": (2, 2), "b2": (2, 1), "b3": (0, 2), "b4": (1, 1), "b5": (0, 0), "y1": (1, 1), "y2": (0, 0)})
    # qC: visual has TWO unlabelled positions (c2, c3) -> whole query dropped
    add("qC", ["c1", "c2", "c3", "c4", "c5"], ["c1", "e2", "e3", "e4", "e5"], ["w1", "w2"],
        {"c1": (1, 1), "c4": (0, 0), "c5": (0, 0), "e2": (1, 1), "e3": (2, 2), "e4": (0, 0), "e5": (0, 0), "w1": (0, 0), "w2": (1, 1)})
    # qD: every label 0 -> IDCG = 0
    add("qD", ["d1", "d2", "d3", "d4", "d5"], ["d1", "n2", "n3", "n4", "n5"], ["z1", "z2"],
        {c: (0, 0) for c in ["d1", "d2", "d3", "d4", "d5", "n2", "n3", "n4", "n5", "z1", "z2"]})
    queries = {"qA": {"group": "G1", "stratum_size": 10, "stratum_queries": 2},
               "qB": {"group": "G1", "stratum_size": 10, "stratum_queries": 2},
               "qC": {"group": "G2", "stratum_size": 6, "stratum_queries": 2},
               "qD": {"group": "G2", "stratum_size": 6, "stratum_queries": 2}}
    return key, queries, la, lb


@pytest.fixture(scope="module")
def fx():
    return build_fixture()


@pytest.fixture(scope="module")
def result(fx):
    key, queries, la, lb = fx
    return E.run_evaluation(key, queries, la, lb, None, B=B)


# ============================================================ label parsing / aggregation
@pytest.mark.parametrize("a,b,adj,expected", [
    (2, 2, None, (2, "agree")), (0, 0, 2, (0, "agree")),
    (1, 2, None, (1, "min_rule")), (2, 1, None, (1, "min_rule")), (0, 1, None, (0, "min_rule")),
    (0, 2, None, (None, "unresolved")), (2, 0, None, (None, "unresolved")),
    (0, 2, 2, (2, "adjudicated")), (0, 2, 0, (0, "adjudicated")), (1, 2, 1, (1, "adjudicated")),
    (1, None, None, (1, "single_label")), (None, 0, None, (0, "single_label")),
    (1, None, 2, (2, "single_adjudicated")),
    (None, None, None, (None, "missing")), (None, None, 2, (None, "missing")),
])
def test_aggregate_pair_all_protocol_cases(a, b, adj, expected):
    assert E.aggregate_pair(a, b, adj) == expected


def test_adjudicator_must_choose_one_of_the_two_labels():
    with pytest.raises(ValueError):
        E.aggregate_pair(0, 2, 1)


def test_unresolved_and_missing_are_never_zero():
    final, status = E.aggregate_all(["p"], {"p": 0}, {"p": 2})
    assert final["p"] is None and status["p"] == "unresolved"


@pytest.mark.parametrize("raw,exp", [("0", (0, "ok")), ("1", (1, "ok")), ("2", (2, "ok")), ("U", (None, "U")), ("u", (None, "U")),
                                     ("", (None, "blank")), (" 2 ", (2, "ok")), (None, (None, "blank"))])
def test_parse_label_ok(raw, exp):
    assert E.parse_label(raw) == exp


@pytest.mark.parametrize("raw", ["3", "-1", "1.5", "x", "two", "00"])
def test_parse_label_rejects_invalid(raw):
    with pytest.raises(ValueError):
        E.parse_label(raw)


def test_read_label_file_rejects_duplicates_and_mixed_annotators(tmp_path):
    def w(name, rows):
        p = tmp_path / name
        with open(p, "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc"]); cw.writerows(rows)
        return p
    ok = E.read_label_file(w("ok.csv", [["p1", "main", "annotator_01", "2", "", "", ""], ["p2", "main", "annotator_01", "U", "1", "", ""]]))
    assert ok["p1"]["label"] == 2 and ok["p2"]["label"] is None and ok["p2"]["marker"] == "U" and ok["p2"]["uncertain"] and ok["__annotator__"] == "annotator_01"
    with pytest.raises(ValueError, match="duplicate"):
        E.read_label_file(w("dup.csv", [["p1", "main", "a", "1", "", "", ""], ["p1", "main", "a", "2", "", "", ""]]))
    with pytest.raises(ValueError, match="annotator_id"):
        E.read_label_file(w("mix.csv", [["p1", "main", "a", "1", "", "", ""], ["p2", "main", "b", "1", "", "", ""]]))
    with pytest.raises(ValueError, match="invalid label"):
        E.read_label_file(w("bad.csv", [["p1", "main", "a", "3", "", "", ""]]))


# ============================================================ position metrics (hand computed)
def test_precision_meanrel_hit_hand_values():
    r = [2, 1, 0, 2, 1]
    assert E.precision5(r) == 0.8 and E.meanrel5(r) == 1.2 and E.hit5(r) == 1.0
    r = [0, 0, 1, 1, 0]
    assert E.precision5(r) == 0.4 and E.meanrel5(r) == 0.4 and E.hit5(r) == 0.0
    assert E.precision5([0] * 5) == 0.0 and E.hit5([1, 1, 1, 1, 1]) == 0.0 and E.meanrel5([2] * 5) == 2.0


def test_missing_positions_use_judged_denominator_never_zero():
    r = [2, None, 1, 0, 1]                       # 4 judged: P5 = 3/4, Mean = 4/4
    assert E.precision5(r) == 0.75 and E.meanrel5(r) == 1.0
    assert E.hit5(r) is None                     # Hit excluded when any position missing (A11)
    assert E.hit5([2, None, None, None, None]) is None
    assert E.precision5([None] * 5) is None and E.meanrel5([None] * 5) is None


def test_pooled_ndcg_hand_values():
    pool = [2, 1, 0, 2, 1, 1, 0, 0, 2, 0, 1]     # ideal = [2,2,2,1,1]: IDCG = 5.07938887...
    assert E.dcg([2, 2, 2, 1, 1]) == pytest.approx(5.079388872, abs=1e-9)
    assert E.pooled_ndcg5([2, 1, 0, 2, 1], pool) == pytest.approx(3.879135677 / 5.079388872, abs=1e-9)   # 0.763701259
    assert E.pooled_ndcg5([2, 1, 0, 0, 2], pool) == pytest.approx(3.404635368 / 5.079388872, abs=1e-9)   # 0.670284448
    assert E.pooled_ndcg5([2, 2, 2, 1, 1], pool) == pytest.approx(1.0)                                  # ideal list
    exp = E.pooled_ndcg5([2, 1, 0, 2, 1], pool, "exp")
    idcg = 3 / 1 + 3 / math.log2(3) + 3 / 2 + 1 / math.log2(5) + 1 / math.log2(6)
    assert exp == pytest.approx((3 + 1 / math.log2(3) + 0 + 3 / math.log2(5) + 1 / math.log2(6)) / idcg)


def test_ndcg_idcg_zero_and_short_pool_and_missing():
    assert E.pooled_ndcg5([0] * 5, [0] * 9) is None                       # IDCG = 0 -> undefined, not 0 and not 1
    assert E.pooled_ndcg5([0] * 5, [0, None, 0, None, 0]) is None
    assert E.pooled_ndcg5([2, 1, None, 0, 0], [2, 1, 0]) is None           # missing Top-5 position
    assert E.pooled_ndcg5([1, 0, 0, 0, 0], [1, None, None, None, None, None]) == pytest.approx(1.0)   # A4: fewer than 5 judged
    assert 0 <= E.pooled_ndcg5([0, 0, 1, 0, 0], [0, 0, 1, 2, 2]) <= 1


def test_ndcg_matches_sklearn_on_random_pools():
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(7, 12)
        pool = [rng.choice([0, 1, 2]) for _ in range(n)]
        top = rng.sample(range(n), 5)
        rels = [pool[i] for i in top]
        score = np.zeros(n)
        for rank, i in enumerate(top):
            score[i] = 10 - rank
        ours = E.pooled_ndcg5(rels, pool)
        if sum(pool) == 0:
            assert ours is None
            continue
        assert ours == pytest.approx(ndcg_score([pool], [score], k=5), abs=1e-9)
        gains = [2 ** p - 1 for p in pool]
        assert E.pooled_ndcg5(rels, pool, "exp") == pytest.approx(ndcg_score([gains], [score], k=5), abs=1e-9)


# ============================================================ bootstrap
def test_bootstrap_hand_cases():
    r = E.stratified_bootstrap({"G": [0.0, 1.0]}, B=20000)                  # means in {0,.5,1} with p=.25/.5/.25
    assert r["mean"] == 0.5 and r["ci_low"] == 0.0 and r["ci_high"] == 1.0 and r["n"] == 2
    r = E.stratified_bootstrap({"G": [0.3, 0.3, 0.3], "H": [0.3]}, B=500)
    assert r["ci_low"] == pytest.approx(0.3) and r["ci_high"] == pytest.approx(0.3)
    assert E.stratified_bootstrap({}, B=10) is None and E.stratified_bootstrap({"G": []}, B=10) is None


def test_bootstrap_is_deterministic_and_matches_frozen_protocol_lib():
    d = {"G1": [0.2, 0.0, 0.4, -0.1], "G2": [0.1, -0.1, 0.0], "G3": [0.5]}
    a = E.stratified_bootstrap(d, B=3000)
    assert a == E.stratified_bootstrap(d, B=3000)
    point, lo, hi = PL.paired_bootstrap(d, b=3000, seed=42)                  # independent earlier implementation
    assert (a["mean"], a["ci_low"], a["ci_high"]) == (pytest.approx(point), pytest.approx(lo), pytest.approx(hi))


def test_bootstrap_ci_brackets_mean_and_has_sane_width():
    rng = np.random.default_rng(1)
    x = rng.normal(0.2, 0.5, 400).tolist()
    r = E.unstratified_bootstrap(x, B=4000)
    se = np.std(x, ddof=1) / math.sqrt(len(x))
    assert r["ci_low"] < r["mean"] < r["ci_high"]
    assert (r["ci_high"] - r["ci_low"]) == pytest.approx(2 * 1.96 * se, rel=0.12)


def test_stratified_equals_unstratified_for_single_stratum_and_excludes_zero_rule():
    v = [0.1, 0.4, 0.2, 0.3, 0.5, 0.2]
    s = E.stratified_bootstrap({"G": v}, B=1000)
    u = E.unstratified_bootstrap(v, B=1000)
    assert s == u
    assert E.excludes_zero({"ci_low": 0.01, "ci_high": 0.5}) and E.excludes_zero({"ci_low": -0.5, "ci_high": -0.01})
    assert not E.excludes_zero({"ci_low": 0.0, "ci_high": 0.5}) and not E.excludes_zero({"ci_low": -0.1, "ci_high": 0.1}) and not E.excludes_zero(None)


def test_default_B_and_seed_follow_protocol():
    assert E.B_DEFAULT == 10000 and E.SEED == 42
    ev = json.loads((W7 / "config/evaluation.json").read_text())
    assert "B = 10000" in ev["inference"]["bootstrap"] and "default_rng(42)" in ev["inference"]["bootstrap"]


# ============================================================ kappa
def test_kappa_hand_value_and_matrix():
    pairs = [("q", 0, 0), ("q", 1, 2), ("q", 2, 2), ("q", 2, 1)]
    r = E.agreement_report(pairs, B=500)
    assert r["kappa_linear"]["kappa"] == pytest.approx(3 / 7)           # 1 - 0.25/0.4375 (hand computed)
    assert r["raw_agreement"] == 0.5 and r["n_pairs"] == 4
    assert r["confusion_rows_a1_cols_a2"] == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]
    assert r["low_reliability"] == (3 / 7 < 0.40)


def test_kappa_perfect_and_degenerate_cases():
    assert E.kappa_from_matrix(np.diag([5, 3, 2])) == pytest.approx(1.0)
    assert E.kappa_from_matrix(np.diag([10, 0, 0])) is None             # A7: undefined, never 0 or 1
    assert E.kappa_from_matrix(np.zeros((3, 3))) is None
    r = E.agreement_report([("q", 1, 1)] * 6 + [("r", 1, 1)] * 4, B=200)
    assert r["kappa_linear"]["kappa"] is None and r["kappa_linear"]["ci_low"] is None and r["low_reliability"] is None
    assert E.agreement_report([], B=10)["n_pairs"] == 0


def test_kappa_matches_sklearn_linear_and_quadratic():
    rng = random.Random(3)
    for _ in range(40):
        n = rng.randint(30, 120)
        a = [rng.choice([0, 1, 2]) for _ in range(n)]
        b = [x if rng.random() < 0.6 else rng.choice([0, 1, 2]) for x in a]
        M = np.zeros((3, 3))
        for x, y in zip(a, b):
            M[x, y] += 1
        for w in ("linear", "quadratic"):
            ref = cohen_kappa_score(a, b, weights=w, labels=[0, 1, 2])
            assert E.kappa_from_matrix(M, w) == pytest.approx(ref, abs=1e-9)


def test_kappa_cluster_bootstrap_is_deterministic_and_brackets_point():
    rng = random.Random(5)
    pairs = []
    for q in range(25):
        for _ in range(8):
            a = rng.choice([0, 1, 2]); b = a if rng.random() < 0.7 else rng.choice([0, 1, 2]); pairs.append((f"q{q}", a, b))
    r1, r2 = E.agreement_report(pairs, B=1500), E.agreement_report(pairs, B=1500)
    assert r1 == r2
    k = r1["kappa_linear"]
    assert k["ci_low"] < k["kappa"] < k["ci_high"] and k["replicates_dropped"] == 0


# ============================================================ query table validation
def test_build_query_table_validations(fx):
    key, queries, _, _ = fx
    qt = E.build_query_table(key, queries)
    assert set(qt) == set(queries) and all(len(qt[q]["top5"][m]) == 5 for q in qt for m in E.METHODS)
    assert qt["qB"]["top5"]["visual"] == qt["qB"]["top5"]["metadata"]            # same candidates for both methods
    with pytest.raises(ValueError, match="pilot"):
        E.build_query_table([dict(key[0], set="pilot")], queries)
    bad = [dict(r) for r in key]
    for r in bad:
        if r["pair_id"] == "qA-a2":
            r["visual_rank"] = "1"                                               # duplicate rank
    with pytest.raises(ValueError, match="duplicate rank"):
        E.build_query_table(bad, queries)
    with pytest.raises(ValueError, match="unknown query"):
        E.build_query_table(key + [dict(key[0], query_id="qZ", pair_id="qZ-1", candidate_id="zz")], queries)
    with pytest.raises(ValueError, match="duplicate pair"):
        E.build_query_table(key + [dict(key[0])], queries)
    anchor_clash = [dict(r) for r in key]
    for r in anchor_clash:
        if r["pair_id"] == "qA-x1":
            r.update(in_visual_top5="True", visual_rank="1")
    with pytest.raises(ValueError):
        E.build_query_table(anchor_clash, queries)
    with pytest.raises(ValueError, match="without key rows"):
        E.build_query_table([r for r in key if r["query_id"] != "qD"], queries)


# ============================================================ end-to-end on the hand fixture
def test_fixture_label_aggregation_counts(result):
    res, _, final = result
    assert res["label_aggregation"]["pairs"] == 40
    assert res["label_aggregation"]["status_counts"] == {"agree": 35, "min_rule": 1, "unresolved": 1, "single_label": 1, "missing": 2}
    assert final["qB-b2"] == 1 and final["qB-b3"] is None and final["qA-a5"] == 1 and final["qC-c2"] is None
    assert res["label_aggregation"]["adjudicator_used"] is False


def test_fixture_exclusions_and_retained_sets(result):
    res, _, _ = result
    p = res["primary"]
    assert p["retained"] == {"P5": 3, "MeanRel5": 3, "Hit5": 2, "NDCG5": 1, "RQ1": 3}
    assert p["exclusions"] == {"dropped_more_than_1_missing_in_a_Top5": ["qC"],
                               "Hit5_NDCG5_excluded_missing_position": ["qB"],
                               "NDCG5_excluded_IDCG_zero": ["qD"],
                               "RQ1_excluded_no_judged_anchor": []}


def test_fixture_primary_values_hand_computed(result):
    m = result[0]["primary"]["metrics"]
    # P@5: qA v .8 / m .6 ; qB .75/.75 (identical lists, 1 unresolved) ; qD 0/0  -> diffs .2, 0, 0
    assert m["P5"]["visual"]["mean"] == pytest.approx((0.8 + 0.75 + 0) / 3)
    assert m["P5"]["metadata"]["mean"] == pytest.approx((0.6 + 0.75 + 0) / 3)
    assert m["P5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(0.2 / 3)
    # MeanRel@5: qA v 1.2 / m 1.0 ; qB 1.0/1.0 ; qD 0/0
    assert m["MeanRel5"]["visual"]["mean"] == pytest.approx((1.2 + 1.0 + 0) / 3)
    assert m["MeanRel5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(0.2 / 3)
    # Hit@5 over qA, qD only (qB excluded: missing position)
    assert m["Hit5"]["n_queries"] == 2 and m["Hit5"]["visual"]["mean"] == 0.5 and m["Hit5"]["diff_visual_minus_metadata"]["mean"] == 0.0
    # pooled NDCG@5 only qA (qD: IDCG 0, qB: missing)
    assert m["NDCG5"]["n_queries"] == 1
    assert m["NDCG5"]["visual"]["mean"] == pytest.approx(0.7637012590219477)
    assert m["NDCG5"]["metadata"]["mean"] == pytest.approx(0.6702844482938307)
    assert m["NDCG5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(0.0934168107281170)


def test_fixture_secondary_aggregations_hand_computed(result):
    mr = result[0]["primary"]["metrics"]["MeanRel5"]
    # strata: G1 = {qA diff .2, qB 0} -> .1 ; G2 = {qD 0} -> 0 ; balanced = .05
    assert mr["balanced_by_stratum"]["diff"] == pytest.approx(0.05)
    assert mr["balanced_by_stratum"]["visual"] == pytest.approx(((1.2 + 1.0) / 2 + 0.0) / 2)
    # catalogue weights: qA,qB = 10/2 = 5 ; qD = 6/2 = 3  -> (5*.2)/13
    assert mr["catalogue_weighted"]["diff"] == pytest.approx(1.0 / 13)
    assert mr["per_stratum_descriptive"]["G1"] == {"n": 2, "visual": pytest.approx(1.1), "metadata": pytest.approx(1.0)}
    assert mr["per_stratum_descriptive"]["G2"]["n"] == 1
    assert mr["diff_unstratified_sensitivity"]["mean"] == pytest.approx(0.2 / 3)


def test_fixture_rq1_hand_computed(result):
    rq1 = result[0]["primary"]["RQ1"]
    # anchors: qA {0,1}->.5 ; qB {1,0}->.5 ; qD {0,0}->0 ; visual P5 .8/.75/0 -> deltas .3,.25,0
    assert rq1["n_queries"] == 3 and rq1["anchor_rate_mean"] == pytest.approx(1 / 3)
    assert rq1["visual_P5_mean"] == pytest.approx(1.55 / 3) and rq1["delta"]["mean"] == pytest.approx(0.55 / 3)
    assert rq1["verdict_visual_above_chance"] == (rq1["delta"]["ci_low"] > 0)


def test_fixture_identical_candidates_give_zero_difference(fx):
    key, queries, la, lb = fx
    qt = E.build_query_table(key, queries)
    final, _ = E.aggregate_all([p for t in qt.values() for p in t["pool"]], la, lb)
    vals = E.per_query_values(qt, final)
    for k in ("P5", "MeanRel5"):
        assert vals["qB"]["visual"][k] == vals["qB"]["metadata"][k]
    assert qt["qB"]["top5"]["visual"] == qt["qB"]["top5"]["metadata"]
    assert len(qt["qB"]["pool"]) == 7           # 5 shared + 2 anchors: shared candidates are judged ONCE
    assert len(qt["qA"]["pool"]) == 11          # 5 + 4 (a1 shared) + 2


def test_fixture_sensitivity_analyses_exist_and_follow_rules(result):
    s = result[0]["sensitivity"]
    assert set(s) == {"annotator_A_only", "annotator_B_only", "mean_label"}
    assert set(s["mean_label"]["metrics"]) == {"MeanRel5", "NDCG5"}
    ml = s["mean_label"]["metrics"]["MeanRel5"]
    # mean label: qB b2 -> 1.5, b3 -> 1.0 (0 and 2 average) ; qB visual rels [2,1.5,1,1,0] -> 5.5/5
    assert ml["n_queries"] == 3 and ml["visual"]["n"] == 3
    qB = (2 + 1.5 + 1.0 + 1 + 0) / 5
    qA = 1.2
    assert ml["visual"]["mean"] == pytest.approx((qA + qB + 0) / 3)
    # annotator A only: a5 label 1 stays, annotators disagree on b2 (A=2) ; b3 (A=0)
    a_only = s["annotator_A_only"]["metrics"]["MeanRel5"]
    assert a_only["n_queries"] == 3


def test_fixture_agreement_before_adjudication(result, fx):
    ag = result[0]["agreement"]
    _, _, la, lb = fx
    both = [(p, la[p], lb[p]) for p in la if la[p] is not None and lb[p] is not None]
    a, b = [x[1] for x in both], [x[2] for x in both]
    assert ag["all_pool_pairs"]["n_pairs"] == len(both) == 37                    # 40 pairs - c2, c3 (nobody) - a5 (annotator B has U)
    assert ag["all_pool_pairs"]["kappa_linear"]["kappa"] == pytest.approx(cohen_kappa_score(a, b, weights="linear", labels=[0, 1, 2]))
    assert ag["all_pool_pairs"]["kappa_quadratic"]["kappa"] == pytest.approx(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2]))
    assert ag["all_pool_pairs"]["raw_agreement"] == pytest.approx(sum(x == y for x, y in zip(a, b)) / len(a))
    assert ag["anchor_pairs"]["n_pairs"] == 8 and set(ag) == {"all_pool_pairs", "visual_top5_pairs", "metadata_top5_pairs", "anchor_pairs"}
    assert np.array(ag["all_pool_pairs"]["confusion_rows_a1_cols_a2"]).sum() == 37


def test_fixture_verdict_strings_are_mechanical(result):
    v = result[0]["verdicts"]
    d = result[0]["primary"]["metrics"]["MeanRel5"]["diff_visual_minus_metadata"]
    assert v["RQ2_primary"].startswith("difference detected") == E.excludes_zero(d)
    assert E.excludes_zero(d) or "NOT evidence of equivalence" in v["RQ2_primary"]
    assert "ambiguities_applied" in result[0] and set(result[0]["ambiguities_applied"]) >= {f"A{i}" for i in range(1, 14)}


def test_clear_difference_is_detected_and_equal_methods_are_not():
    """Synthetic 24-query set: visual clearly better -> CI excludes 0; identical methods -> CI is [0,0] and not 'detected'."""
    key, queries, la, lb = [], {}, {}, {}
    rng = random.Random(11)
    for i in range(24):
        q = f"q{i}"
        queries[q] = {"group": "G1" if i < 12 else "G2", "stratum_size": 12, "stratum_queries": 12}
        vis = [f"v{k}" for k in range(1, 6)]; meta = [f"m{k}" for k in range(1, 6)]; anc = ["x1", "x2"]
        for c in vis + meta + anc:
            pid = f"{q}-{c}"
            key.append({"pair_id": pid, "set": "main", "query_id": q, "candidate_id": c, "in_visual_top5": str(c in vis),
                        "visual_rank": vis.index(c) + 1 if c in vis else "", "in_metadata_top5": str(c in meta),
                        "metadata_rank": meta.index(c) + 1 if c in meta else "", "is_anchor": str(c in anc)})
            lab = rng.choice([1, 2]) if c in vis else (0 if c in meta else rng.choice([0, 1]))
            la[pid] = lb[pid] = lab
    res, _, _ = E.run_evaluation(key, queries, la, lb, None, B=1500)
    d = res["primary"]["metrics"]["MeanRel5"]["diff_visual_minus_metadata"]
    assert d["ci_low"] > 0 and "difference detected: visual higher" in res["verdicts"]["RQ2_primary"]
    assert res["primary"]["RQ1"]["verdict_visual_above_chance"] is True
    for r in key:                                    # make metadata = visual (same candidates, same ranks)
        r["in_metadata_top5"], r["metadata_rank"] = r["in_visual_top5"], r["visual_rank"]
    key2 = [r for r in key if not r["candidate_id"].startswith("m")]
    res2, _, _ = E.run_evaluation(key2, queries, la, lb, None, B=1500)
    d2 = res2["primary"]["metrics"]["MeanRel5"]["diff_visual_minus_metadata"]
    assert d2["mean"] == 0 and d2["ci_low"] == 0 == d2["ci_high"] and "no difference detected" in res2["verdicts"]["RQ2_primary"]


def test_full_default_bootstrap_size_runs_once(fx):
    key, queries, la, lb = fx
    res, _, _ = E.run_evaluation(key, queries, la, lb, None)                    # B = 10000
    assert res["bootstrap"] == {"B": 10000, "seed": 42}


def test_adjudicator_changes_only_disagreements(fx):
    key, queries, la, lb = fx
    adj = {"qB-b2": 2, "qB-b3": 0}               # the two disagreement pairs of the fixture
    res, _, final = E.run_evaluation(key, queries, la, lb, adj, B=B)
    assert final["qB-b2"] == 2 and final["qB-b3"] == 0 and final["qA-a1"] == 2
    assert res["label_aggregation"]["status_counts"].get("adjudicated") == 2 and "unresolved" not in res["label_aggregation"]["status_counts"]
    assert res["primary"]["retained"]["Hit5"] == 3                              # qB now has no missing position


def test_exact_protocol_missing_rules_are_not_changed():
    ev = json.loads((W7 / "config/evaluation.json").read_text())["missing_labels"]
    assert ev["never_treated_as_zero"] is True
    assert "more than 1 missing" in ev["rule_query_drop"] and "any Top-5 position" in ev["rule_NDCG_and_Hit"]


# ============================================================ lock / freeze / CLI flow on a synthetic package
@pytest.fixture()
def pkg(tmp_path, fx):
    key, queries, la, lb = fx
    ev, priv, hm = tmp_path / "evaluation", tmp_path / "private", tmp_path / "hm"
    for d in (ev, priv, hm):
        d.mkdir()
    pids = sorted(r["pair_id"] for r in key)
    with open(ev / "annotation_tasks.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair_id", "set", "query_image", "candidate_image", "order_annotator_01", "order_annotator_02", "order_annotator_03"])
        for i, p in enumerate(pids):
            w.writerow([p, "main", "i", "c", i, i, i])
    (ev / "annotation_manifest.json").write_text(json.dumps({"hashes": {"annotation_tasks.csv": E.sha_file(ev / "annotation_tasks.csv")}}))
    with open(priv / "annotation_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    (priv / "pool_manifest.json").write_text(json.dumps({"annotation_key_sha256": E.sha_file(priv / "annotation_key.csv")}))
    with open(hm / "query_manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["article_id", "product_group_name", "stratum_size", "stratum_queries"])
        for q, v in queries.items():
            w.writerow([q, v["group"], v["stratum_size"], v["stratum_queries"]])

    def labfile(name, annot, labels):
        p = tmp_path / name
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc"])
            for pid in pids:
                v = labels[pid]
                w.writerow([pid, "main", annot, "U" if v is None else v, "", "", ""])
        return p
    return {"ev": ev, "priv": priv, "hm": hm, "out": tmp_path / "results", "a1": labfile("annotator_01.csv", "annotator_01", la),
            "a2": labfile("annotator_02.csv", "annotator_02", lb), "labfile": labfile, "tmp": tmp_path}


def _lock(pkg, **kw):
    ns = type("A", (), {"a1": pkg["a1"], "a2": pkg["a2"], "adjudicator": kw.get("adj")})()
    E.cmd_lock(ns, pkg["ev"], pkg["out"])


def _eval(pkg, frozen=None):
    ns = type("A", (), {"B": 800})()
    E.cmd_evaluate(ns, pkg["ev"], pkg["priv"], pkg["hm"], pkg["out"], frozen)


def test_evaluate_refuses_without_lock(pkg):
    with pytest.raises(SystemExit, match="lock"):
        _eval(pkg)


def test_lock_does_not_need_or_read_the_key(pkg):
    (pkg["priv"] / "annotation_key.csv").unlink()           # key absent: lock must still work
    _lock(pkg)
    lock = json.loads((pkg["out"] / "labels_lock.json").read_text())
    assert lock["private_key_opened"] is False and set(lock["files"]) == {"annotator_1", "annotator_2"}
    assert all(len(v["sha256"]) == 64 for v in lock["files"].values())


def test_full_cli_flow_matches_in_memory_run(pkg, fx, capsys):
    _lock(pkg)
    _eval(pkg)
    res = json.loads((pkg["out"] / "results.json").read_text())
    key, queries, la, lb = fx
    assert res["label_aggregation"]["pairs"] == 40 and res["evaluator_frozen_verified"] is False
    assert res["primary"]["retained"] == {"P5": 3, "MeanRel5": 3, "Hit5": 2, "NDCG5": 1, "RQ1": 3}
    assert res["primary"]["metrics"]["MeanRel5"]["diff_visual_minus_metadata"]["mean"] == pytest.approx(0.2 / 3)
    assert res["marker_counts"]["annotator_1"] == {"ok": 38, "U": 2} and res["marker_counts"]["annotator_2"] == {"ok": 37, "U": 3}
    rows = list(csv.DictReader(open(pkg["out"] / "per_query_metrics.csv")))
    assert len(rows) == 8 and {r["method"] for r in rows} == {"visual", "metadata"}
    qc = [r for r in rows if r["query_id"] == "qC" and r["method"] == "visual"][0]
    assert qc["n_missing_top5"] == "2"


def test_label_change_after_lock_blocks_evaluation(pkg):
    _lock(pkg)
    txt = pkg["a1"].read_text()
    pkg["a1"].write_text(txt.replace(",1,,,", ",2,,,", 1))
    with pytest.raises(SystemExit, match="changed after lock"):
        _eval(pkg)


def test_key_tampering_blocks_evaluation(pkg):
    _lock(pkg)
    (pkg["priv"] / "annotation_key.csv").write_text((pkg["priv"] / "annotation_key.csv").read_text() + "\n")
    with pytest.raises(SystemExit, match="pool_manifest"):
        _eval(pkg)


def test_lock_rejects_invalid_incomplete_or_same_annotator(pkg, fx):
    _, _, la, lb = fx
    bad = pkg["a1"].read_text().replace(",U,", ",9,", 1)
    pkg["a1"].write_text(bad)
    with pytest.raises(SystemExit, match="no lock"):
        _lock(pkg)
    same = pkg["labfile"]("annotator_01_copy.csv", "annotator_01", lb)
    pkg["a1"] = pkg["labfile"]("annotator_01.csv", "annotator_01", la)
    pkg["a2"] = same
    with pytest.raises(SystemExit):
        _lock(pkg)


def test_lock_with_adjudicator_file_is_used(pkg, fx):
    _, _, la, lb = fx
    adj = {p: (la[p] if la[p] is not None else lb[p]) for p in la}
    adj["qB-b2"], adj["qB-b3"] = 2, 0
    pkg["adj"] = pkg["labfile"]("annotator_03.csv", "annotator_03", adj)
    _lock(pkg, adj=pkg["adj"])
    _eval(pkg)
    res = json.loads((pkg["out"] / "results.json").read_text())
    assert res["label_aggregation"]["adjudicator_used"] is True and res["label_aggregation"]["status_counts"].get("adjudicated") == 2


def test_verify_frozen_detects_change(tmp_path):
    f = tmp_path / "x.py"; f.write_text("a")
    man = tmp_path / "m.json"; man.write_text(json.dumps({"files": {"x.py": {"sha256": E.sha_file(f)}}}))
    E.verify_frozen(man, tmp_path)
    f.write_text("b")
    with pytest.raises(RuntimeError, match="changed after freeze"):
        E.verify_frozen(man, tmp_path)


# ============================================================ repository state guards
def test_no_real_labels_or_results_exist_and_module_has_no_import_side_effects():
    assert list(W7.rglob("annotator_0*.csv")) == []
    assert not (W7 / "outputs/evaluation_results").exists()
    src = (W7 / "src/evaluate_recommendations.py").read_text()
    assert "annotation_key" not in src.split("def cmd_lock")[1].split("def cmd_evaluate")[0]   # lock never touches the key
    assert src.index("labels_lock.json") < src.index("annotation_key.csv", src.index("def cmd_evaluate"))   # key opened only after lock check


def test_evaluator_does_not_modify_frozen_protocol_files():
    man = json.loads((W7 / "outputs/hm/protocol_manifest.json").read_text())
    for f, v in man["files"].items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == v["sha256"], f
