"""Tests for the subset annotation workflow (decisions N1-N4): subset interfaces, second judgments, adjudication, merge, gate.

Everything here uses SYNTHETIC fixtures inside pytest tmp dirs. No fixture is a research label; nothing is written to the official
output folders; the private key and the primary label files of the real study are never opened.
Run: python3 -I -B -m pytest -q week7/tests/test_subset_workflow.py -p no:cacheprovider
"""
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
sys.path.insert(0, str(W7 / "src"))
sys.path.insert(0, str(W7 / "tests"))
import evaluate_recommendations as E1  # noqa: E402
import evaluate_recommendations_v2 as E2  # noqa: E402
import prepare_subset_tasks as PS  # noqa: E402
import reporting_gate as RG  # noqa: E402
import subset_labels as SL  # noqa: E402
from test_evaluation_v2 import build_fx2  # noqa: E402  (synthetic hand fixture of the evaluator tests)

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
LEAK = re.compile(r"visual|metadata|cosine|resnet|anchor|top-?5|rank|score|method|baseline|\bcnn\b|embedding|tf-?idf|evaluator|amendment", re.I)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ns(**kw):
    return type("A", (), kw)()


# ======================================================================= synthetic package + plan
@pytest.fixture()
def sub(tmp_path):
    """Synthetic annotation package with tiny images; two single-numeric pairs and two disagreements."""
    key, queries, la, lb = build_fx2()
    la, lb = dict(la), dict(lb)
    la["qA-a5"], lb["qA-a5"] = None, 2          # single numeric (annotator 2)         -> second judgment
    la["qB-b2"], lb["qB-b2"] = 2, 0             # disagreement by 2                       -> adjudication
    la["qB-b3"], lb["qB-b3"] = 0, 1             # disagreement by 1                       -> adjudication
    ev, priv, hm, out = tmp_path / "evaluation", tmp_path / "private", tmp_path / "hm", tmp_path / "results"
    for d in (ev, priv, hm, ev / "images"):
        d.mkdir(parents=True)
    pids = sorted(r["pair_id"] for r in key)
    byp = {r["pair_id"]: r for r in key}
    names = {}

    def img(article):
        n = f"images/img_{hashlib.sha256(article.encode()).hexdigest()[:12]}.jpg"
        if article not in names:
            Image.new("RGB", (40, 60), (hash(article) % 200 + 30, 120, 90)).save(ev / n)
            names[article] = n
        return names[article]
    with open(ev / "annotation_tasks.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair_id", "set", "query_image", "candidate_image", "order_annotator_01", "order_annotator_02", "order_annotator_03"])
        for i, p in enumerate(pids):
            w.writerow([p, "main", img(byp[p]["query_id"]), img(byp[p]["candidate_id"]), i, i, i])
    (ev / "annotation_guide.md").write_text((W7 / "outputs/evaluation/annotation_guide.md").read_text(encoding="utf-8"), encoding="utf-8")
    (ev / "annotation_manifest.json").write_text(json.dumps({"hashes": {"annotation_tasks.csv": sha(ev / "annotation_tasks.csv")}}))
    with open(priv / "annotation_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    (priv / "pool_manifest.json").write_text(json.dumps({"annotation_key_sha256": sha(priv / "annotation_key.csv")}))
    with open(hm / "query_manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["article_id", "product_group_name", "stratum_size", "stratum_queries"])
        for q, v in queries.items():
            w.writerow([q, v["group"], v["stratum_size"], v["stratum_queries"]])

    def labfile(name, annot, labels, only=None):
        p = tmp_path / name
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc"])
            for pid in pids:
                if only is not None and pid not in only:
                    continue
                v = labels.get(pid)
                w.writerow([pid, "main", annot, "U" if v is None else v, "", "", ""])
        return p
    a1, a2 = labfile("annotator_01.csv", "annotator_01", la), labfile("annotator_02.csv", "annotator_02", lb)
    return {"ev": ev, "priv": priv, "hm": hm, "out": out, "a1": a1, "a2": a2, "tmp": tmp_path, "la": la, "lb": lb, "labfile": labfile,
            "queries": queries, "key": key}


@pytest.fixture()
def planned(sub):
    E2.cmd_lock(ns(a1=sub["a1"], a2=sub["a2"], third=None), sub["ev"], sub["out"])
    E2.cmd_plan(sub["out"])
    sub["pkg"] = sub["tmp"] / "subset_package"
    PS.build(sub["out"], sub["ev"], sub["pkg"])
    return sub


def cfg_of(html: str) -> dict:
    return json.loads(re.search(r'<script id="cfg-json" type="application/json">(.*?)</script>', html, re.S).group(1))


# ======================================================================= plan -> subset interfaces
def test_plan_lists_exactly_the_expected_pairs(planned):
    p = SL.read_plan(planned["out"])
    assert p["second_judgment"] == {"qC-d2", "qA-a5"}
    assert p["adjudication"] == {"qB-b2": ["0", "2"], "qB-b3": ["0", "1"]}


def test_generator_writes_two_interfaces_images_and_manifest(planned):
    pkg = planned["pkg"]
    assert sorted(x.name for x in pkg.iterdir()) == ["adjudication", "second_judgment", "subset_manifest.json"]
    m = json.loads((pkg / "subset_manifest.json").read_text())
    assert m["counts"] == {"second_judgment": 2, "adjudication": 2} and m["private_key_opened"] is False
    assert all(sha(pkg / n) == h for n, h in m["files"].items())
    for mode in ("second_judgment", "adjudication"):                       # self-contained role folders, delivered separately
        d = pkg / mode
        assert {x.name for x in d.iterdir()} == {"annotation_guide.md", f"{mode}_interface.html", "images"}
        cfg = cfg_of((d / f"{mode}_interface.html").read_text(encoding="utf-8"))
        assert cfg["mode"] == mode
        for t in cfg["tasks"]:
            assert (d / t["q"]).is_file() and (d / t["c"]).is_file()
        assert {x.name for x in (d / "images").iterdir()} == {Path(t[k]).name for t in cfg["tasks"] for k in ("q", "c")}


def test_second_judgment_interface_carries_no_label_and_no_method(planned):
    html = (planned["pkg"] / "second_judgment/second_judgment_interface.html").read_text(encoding="utf-8")
    cfg = cfg_of(html)
    assert {p["p"] for p in cfg["tasks"]} == {"qC-d2", "qA-a5"}
    assert all(sorted(t) == ["c", "o", "p", "q"] for t in cfg["tasks"])            # pair_id, images, order only
    assert not LEAK.findall(html) and "opts" not in cfg_of(html)["tasks"][0]
    # the label values of the primary annotators (1 for qC-d2, 2 for qA-a5) are not attached to any pair in the page
    assert "label_annotator" not in html and "annotator_1" not in html


def test_adjudication_interface_shows_two_unattributed_ascending_labels(planned):
    html = (planned["pkg"] / "adjudication/adjudication_interface.html").read_text(encoding="utf-8")
    cfg = cfg_of(html)
    assert {t["p"]: t["opts"] for t in cfg["tasks"]} == {"qB-b2": [0, 2], "qB-b3": [0, 1]}
    assert not LEAK.findall(html) and "label_annotator" not in html and "annotator_1" not in html and "annotator_2" not in html.replace("annotator_02", "")


def test_generator_never_touches_primary_files_or_the_key(planned):
    src = (W7 / "src/prepare_subset_tasks.py").read_text()
    for forbidden in ("annotation_key", "pool_manifest", "evaluation_private", "annotator_01", "annotator_02"):
        assert forbidden not in src.replace("`annotator_01/02`", ""), forbidden
    before = (sha(planned["a1"]), sha(planned["a2"]), sha(planned["priv"] / "annotation_key.csv"))
    PS.build(planned["out"], planned["ev"], planned["tmp"] / "again")
    assert (sha(planned["a1"]), sha(planned["a2"]), sha(planned["priv"] / "annotation_key.csv")) == before


def test_generator_is_deterministic(planned):
    PS.build(planned["out"], planned["ev"], planned["tmp"] / "again")
    for mode in ("second_judgment", "adjudication"):
        n = f"{mode}/{mode}_interface.html"
        assert (planned["tmp"] / "again" / n).read_text() == (planned["pkg"] / n).read_text()


def test_generator_refuses_without_lock_plan_or_with_bad_plan(sub, tmp_path):
    with pytest.raises(SystemExit, match="lock"):
        PS.build(sub["out"], sub["ev"], tmp_path / "x")
    E2.cmd_lock(ns(a1=sub["a1"], a2=sub["a2"], third=None), sub["ev"], sub["out"])
    with pytest.raises(SystemExit, match="plan"):
        PS.build(sub["out"], sub["ev"], tmp_path / "x")
    E2.cmd_plan(sub["out"])
    (sub["out"] / "second_judgment_needed.csv").write_text("pair_id,reason\nPnotreal,single_numeric\nqC-d2,single_numeric\nqA-a5,single_numeric\n")
    with pytest.raises(SystemExit, match="plan_summary"):
        PS.build(sub["out"], sub["ev"], tmp_path / "x")
    (sub["out"] / "plan_summary.json").write_text(json.dumps({"second_judgment_needed": 3, "adjudication_needed": 2}))
    with pytest.raises(SystemExit, match="not in annotation_tasks"):
        PS.build(sub["out"], sub["ev"], tmp_path / "x")


def test_generator_refuses_pair_in_both_roles_and_equal_labels(sub, tmp_path):
    E2.cmd_lock(ns(a1=sub["a1"], a2=sub["a2"], third=None), sub["ev"], sub["out"])
    E2.cmd_plan(sub["out"])
    (sub["out"] / "adjudication_needed.csv").write_text("pair_id,label_annotator_1,label_annotator_2\nqC-d2,0,2\nqB-b2,2,0\n")
    with pytest.raises(SystemExit, match="both"):
        PS.build(sub["out"], sub["ev"], tmp_path / "x")
    (sub["out"] / "adjudication_needed.csv").write_text("pair_id,label_annotator_1,label_annotator_2\nqB-b2,1,1\nqB-b3,0,1\n")
    with pytest.raises(SystemExit, match="different labels"):
        PS.build(sub["out"], sub["ev"], tmp_path / "x")


def test_empty_plan_produces_no_interface_for_that_role(sub, tmp_path):
    la, lb = dict(sub["la"]), dict(sub["lb"])
    for p in ("qC-d2", "qA-a5"):
        la[p] = lb[p] = None
    a1 = sub["labfile"]("annotator_01.csv", "annotator_01", la); a2 = sub["labfile"]("annotator_02.csv", "annotator_02", lb)
    E2.cmd_lock(ns(a1=a1, a2=a2, third=None), sub["ev"], sub["out"]); E2.cmd_plan(sub["out"])
    m = PS.build(sub["out"], sub["ev"], tmp_path / "pk")
    assert m["counts"]["second_judgment"] == 0 and not (tmp_path / "pk/second_judgment").exists() and (tmp_path / "pk/adjudication/adjudication_interface.html").exists()


# ======================================================================= validation / merge of third-annotator files
HEAD = SL.HEAD


def wcsv(path, rows, head=HEAD):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)
    return path


def sj_rows(annot="annotator_03", labels=None, role="second_judgment"):
    labels = labels or {"qC-d2": "1", "qA-a5": "2"}
    return [[p, "main", annot, l, "", "", "", role] for p, l in labels.items()]


def adj_rows(annot="annotator_03", labels=None, role="adjudication"):
    labels = labels or {"qB-b2": "2", "qB-b3": "0"}
    return [[p, "main", annot, l, "", "", "", role] for p, l in labels.items()]


def test_validate_accepts_good_files(planned):
    plan = SL.read_plan(planned["out"])
    r = SL.validate_file(wcsv(planned["tmp"] / "s.csv", sj_rows()), "second_judgment", plan)
    assert r["ok"] and r["complete"] and r["answered"] == 2
    r = SL.validate_file(wcsv(planned["tmp"] / "a.csv", adj_rows()), "adjudication", plan)
    assert r["ok"] and r["complete"]
    r = SL.validate_file(wcsv(planned["tmp"] / "s2.csv", sj_rows(labels={"qC-d2": "U", "qA-a5": ""})), "second_judgment", plan)
    assert r["ok"] and r["answered"] == 1 and not r["complete"]                       # U is an answer; blank is not


@pytest.mark.parametrize("name,rows,mode,expect", [
    ("unknown pair_id", sj_rows(labels={"Pdeadbeef00": "1"}), "second_judgment", "not in the plan"),
    ("pair of the other role", sj_rows(labels={"qB-b2": "1"}), "second_judgment", "belong to the 'adjudication' list"),
    ("adjudication pair in second file", adj_rows(labels={"qC-d2": "1"}), "adjudication", "belong to the 'second_judgment' list"),
    ("wrong role value", sj_rows(role="adjudication"), "second_judgment", "role !="),
    ("annotator_01 id", sj_rows(annot="annotator_01"), "second_judgment", "not allowed here"),
    ("annotator_02 id", sj_rows(annot="annotator_02"), "second_judgment", "not allowed here"),
    ("label 3", sj_rows(labels={"qC-d2": "3", "qA-a5": "1"}), "second_judgment", "outside"),
    ("label -1", sj_rows(labels={"qC-d2": "-1"}), "second_judgment", "outside"),
    ("adjudication outside the two labels", adj_rows(labels={"qB-b2": "1", "qB-b3": "0"}), "adjudication", "not one of the two existing labels"),
    ("adjudication outside (b3)", adj_rows(labels={"qB-b2": "2", "qB-b3": "2"}), "adjudication", "not one of the two existing labels"),
    ("duplicate pair", sj_rows() + sj_rows(labels={"qC-d2": "1"}), "second_judgment", "duplicate"),
])
def test_validate_rejects_bad_files(planned, name, rows, mode, expect):
    r = SL.validate_file(wcsv(planned["tmp"] / "bad.csv", rows), mode, SL.read_plan(planned["out"]))
    assert not r["ok"] and any(expect in e for e in r["errors"]), (name, r["errors"])


def test_validate_rejects_missing_columns_mixed_ids_and_bad_flags(planned):
    plan = SL.read_plan(planned["out"])
    r = SL.validate_file(wcsv(planned["tmp"] / "m.csv", [["qC-d2", "1"]], head=["pair_id", "label"]), "second_judgment", plan)
    assert not r["ok"] and "missing columns" in r["errors"][0]
    rows = sj_rows(); rows[1][2] = "annotator_04"
    assert any("several annotator_id" in e for e in SL.validate_file(wcsv(planned["tmp"] / "x.csv", rows), "second_judgment", plan)["errors"])
    rows = sj_rows(); rows[0][4] = "yes"
    assert any("uncertain" in e for e in SL.validate_file(wcsv(planned["tmp"] / "y.csv", rows), "second_judgment", plan)["errors"])
    rows = sj_rows(); rows[0][1] = "pilot"
    assert any("set 'main'" in e for e in SL.validate_file(wcsv(planned["tmp"] / "z.csv", rows), "second_judgment", plan)["errors"])


def test_merge_produces_one_file_and_never_changes_inputs(planned):
    s = wcsv(planned["tmp"] / "annotator_03_second_judgment.csv", sj_rows())
    a = wcsv(planned["tmp"] / "annotator_03_adjudication.csv", adj_rows())
    before = (sha(s), sha(a), sha(planned["a1"]), sha(planned["a2"]))
    out = planned["tmp"] / "annotator_03.csv"
    rep = SL.merge(planned["out"], s, a, out)
    assert rep["rows"] == 4 and rep["annotator_id"] == "annotator_03" and rep["second_judgment"] == {"needed": 2, "answered": 2} and rep["adjudication"]["answered"] == 2
    rows = list(csv.DictReader(open(out, newline="")))
    assert [r["pair_id"] for r in rows] == sorted(r["pair_id"] for r in rows)
    assert {r["pair_id"]: r["role"] for r in rows} == {"qC-d2": "second_judgment", "qA-a5": "second_judgment", "qB-b2": "adjudication", "qB-b3": "adjudication"}
    assert (sha(s), sha(a), sha(planned["a1"]), sha(planned["a2"])) == before


def test_merge_allows_one_role_only_and_refuses_inconsistent_inputs(planned):
    out = planned["tmp"] / "o.csv"
    assert SL.merge(planned["out"], wcsv(planned["tmp"] / "s.csv", sj_rows()), None, out)["rows"] == 2
    with pytest.raises(SystemExit, match="different annotators"):
        SL.merge(planned["out"], wcsv(planned["tmp"] / "s.csv", sj_rows(annot="annotator_03")), wcsv(planned["tmp"] / "a.csv", adj_rows(annot="annotator_04")), out)
    with pytest.raises(SystemExit, match="merge refused"):
        SL.merge(planned["out"], wcsv(planned["tmp"] / "s.csv", sj_rows(labels={"qB-b2": "1"})), None, out)
    with pytest.raises(SystemExit, match="nothing to merge"):
        SL.merge(planned["out"], None, None, out)


def test_cli_exit_codes(planned):
    cmd = [sys.executable, "-I", "-B", str(W7 / "src/subset_labels.py"), "validate", "--mode", "second_judgment", "--plan-dir", str(planned["out"])]
    good = wcsv(planned["tmp"] / "g.csv", sj_rows()); bad = wcsv(planned["tmp"] / "b.csv", sj_rows(labels={"qC-d2": "5"}))
    assert subprocess.run(cmd + [str(good)], capture_output=True).returncode == 0
    assert subprocess.run(cmd + [str(bad)], capture_output=True).returncode == 1


# ======================================================================= compatibility with lock / plan / evaluate (v2, frozen)
def test_merged_file_works_with_lock_third_and_evaluate(planned):
    E1.verify_frozen(W7 / "outputs/methodology/evaluator_v2_manifest.json")            # the evaluator itself is the frozen one
    s = wcsv(planned["tmp"] / "s.csv", sj_rows(labels={"qC-d2": "1", "qA-a5": "2"}))
    a = wcsv(planned["tmp"] / "a.csv", adj_rows(labels={"qB-b2": "2", "qB-b3": "0"}))
    third = planned["tmp"] / "annotator_03.csv"
    SL.merge(planned["out"], s, a, third)
    E2.cmd_lock(ns(a1=planned["a1"], a2=planned["a2"], third=third), planned["ev"], planned["out"])
    E2.cmd_evaluate(ns(B=200), planned["ev"], planned["priv"], planned["hm"], planned["out"], None)
    res = json.loads((planned["out"] / "results.json").read_text())
    sc = res["label_aggregation"]["status_counts"]
    assert sc["second_judgment_agree"] == 2 and sc["adjudicated"] == 2 and res["label_aggregation"]["third_file_used"] is True
    assert "single_numeric_not_ground_truth" not in sc and res["label_aggregation"]["pairs_single_numeric_awaiting_second_judgment"] == 0
    lock = json.loads((planned["out"] / "labels_lock.json").read_text())
    assert set(lock["files"]) == {"annotator_1", "annotator_2", "annotator_3"} and lock["private_key_opened"] is False


def test_second_judgment_never_overrides_and_adjudication_resolves(planned):
    s = wcsv(planned["tmp"] / "s.csv", sj_rows(labels={"qC-d2": "0", "qA-a5": "0"}))       # qC-d2: primary 1 vs 0 -> min ; qA-a5: primary 2 vs 0 -> unresolved
    third = planned["tmp"] / "annotator_03.csv"
    SL.merge(planned["out"], s, None, third)
    E2.cmd_lock(ns(a1=planned["a1"], a2=planned["a2"], third=third), planned["ev"], planned["out"])
    E2.cmd_evaluate(ns(B=100), planned["ev"], planned["priv"], planned["hm"], planned["out"], None)
    sc = json.loads((planned["out"] / "results.json").read_text())["label_aggregation"]["status_counts"]
    assert sc["second_judgment_min_rule"] == 1 and sc["second_judgment_unresolved"] == 1
    assert sc["min_rule"] == 1 and sc["unresolved"] == 1                                    # the two disagreements, still without adjudication


def test_gate_reads_counts_only_and_applies_thresholds():
    groups = {f"q{i}": f"G{i % 6}" for i in range(50)}
    # build counts: all valid
    def results(excl2=(), excl1v=(), excl1a=(), idcg0=()):
        valid2 = [q for q in groups if q not in set(excl2)]
        valid1 = [q for q in groups if q not in set(excl1v) | set(excl1a)]
        bg = lambda qs: {g: sum(1 for q in qs if groups[q] == g) for g in sorted(set(groups.values())) if any(groups[q] == g for q in qs)}  # noqa: E731
        return {"primary": {"retained": {"P5": len(valid2), "MeanRel5": len(valid2), "Hit5": len(valid2), "NDCG5": len(valid2) - len(idcg0), "RQ1": len(valid1)},
                            "retained_by_stratum": {"MeanRel5": bg(valid2), "RQ1": bg(valid1)},
                            "exclusions": {"RQ2_excluded_incomplete_Top5": {q: {"visual_missing": 1, "metadata_missing": 0} for q in excl2},
                                           "RQ2_NDCG5_excluded_IDCG_zero": list(idcg0), "RQ1_excluded_visual_Top5_incomplete": list(excl1v),
                                           "RQ1_excluded_anchor_unlabelled": list(excl1a)},
                            "metrics": "MUST_NOT_BE_READ"}}
    th = {"min_valid_queries": 40, "min_per_group": 4, "decision_status": "APPROVED_BY_RESEARCH_OWNER"}
    g = RG.compute_gate(results(), groups, th)
    assert g["RQ"]["RQ1"]["meets_threshold"] and g["RQ"]["RQ2"]["meets_threshold"] and g["metric_values_read"] is False
    assert g["overall"] == "MAIN_RESULTS_REPORTABLE_FOR_BOTH_RQ"
    # exactly 40 valid is enough ; 39 is not
    q = list(groups)
    g = RG.compute_gate(results(excl2=q[:10]), groups, th)
    assert g["RQ"]["RQ2"]["valid_queries"] == 40 and g["RQ"]["RQ2"]["meets_threshold"]
    g = RG.compute_gate(results(excl2=q[:11], excl1v=q[:1]), groups, th)
    assert g["RQ"]["RQ2"]["valid_queries"] == 39 and g["RQ"]["RQ2"]["status"] == "BELOW_THRESHOLD_DESCRIPTIVE_ONLY"
    assert any("39 < 40" in r for r in g["RQ"]["RQ2"]["reasons"])
    assert g["RQ"]["RQ1"]["meets_threshold"] and g["RQ"]["RQ2"]["meets_threshold"] is False
    # group condition: G0 has 9 queries; drop 6 of them -> 3 left in G0, total still 44 >= 40
    g0 = [x for x in q if groups[x] == "G0"]
    assert len(g0) == 9
    g = RG.compute_gate(results(excl2=g0[:5]), groups, th)                   # exactly 4 left in G0: boundary is INCLUSIVE (>= 4)
    assert g["RQ"]["RQ2"]["per_group"]["G0"] == 4 and g["RQ"]["RQ2"]["meets_threshold"] is True
    g = RG.compute_gate(results(excl2=g0[:6]), groups, th)
    assert g["RQ"]["RQ2"]["valid_queries"] >= 40 and g["RQ"]["RQ2"]["meets_threshold"] is False
    assert "G0" in json.dumps(g["RQ"]["RQ2"]["reasons"]) and g["RQ"]["RQ1"]["meets_threshold"]
    # a group with zero valid queries fails
    g = RG.compute_gate(results(excl1a=g0), groups, th)
    assert g["RQ"]["RQ1"]["per_group"]["G0"] == 0 and g["RQ"]["RQ1"]["meets_threshold"] is False
    # N4: composition published, sets separate, overlap computed
    c = g["composition_N4"]
    assert c["RQ1_only"] + c["RQ2_only"] + c["both"] + c["neither"] == 50 and "must not be compared" in c["statement"]
    # inconsistent results are refused, not guessed
    bad = results(); bad["primary"]["retained"]["RQ1"] = 49
    with pytest.raises(SystemExit, match="inconsistent|do not reproduce"):
        RG.compute_gate(bad, groups, th)


def test_gate_markdown_has_no_metric_and_states_thresholds():
    groups = {f"q{i}": f"G{i % 6}" for i in range(50)}
    res = {"primary": {"retained": {"P5": 50, "MeanRel5": 50, "Hit5": 50, "NDCG5": 48, "RQ1": 50},
                       "retained_by_stratum": {"MeanRel5": {f"G{i}": 8 + (i < 2) for i in range(6)}, "RQ1": {f"G{i}": 8 + (i < 2) for i in range(6)}},
                       "exclusions": {"RQ2_excluded_incomplete_Top5": {}, "RQ2_NDCG5_excluded_IDCG_zero": ["q1", "q2"],
                                      "RQ1_excluded_visual_Top5_incomplete": [], "RQ1_excluded_anchor_unlabelled": []}}}
    g = RG.compute_gate(res, groups, {"min_valid_queries": 40, "min_per_group": 4, "decision_status": "APPROVED_BY_RESEARCH_OWNER"})
    md = RG.to_markdown(g)
    assert "statistical power" in md and "RQ1" in md and "RQ2" in md and "không đọc giá trị metric" in md
    assert not re.search(r"\bmean\b|ci_low|ci_high|verdict", json.dumps(g))


def test_gate_requires_an_approved_decision_record(tmp_path):
    d = json.loads((W7 / "outputs/methodology/research_decisions.json").read_text())
    ok = RG.load_thresholds(W7 / "outputs/methodology/research_decisions.json")
    assert ok == {"min_valid_queries": 40, "min_per_group": 4, "decision_status": "APPROVED_BY_RESEARCH_OWNER"}
    d["status"] = "DRAFT"
    (tmp_path / "d.json").write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="APPROVED_BY_RESEARCH_OWNER"):
        RG.load_thresholds(tmp_path / "d.json")


def test_gate_source_never_reads_metric_fields_or_the_key():
    src = (W7 / "src/reporting_gate.py").read_text()
    body = src.split("def compute_gate")[1]
    assert '["metrics"]' not in body and "['metrics']" not in body and "verdict" not in body.split("def to_markdown")[0].replace("metric_values_read", "")
    for forbidden in ("annotation_key", "pool_manifest", "evaluation_private"):
        assert forbidden not in src


# ======================================================================= the interfaces in a real browser
def run_chrome(html_text: str, tmp: Path, extra_js: list, query: str = "?selftest=1", inline_js: str = "") -> str:
    page = tmp / f"page_{abs(hash(html_text + query + inline_js)) % 10**8}.html"
    inject = "<script>" + inline_js + "</script>" + "".join("<script>" + (W7 / "src" / n).read_text(encoding="utf-8") + "</script>" for n in extra_js)
    page.write_text(html_text.replace("</body>", inject + "</body>"), encoding="utf-8")
    prof = tmp / ("profile_" + page.stem)
    r = subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}", "--virtual-time-budget=9000",
                        "--dump-dom", f"file://{page}{query}"], capture_output=True, text=True, timeout=180)
    return r.stdout


needs_chrome = pytest.mark.skipif(not CHROME.exists(), reason="Google Chrome not available")


@needs_chrome
@pytest.mark.parametrize("mode", ["second_judgment", "adjudication"])
def test_subset_interface_selftest_in_headless_chrome(planned, mode):
    html = (planned["pkg"] / mode / f"{mode}_interface.html").read_text(encoding="utf-8")
    out = run_chrome(html, planned["tmp"], ["subset_selftest.js"])
    m = re.search(r"SELFTEST_RESULT\n(.*?)SELFTEST_END (\w+)", out, re.S)
    assert m, out[-500:]
    lines = m.group(1).splitlines()
    fails = [l for l in lines if l.startswith("FAIL")]
    assert fails == [] and m.group(2) == "PASS" and sum(l.startswith("PASS") for l in lines) >= 24, lines


def interface_export(planned, mode, annot, auto):
    html = (planned["pkg"] / mode / f"{mode}_interface.html").read_text(encoding="utf-8")
    inline = f"window.AUTO_ANNOT={json.dumps(annot)};window.AUTO={json.dumps(auto)};"
    out = run_chrome(html, planned["tmp"], ["subset_autolabel.js"], inline_js=inline)
    m = re.search(r"CSV_BEGIN\n(.*?)CSV_END\nREFUSED (.*?)</pre>", out, re.S)
    assert m, out[-600:]
    return m.group(1).replace("\r\n", "\n"), json.loads(m.group(2).strip())


@needs_chrome
def test_interface_exports_are_accepted_end_to_end_by_merge_lock_evaluate(planned):
    s_csv, refused = interface_export(planned, "second_judgment", "annotator_03", {"qC-d2": 1, "qA-a5": 2})
    assert refused == []
    a_csv, refused = interface_export(planned, "adjudication", "annotator_03", {"qB-b2": 2, "qB-b3": 0})
    assert refused == []
    s = planned["tmp"] / "annotator_03_second_judgment.csv"; a = planned["tmp"] / "annotator_03_adjudication.csv"
    s.write_text(s_csv, encoding="utf-8"); a.write_text(a_csv, encoding="utf-8")
    plan = SL.read_plan(planned["out"])
    assert SL.validate_file(s, "second_judgment", plan)["ok"] and SL.validate_file(a, "adjudication", plan)["ok"]
    third = planned["tmp"] / "annotator_03.csv"
    SL.merge(planned["out"], s, a, third)
    E2.cmd_lock(ns(a1=planned["a1"], a2=planned["a2"], third=third), planned["ev"], planned["out"])
    E2.cmd_evaluate(ns(B=150), planned["ev"], planned["priv"], planned["hm"], planned["out"], None)
    sc = json.loads((planned["out"] / "results.json").read_text())["label_aggregation"]["status_counts"]
    assert sc["second_judgment_agree"] == 2 and sc["adjudicated"] == 2
    rows = {r["pair_id"]: r for r in csv.DictReader(open(third, newline=""))}
    assert rows["qB-b2"]["role"] == "adjudication" and rows["qC-d2"]["role"] == "second_judgment"


@needs_chrome
def test_adjudication_interface_refuses_a_label_outside_the_two_existing_ones(planned):
    csv_text, refused = interface_export(planned, "adjudication", "annotator_03", {"qB-b2": 1, "qB-b3": 1, "qB-b3x": 0})
    assert "qB-b2:1" in refused and "qB-b3:1" not in refused                     # 1 is not in {0,2} but is in {0,1}
    rows = {r["pair_id"]: r["label"] for r in csv.DictReader(csv_text.splitlines(True))}
    assert rows["qB-b2"] == "" and rows["qB-b3"] == "1"


@needs_chrome
def test_primary_ids_cannot_use_the_subset_interfaces(planned):
    csv_text, refused = interface_export(planned, "second_judgment", "annotator_01", {"qC-d2": 1})
    assert csv_text.strip().count("\n") == 0 or "qC-d2:1" in refused           # no task loaded: nothing exported / nothing accepted


# ======================================================================= packaging / leaks / repository guards
def test_subset_package_contents_are_blind_safe(planned):
    pkg = planned["pkg"]
    assert not LEAK.findall((pkg / "subset_manifest.json").read_text(encoding="utf-8"))
    for mode in ("second_judgment", "adjudication"):
        d = pkg / mode
        assert {x.name for x in d.iterdir()} == {f"{mode}_interface.html", "annotation_guide.md", "images"}
        for n in (f"{mode}_interface.html", "annotation_guide.md"):
            assert not LEAK.findall((d / n).read_text(encoding="utf-8")), n
        assert all(re.fullmatch(r"img_[0-9a-f]{12}\.jpg", x.name) for x in (d / "images").iterdir())
        assert not list(d.rglob("*needed*")) and not list(d.rglob("*lock*")) and not list(d.rglob("*key*")) and not list(d.rglob("*.csv"))
    # the folder that displays existing labels must not contain the blind folder's page and vice versa (staged delivery)
    assert not (pkg / "adjudication" / "second_judgment_interface.html").exists() and not (pkg / "second_judgment" / "adjudication_interface.html").exists()


def test_instruction_files_are_blind_safe_and_complete():
    d = W7 / "outputs/instructions"
    files = {p.name: p.read_text(encoding="utf-8") for p in d.glob("*.md")}
    assert {"huong_dan_annotator_01_02.md", "huong_dan_annotator_03.md", "huong_dan_dieu_phoi.md"} <= set(files)
    for n in ("huong_dan_annotator_01_02.md", "huong_dan_annotator_03.md"):                # these go to annotators
        assert not LEAK.findall(files[n]), (n, LEAK.findall(files[n])[:4])
    assert "annotator_01" in files["huong_dan_annotator_01_02.md"] and "annotator_02" in files["huong_dan_annotator_01_02.md"]
    t = files["huong_dan_annotator_03.md"]
    assert "second_judgment_interface.html" in t and "adjudication_interface.html" in t and "chấm mù" in t.lower() and "không" in t
    c = files["huong_dan_dieu_phoi.md"]
    for cmd in ("lock", "plan", "prepare_subset_tasks.py", "subset_labels.py", "--third", "evaluate", "reporting_gate.py"):
        assert cmd in c, cmd


def test_decision_record_is_approved_and_consistent_with_json():
    md = (W7 / "outputs/methodology/research_decisions.md").read_text(encoding="utf-8")
    js = json.loads((W7 / "outputs/methodology/research_decisions.json").read_text())
    assert "APPROVED_BY_RESEARCH_OWNER" in md and js["status"] == "APPROVED_BY_RESEARCH_OWNER"
    assert js["decisions"]["N3"]["min_valid_queries_per_rq"] == 40 and js["decisions"]["N3"]["min_queries_per_product_group"] == 4
    assert js["decisions"]["N3"]["statistical_power_evidence"] is False
    for k in ("N1", "N2", "N3", "N4"):
        assert f"## {k}" in md
    assert "40/50" in md and "statistical power" in md


def test_frozen_components_official_interface_and_labels_untouched():
    for m in ("outputs/hm/protocol_manifest.json", "outputs/evaluation/evaluator_manifest.json", "outputs/methodology/evaluator_v2_manifest.json"):
        d = json.loads((W7 / m).read_text())
        assert [f for f, v in d["files"].items() if sha(ROOT / f) != v["sha256"]] == [], m
    am = json.loads((W7 / "outputs/evaluation/annotation_manifest.json").read_text())
    assert all(sha(W7 / "outputs/evaluation" / n) == h for n, h in am["hashes"].items())      # official interface + guide + tasks unchanged
    mm = json.loads((W7 / "outputs/methodology/methodology_manifest.json").read_text())
    assert [f for f, v in mm["files"].items() if sha(ROOT / f) != v["sha256"]] == []
    assert list(W7.rglob("annotator_0*.csv")) == [] and not (W7 / "outputs/evaluation_results").exists() and not (W7 / "outputs/subset_package").exists()
    assert not list(W7.rglob("second_judgment_needed.csv")) and not list(W7.rglob("adjudication_needed.csv"))


def test_new_code_never_opens_the_private_mapping():
    for n in ("prepare_subset_tasks.py", "subset_labels.py", "reporting_gate.py", "subset_interface_template.html"):
        s = (W7 / "src" / n).read_text(encoding="utf-8")
        assert "annotation_key" not in s and "pool_manifest" not in s and "evaluation_private" not in s, n
