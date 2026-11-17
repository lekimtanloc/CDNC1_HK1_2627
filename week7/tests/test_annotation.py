"""Tests for round 4: metadata baseline, annotation package, validator, interface.

Metadata scoring is re-implemented here independently (no import of metadata_baseline / protocol_lib).
Label fixtures exist only inside pytest tmp dirs and are never written to the official outputs.
Run: python3 -I -B -m pytest -q week7/tests/test_annotation.py -p no:cacheprovider
"""
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
HM = W7 / "outputs/hm"
EV = W7 / "outputs/evaluation"
PRIV = W7 / "outputs/evaluation_private"
sys.path.insert(0, str(W7 / "src"))
import validate_annotations as VA  # noqa: E402

EXP = json.loads((W7 / "config/experiment.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rcsv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ======================= frozen protocol & upstream outputs =======================
def test_all_frozen_files_unchanged_including_guard_test():
    man = json.loads((HM / "protocol_manifest.json").read_text())
    assert len(man["files"]) == 10
    bad = [f for f, v in man["files"].items() if sha(ROOT / f) != v["sha256"]]
    assert bad == []


def test_main_queries_and_visual_top5_unchanged():
    man = json.loads((HM / "protocol_manifest.json").read_text())
    assert sha(HM / "query_manifest.csv") == man["files"]["week7/outputs/hm/query_manifest.csv"]["sha256"]
    log = json.loads((HM / "visual_run_log.json").read_text())
    for n, h in log["outputs_sha256"].items():
        assert sha(HM / n) == h
    Q = rcsv(HM / "query_manifest.csv")
    assert len(Q) == 50 and len({q["article_id"] for q in Q}) == 50
    assert len(rcsv(HM / "visual_top5.csv")) == 250


# ======================= metadata baseline =======================
def _independent_metadata():
    w = EXP["methods"]["M-ATTR-TFIDF"]["weights"]
    U = rcsv(HM / "product_universe.csv")
    arts = {}
    with open(ROOT / EXP["data"]["articles_csv"], encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            arts[r["article_id"]] = r
    miss = lambda v: v is None or v.strip().lower() in {"", "-1", "unknown", "undefined"}  # noqa: E731
    docs = {r["article_id"]: re.findall(r"[a-z0-9]+", arts[r["article_id"]]["detail_desc"].lower()) for r in U}
    N = len(docs)
    df = Counter(t for ts in docs.values() for t in set(ts))
    vec = {}
    for a, ts in docs.items():
        v = {t: (1 + math.log(c)) * (math.log((1 + N) / (1 + df[t])) + 1) for t, c in Counter(ts).items()}
        n = math.sqrt(sum(x * x for x in v.values()))
        vec[a] = {t: x / n for t, x in v.items()} if n else {}
    out = {}
    for q in rcsv(HM / "query_manifest.csv"):
        qa = arts[q["article_id"]]
        sc = []
        for r in U:
            if r["article_id"] == q["article_id"]:
                continue
            ca = arts[r["article_id"]]
            s = 0.0
            for f, wt in w.items():
                if f == "detail_desc_tfidf":
                    s += wt * sum(x * vec[r["article_id"]].get(t, 0.0) for t, x in vec[q["article_id"]].items())
                elif not miss(qa[f]) and not miss(ca[f]) and qa[f] == ca[f]:
                    s += wt
            sc.append((round(s, 9), hashlib.sha256(f"42:{q['article_id']}:{r['article_id']}".encode()).hexdigest(), r["article_id"]))
        sc.sort(key=lambda t: (-t[0], t[1]))
        out[q["article_id"]] = sc[:5]
    return out


def test_metadata_top5_matches_independent_reimplementation():
    exp = _independent_metadata()
    rows = rcsv(HM / "metadata_top5.csv")
    got = {}
    for r in rows:
        got.setdefault(r["query_id"], []).append(r)
    assert set(got) == set(exp)
    for q, lst in got.items():
        assert [r["candidate_id"] for r in lst] == [t[2] for t in exp[q]], q
        for r, t in zip(lst, exp[q]):
            assert abs(float(r["metadata_score"]) - t[0]) < 1e-8


def test_metadata_top5_structure_and_no_duplicates():
    rows = rcsv(HM / "metadata_top5.csv")
    U = {r["article_id"]: r for r in rcsv(HM / "product_universe.csv")}
    assert len(rows) == 250
    by = {}
    for r in rows:
        by.setdefault(r["query_id"], []).append(r)
    assert len(by) == 50
    for q, lst in by.items():
        assert [int(r["rank"]) for r in lst] == [1, 2, 3, 4, 5]
        c = [r["candidate_id"] for r in lst]
        assert q not in c and len(set(c)) == 5 and all(x in U for x in c)
        assert U[q]["product_code"] not in {U[x]["product_code"] for x in c}
        s = [float(r["metadata_score"]) for r in lst]
        assert s == sorted(s, reverse=True)
    assert len({(r["query_id"], r["candidate_id"]) for r in rows}) == 250


def test_weights_are_the_frozen_config_values_and_not_tuned():
    log = json.loads((HM / "metadata_run_log.json").read_text())
    assert log["weights"] == EXP["methods"]["M-ATTR-TFIDF"]["weights"]
    assert math.isclose(sum(log["weights"].values()), 1.0)
    assert "CNN embeddings" in log["not_used"] and "relevance labels" in log["not_used"]
    src = (W7 / "src/metadata_baseline.py").read_text()
    assert "visual_top5" in src  # only for the descriptive comparison table
    assert "visual" not in src.split("def compute")[1].split("def to_rows")[0]  # ranking code path never touches visual data


def test_candidate_pool_identical_for_visual_and_metadata():
    vjs = json.loads((HM / "visual_top5.json").read_text())
    U = [r["article_id"] for r in rcsv(HM / "product_universe.csv")]
    assert sorted(vjs["candidate_ids_valid"]) == sorted(U) and vjs["candidates_excluded_invalid_image"] == []
    comp = rcsv(HM / "method_comparison.csv")
    assert len(comp) == 50
    for c in comp:
        assert c["pools_identical"] == "True" and c["pool_visual"] == c["pool_metadata"] == "144"
        assert c["visual_top5_in_pool"] == c["metadata_top5_in_pool"] == "True"
        assert c["query_in_visual_top5"] == c["query_in_metadata_top5"] == "False"
    vis = {}
    for r in rcsv(HM / "visual_top5.csv"):
        vis.setdefault(r["query_id"], []).append(r["candidate_id"])
    for c in comp:
        m = c["metadata_top5"].split("|")
        assert c["visual_top5"].split("|") == vis[c["query_id"]]
        assert int(c["overlap_count"]) == len(set(m) & set(vis[c["query_id"]]))


def test_metadata_run_is_reproducible():
    before = sha(HM / "metadata_top5.csv"), sha(HM / "method_comparison.csv")
    subprocess.run([sys.executable, "-I", "-B", str(W7 / "src/metadata_baseline.py")], check=True, capture_output=True, cwd=ROOT)
    assert (sha(HM / "metadata_top5.csv"), sha(HM / "method_comparison.csv")) == before


# ======================= annotation package =======================
def _union_pairs():
    vis, meta = {}, {}
    for r in rcsv(HM / "visual_top5.csv"):
        vis.setdefault(r["query_id"], []).append(r["candidate_id"])
    for r in rcsv(HM / "metadata_top5.csv"):
        meta.setdefault(r["query_id"], []).append(r["candidate_id"])
    return vis, meta


def test_main_pairs_follow_frozen_pool_rule():
    vis, meta = _union_pairs()
    U = [r["article_id"] for r in rcsv(HM / "product_universe.csv")]
    key = [k for k in rcsv(PRIV / "annotation_key.csv") if k["set"] == "main"]
    got = {}
    for k in key:
        got.setdefault(k["query_id"], set()).add(k["candidate_id"])
    assert set(got) == set(vis) and len(key) <= 600
    for q in vis:
        union = set(vis[q]) | set(meta[q])
        rest = sorted((c for c in U if c != q and c not in union),
                      key=lambda c: hashlib.sha256(f"42:anchor:{q}:{c}".encode()).hexdigest())
        assert got[q] == union | set(rest[:2]), q
        assert q not in got[q] and 7 <= len(got[q]) <= 12
    assert len({(k["query_id"], k["candidate_id"]) for k in key}) == len(key)  # duplicates collapsed once


def test_key_is_consistent_with_rankings():
    vis, meta = _union_pairs()
    for k in rcsv(PRIV / "annotation_key.csv"):
        if k["set"] != "main":
            continue
        q, c = k["query_id"], k["candidate_id"]
        assert (k["in_visual_top5"] == "True") == (c in vis[q])
        assert (k["in_metadata_top5"] == "True") == (c in meta[q])
        if c in vis[q]:
            assert int(k["visual_rank"]) == vis[q].index(c) + 1
        if c in meta[q]:
            assert int(k["metadata_rank"]) == meta[q].index(c) + 1
        assert (k["is_anchor"] == "True") == (c not in vis[q] and c not in meta[q])


def test_pilot_is_separate_and_not_in_main():
    Q = {q["article_id"] for q in rcsv(HM / "query_manifest.csv")}
    P = {p["article_id"] for p in rcsv(HM / "pilot_manifest.csv")}
    assert len(P) == 5 and not (P & Q)
    tasks = rcsv(EV / "annotation_tasks.csv")
    key = {k["pair_id"]: k for k in rcsv(PRIV / "annotation_key.csv")}
    for t in tasks:
        k = key[t["pair_id"]]
        assert k["set"] == t["set"]
        assert (k["query_id"] in P) == (t["set"] == "pilot")
    ids = [t["pair_id"] for t in tasks]
    assert len(ids) == len(set(ids)) == len(key)
    assert Counter(t["set"] for t in tasks) == {"main": 502, "pilot": 48}


def test_orders_are_permutations_and_differ_between_annotators():
    tasks = rcsv(EV / "annotation_tasks.csv")
    for s in ("main", "pilot"):
        ts = [t for t in tasks if t["set"] == s]
        for col in ("order_annotator_01", "order_annotator_02", "order_annotator_03"):
            assert sorted(int(t[col]) for t in ts) == list(range(len(ts)))
        assert [t["order_annotator_01"] for t in ts] != [t["order_annotator_02"] for t in ts]


def test_images_exist_neutral_names_and_originals_untouched():
    tasks = rcsv(EV / "annotation_tasks.csv")
    U = rcsv(HM / "product_universe.csv")
    for t in tasks:
        for col in ("query_image", "candidate_image"):
            assert (EV / t[col]).is_file() and re.fullmatch(r"images/img_[0-9a-f]{12}\.jpg", t[col])
    for r in U:   # originals still match the frozen hash
        assert sha(ROOT / r["image_relpath"]) == r["image_sha256"]


def test_no_method_rank_score_or_anchor_leak_in_annotator_folder():
    words = re.compile(r"visual|metadata|cosine|resnet|anchor|top-?5|rank|score|method|baseline|\bcnn\b|embedding|tf-?idf", re.I)
    # evaluator_* files (round 5) live in this folder by instruction but are NEVER packaged: see make_annotator_package.py
    for p in EV.iterdir():
        if p.is_file() and not p.name.startswith("evaluator_"):
            hit = words.findall(p.read_text(encoding="utf-8"))
            assert hit == [], (p.name, hit[:5])
    assert sorted(x.name for x in EV.iterdir() if x.is_file() and not x.name.startswith("evaluator_")) == [
        "annotation_guide.md", "annotation_interface.html", "annotation_manifest.json", "annotation_tasks.csv",
        "annotation_template.csv", "annotation_template_pilot.csv"]
    # order of tasks file is by pair_id (hash) -> carries no information about methods
    ids = [t["pair_id"] for t in rcsv(EV / "annotation_tasks.csv") if t["set"] == "main"]
    assert ids == sorted(ids)


def test_manifest_hashes_and_state():
    man = json.loads((EV / "annotation_manifest.json").read_text())
    assert man["state"] == "AWAITING_HUMAN_LABELS" and man["labels_present"] is False
    for n, h in man["hashes"].items():
        assert sha(EV / n) == h, n
    assert man["sets"]["main"]["pairs"] == 502 <= 600 and man["sets"]["pilot"]["pairs"] == 48
    pm = json.loads((PRIV / "pool_manifest.json").read_text())
    assert pm["annotation_key_sha256"] == sha(PRIV / "annotation_key.csv")
    assert pm["inputs_sha256"]["visual_top5.csv"] == sha(HM / "visual_top5.csv")


def test_no_fake_labels_in_official_outputs():
    assert list(EV.glob("annotator_*.csv")) == [] and list(W7.rglob("annotator_0*.csv")) == []
    for n in ("annotation_template.csv", "annotation_template_pilot.csv"):
        rows = rcsv(EV / n)
        assert rows and all(r["label"] == "" and r["annotator_id"] == "" and r["uncertain"] == "" and r["note"] == "" for r in rows)
    assert not any("label" in p.name and p.suffix == ".csv" and "template" not in p.name for p in (W7 / "outputs").rglob("*"))


def test_prepare_annotations_is_deterministic():
    names = ["annotation_tasks.csv", "annotation_template.csv", "annotation_template_pilot.csv", "annotation_interface.html"]
    before = {n: sha(EV / n) for n in names} | {"key": sha(PRIV / "annotation_key.csv")}
    # prepare_annotations.py rewrites annotation_manifest.json / pool_manifest.json with a fresh timestamp: restore them so that running the
    # tests never changes the hashes recorded in the manifests (side effect found in round 7 and fixed)
    saved = {p: p.read_bytes() for p in (EV / "annotation_manifest.json", PRIV / "pool_manifest.json")}
    try:
        subprocess.run([sys.executable, "-I", "-B", str(W7 / "src/prepare_annotations.py")], check=True, capture_output=True, cwd=ROOT)
        after = {n: sha(EV / n) for n in names} | {"key": sha(PRIV / "annotation_key.csv")}
    finally:
        for p, data in saved.items():
            p.write_bytes(data)
    assert before == after


# ======================= validator =======================
HEAD = ["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc"]


@pytest.fixture(scope="module")
def tasks():
    return VA.load_tasks()


def write(path, rows, head=HEAD):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)
    return path


def rows_for(tasks, set_, annot, labels, n=None):
    ids = sorted(tasks[set_])[: n or len(tasks[set_])]
    return [[p, set_, annot, labels(i), "0", "", ""] for i, p in enumerate(ids)]


def test_validator_accepts_complete_and_partial(tmp_path, tasks):
    full = write(tmp_path / "annotator_01.csv", rows_for(tasks, "main", "annotator_01", lambda i: str(i % 3)))
    r = VA.validate_many([full], "main", tasks)
    assert r["status"] == "COMPLETE" and r["files"][0]["pairs_answered"] == 502
    part = write(tmp_path / "annotator_02.csv", rows_for(tasks, "main", "annotator_02", lambda i: "1" if i < 100 else ""))
    r = VA.validate_many([part], "main", tasks)
    assert r["status"] == "PARTIAL" and r["files"][0]["pairs_unanswered"] == 402
    assert VA.validate_many([part], "main", tasks, require_complete=True)["status"] == "INVALID"


def test_validator_flags_bad_labels_missing_duplicates(tmp_path, tasks):
    base = rows_for(tasks, "main", "annotator_01", lambda i: "1")
    for bad in ("3", "-1", "x", "1.5", "two"):
        rows = [r[:] for r in base]; rows[0][3] = bad
        r = VA.validate_many([write(tmp_path / f"annotator_01_{bad}.csv", rows)], "main", tasks)
        assert r["status"] == "INVALID" and any("outside" in e for e in r["errors"]), bad
    rows = [r[:] for r in base] + [base[0][:]]
    assert any("duplicate" in e for e in VA.validate_many([write(tmp_path / "annotator_01_dup.csv", rows)], "main", tasks)["errors"])
    rows = [r[:] for r in base]; rows[1][2] = ""
    assert any("without annotator_id" in e for e in VA.validate_many([write(tmp_path / "annotator_01_noid.csv", rows)], "main", tasks)["errors"])
    rows = [r[:] for r in base]; rows[2][2] = "annotator_02"
    assert any("several annotator_id" in e for e in VA.validate_many([write(tmp_path / "annotator_01_mix.csv", rows)], "main", tasks)["errors"])
    rows = [r[:] for r in base]; rows[3][0] = "Pdeadbeef00"
    assert any("not in the main task list" in e for e in VA.validate_many([write(tmp_path / "annotator_01_unk.csv", rows)], "main", tasks)["errors"])
    rows = [r[:] for r in base]; rows[4][4] = "yes"
    assert any("uncertain" in e for e in VA.validate_many([write(tmp_path / "annotator_01_unc.csv", rows)], "main", tasks)["errors"])
    r = VA.validate_many([write(tmp_path / "annotator_01_nocol.csv", [[r[0], r[3]] for r in base], head=["pair_id", "label"])], "main", tasks)
    assert r["status"] == "INVALID" and "missing columns" in r["errors"][0]


def test_validator_blocks_mixing_pilot_into_main_and_missing_rows(tmp_path, tasks):
    pil = rows_for(tasks, "pilot", "annotator_01", lambda i: "1")
    r = VA.validate_many([write(tmp_path / "annotator_01.csv", pil)], "main", tasks)
    assert r["status"] == "INVALID" and any("not in set 'main'" in e for e in r["errors"])
    assert VA.validate_many([write(tmp_path / "annotator_01_pilot.csv", pil)], "pilot", tasks)["status"] == "COMPLETE"
    few = write(tmp_path / "annotator_01_few.csv", rows_for(tasks, "main", "annotator_01", lambda i: "2", n=10))
    r = VA.validate_many([few], "main", tasks)
    assert r["files"][0]["pairs_present"] == 10 and any("absent" in w for w in r["files"][0]["warnings"])


def test_validator_u_is_missing_not_zero_and_flagged(tmp_path, tasks):
    rows = rows_for(tasks, "main", "annotator_01", lambda i: "U" if i < 5 else "0")
    r = VA.validate_many([write(tmp_path / "annotator_01.csv", rows)], "main", tasks)["files"][0]
    assert r["label_counts"]["U"] == 5 and r["label_counts"]["0"] == len(rows) - 5 and any("never 0" in w for w in r["warnings"])


def test_validator_two_annotators_must_be_distinct(tmp_path, tasks):
    a = write(tmp_path / "annotator_01.csv", rows_for(tasks, "main", "annotator_01", lambda i: "1"))
    b = write(tmp_path / "annotator_02.csv", rows_for(tasks, "main", "annotator_02", lambda i: "2"))
    r = VA.validate_many([a, b], "main", tasks)
    assert r["status"] == "COMPLETE" and r["cross_file"]["pairs_in_all_files"] == 502
    c = write(tmp_path / "annotator_03.csv", rows_for(tasks, "main", "annotator_01", lambda i: "2"))
    assert VA.validate_many([a, c], "main", tasks)["status"] == "INVALID"


def test_blank_official_template_is_partial_with_zero_answers(tasks):
    r = VA.validate_many([EV / "annotation_template.csv"], "main", tasks)
    assert r["files"][0]["pairs_answered"] == 0 and r["files"][0]["pairs_unanswered"] == 502
    assert r["status"] == "INVALID"      # no annotator_id yet -> not a valid submission


def test_validator_detects_tampered_tasks_file(tmp_path):
    ev = tmp_path / "ev"; ev.mkdir()
    for n in ("annotation_tasks.csv", "annotation_manifest.json"):
        shutil.copy(EV / n, ev / n)
    VA.load_tasks(ev)
    (ev / "annotation_tasks.csv").write_text((ev / "annotation_tasks.csv").read_text() + "x\n")
    with pytest.raises(RuntimeError):
        VA.load_tasks(ev)


def test_validator_cli_exit_codes(tmp_path, tasks):
    good = write(tmp_path / "annotator_01.csv", rows_for(tasks, "main", "annotator_01", lambda i: "1", n=20))
    bad = write(tmp_path / "annotator_02.csv", rows_for(tasks, "main", "annotator_02", lambda i: "9", n=20))
    cmd = [sys.executable, "-I", "-B", str(W7 / "src/validate_annotations.py"), "--set", "main"]
    assert subprocess.run(cmd + [str(good)], capture_output=True, cwd=ROOT).returncode == 0
    assert subprocess.run(cmd + [str(bad)], capture_output=True, cwd=ROOT).returncode == 1


def test_validator_does_not_compute_quality_metrics():
    src = (W7 / "src/validate_annotations.py").read_text().lower()
    assert "kappa(" not in src and "ndcg" not in src.replace("no precision/ndcg/kappa", "")


# ======================= interface in a real browser =======================
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


@pytest.mark.skipif(not CHROME.exists(), reason="Google Chrome not available")
def test_interface_selftest_in_headless_chrome(tmp_path):
    html = (EV / "annotation_interface.html").read_text(encoding="utf-8")
    js = (W7 / "src/annotation_selftest.js").read_text(encoding="utf-8")
    page = tmp_path / "t.html"
    page.write_text(html.replace("</body>", "<script>" + js + "</script></body>"), encoding="utf-8")
    prof = tmp_path / "profile"
    out = subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}",
                          "--virtual-time-budget=8000", "--dump-dom", f"file://{page}?selftest=1"],
                         capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"SELFTEST_RESULT\n(.*?)SELFTEST_END (\w+)", out, re.S)
    assert m, "selftest output not found"
    fails = [l for l in m.group(1).splitlines() if l.startswith("FAIL")]
    assert fails == [] and m.group(2) == "PASS" and m.group(1).count("PASS") >= 20


def test_annotator_zip_uses_allowlist_and_excludes_evaluator_files(tmp_path):
    import make_annotator_package as MP
    out = tmp_path / "pkg.zip"
    names = MP.build(out)
    import zipfile
    with zipfile.ZipFile(out) as z:
        listed = z.namelist()
        assert sorted(listed) == sorted(names)
        assert not [n for n in listed if "evaluator_" in n or "private" in n or "key" in n.lower()]
        assert {f"evaluation/{n}" for n in MP.ALLOWED} <= set(listed)
        assert len([n for n in listed if n.startswith("evaluation/images/")]) == 140
        blob = b"".join(z.read(n) for n in listed if n.endswith((".md", ".html", ".json", ".csv"))).decode("utf-8")
    assert not re.search(r"visual|metadata|cosine|resnet|anchor|top-?5|rank|score|method|baseline", blob, re.I)
