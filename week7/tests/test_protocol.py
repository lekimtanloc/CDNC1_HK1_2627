"""Validity tests for the Week 7 H&M protocol (no CNN, no ranking of real queries).

Run from repo root:  python3 -I -m pytest -q week7/tests/test_protocol.py -p no:cacheprovider
Data checks recompute everything from articles.csv / hm_catalog.json / image files,
independently of build_hm_protocol.py.
"""
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
OUT = W7 / "outputs" / "hm"
sys.path.insert(0, str(W7 / "scripts"))
import protocol_lib as L  # noqa: E402

EXP = json.loads((W7 / "config/experiment.json").read_text())
EVAL = json.loads((W7 / "config/evaluation.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rcsv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def data():
    arts = {}
    with open(ROOT / EXP["data"]["articles_csv"], encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            arts[r["article_id"]] = r
    cat = json.loads((ROOT / EXP["data"]["catalog_manifest"]).resolve().read_text())
    return arts, cat, rcsv(OUT / "product_universe.csv"), rcsv(OUT / "query_manifest.csv"), rcsv(OUT / "pilot_manifest.csv")


# ---------------- dataset / universe ----------------
def test_universe_is_one_representative_per_product_code(data):
    arts, cat, U, _, _ = data
    ids = [c["articleId"] for c in cat]
    assert len(ids) == len(set(ids)) == 500 and all(a in arts for a in ids)
    fam = defaultdict(list)
    for a in ids:
        fam[arts[a]["product_code"]].append(a)
    expected = {pc: min(v, key=lambda a: hashlib.sha256(f"42:rep:{a}".encode()).hexdigest()) for pc, v in fam.items()}
    assert len(U) == len(expected) == 145
    assert {r["product_code"]: r["article_id"] for r in U} == expected
    assert len({r["product_code"] for r in U}) == len(U)


def test_universe_rows_match_articles_csv(data):
    arts, _, U, _, _ = data
    for r in U:
        a = arts[r["article_id"]]
        for f in ("product_code", "product_type_no", "colour_group_code", "garment_group_no", "graphical_appearance_no",
                  "index_group_no", "perceived_colour_master_id", "perceived_colour_value_id", "product_group_name"):
            assert r[f] == a[f], (r["article_id"], f)


def test_universe_images_valid_and_unique(data):
    _, _, U, _, _ = data
    hashes = []
    for r in U:
        p = ROOT / r["image_relpath"]
        assert sha(p) == r["image_sha256"]
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            assert min(im.convert("RGB").size) >= 300
        hashes.append(r["image_sha256"])
    assert len(set(hashes)) == len(hashes)


def test_metadata_fields_exist_in_articles_csv(data):
    arts, _, _, _, _ = data
    cols = set(next(iter(arts.values())))
    w = EXP["methods"]["M-ATTR-TFIDF"]["weights"]
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)
    for f in w:
        assert (f if f != "detail_desc_tfidf" else "detail_desc") in cols


# ---------------- queries ----------------
def test_query_manifest_independent_resample(data):
    _, _, U, Q, P = data
    groups = defaultdict(list)
    for r in U:
        groups[r["product_group_name"]].append(r["article_id"])
    sizes = {g: len(v) for g, v in groups.items()}
    total, floor = EXP["query_sampling"]["max_queries"], 5
    rest = total - floor * len(sizes)
    quota = {g: rest * sizes[g] / sum(sizes.values()) for g in sizes}
    alloc = {g: floor + int(quota[g]) for g in sizes}
    for g in sorted(sizes, key=lambda g: (-(quota[g] - int(quota[g])), g))[: total - sum(alloc.values())]:
        alloc[g] += 1
    rng = random.Random(42)
    exp_q = [a for g in sorted(groups) for a in sorted(rng.sample(sorted(groups[g]), alloc[g]))]
    assert [r["article_id"] for r in Q] == exp_q
    assert Counter(r["product_group_name"] for r in Q) == Counter(alloc)
    assert len(Q) == 50 and min(alloc.values()) >= 5
    left = sorted(set(r["article_id"] for r in U) - set(exp_q))
    assert [r["article_id"] for r in P] == sorted(random.Random(4242).sample(left, 5))


def test_queries_valid_unique_and_disjoint_from_pilot(data):
    _, _, U, Q, P = data
    uid = {r["article_id"]: r for r in U}
    assert len({r["article_id"] for r in Q}) == len({r["product_code"] for r in Q}) == 50
    assert not ({r["article_id"] for r in Q} & {r["article_id"] for r in P})
    for r in Q:
        assert r["article_id"] in uid and r["image_sha256"] == uid[r["article_id"]]["image_sha256"]
        assert math.isclose(float(r["inclusion_prob"]), int(r["stratum_queries"]) / int(r["stratum_size"]), abs_tol=1e-6)


def test_candidate_pool_shared_and_excludes_query(data):
    _, _, U, Q, _ = data
    ids = [r["article_id"] for r in U]
    for q in Q:
        pool = [c for c in ids if c != q["article_id"]]
        assert len(pool) == 144 and q["article_id"] not in pool
    # same pool definition for both methods
    assert EXP["candidate_pool"]["shared_by_methods"] == list(EXP["methods"])


def test_no_ranking_or_embedding_exists_yet():
    banned = [p for p in W7.rglob("*") if p.suffix in {".npy", ".pt", ".pth"} or "top5" in p.name or "ranking" in p.name
              or "labels" in p.name]
    assert banned == [], banned


# ---------------- reference definitions ----------------
def test_rank_excludes_query_and_ties_deterministic():
    s = {"q": 9.0, "a": 1.0, "b": 1.0, "c": 0.5}
    r1 = L.rank("q", s, k=3)
    assert "q" not in r1 and r1 == L.rank("q", dict(reversed(list(s.items()))), k=3)
    assert set(r1[:2]) == {"a", "b"} and r1[2] == "c"
    assert r1[:2] == sorted(["a", "b"], key=lambda c: L.tie_key("q", c))


def test_metadata_score_missing_contributes_zero():
    w = {"colour_group_code": 0.5, "product_type_no": 0.4, "detail_desc_tfidf": 0.1}
    v = L.tfidf_vectors({"q": "black slim jeans", "c": "black slim jeans", "d": ""})
    assert math.isclose(L.metadata_score({"colour_group_code": "09", "product_type_no": "272"},
                                         {"colour_group_code": "09", "product_type_no": "272"}, w, v["q"], v["c"]), 1.0)
    assert L.metadata_score({"colour_group_code": "-1", "product_type_no": "272"},
                            {"colour_group_code": "-1", "product_type_no": "1"}, w, v["q"], v["d"]) == 0.0
    assert L.is_missing("Undefined") and L.is_missing("") and not L.is_missing("0")


def test_metrics_hand_computed():
    rels = [2, 0, 1, None, 2]
    assert math.isclose(L.precision_at5(rels), 3 / 4)
    assert math.isclose(L.mean_rel_at5(rels), 5 / 4)
    assert L.hit_at5(rels) is None and L.pooled_ndcg_at5(rels, rels) is None  # missing never becomes 0
    full = [2, 0, 1, 0, 2]
    pool = full + [2, 1]
    dcg = 2 / math.log2(2) + 1 / math.log2(4) + 2 / math.log2(6)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 2 / math.log2(4) + 1 / math.log2(5) + 1 / math.log2(6)
    assert math.isclose(L.pooled_ndcg_at5(full, pool), dcg / idcg)
    assert L.hit_at5(full) == 1.0 and L.hit_at5([1, 1, 0, 0, 1]) == 0.0
    assert L.pooled_ndcg_at5([0] * 5, [0] * 7) is None
    assert L.precision_at5([None] * 5) is None


def test_label_aggregation_rules():
    assert L.aggregate(2, 2) == 2
    assert L.aggregate(1, 2) == 1 and L.aggregate(2, 1) == 1
    assert L.aggregate(0, 2) is None
    assert L.aggregate(0, 2, adjudicated=2) == 2
    assert L.aggregate(None, None) is None and L.aggregate(None, 1) == 1
    with pytest.raises(ValueError):
        L.aggregate(0, 2, adjudicated=1)


def test_weighted_kappa_reference_values():
    a = [0, 1, 2, 2, 1, 0, 2, 1]
    assert math.isclose(L.weighted_kappa(a, a), 1.0)
    b = [0, 2, 2, 1, 1, 0, 2, 0]
    # hand computation, linear weights
    n = 8
    obs = Counter(zip(a, b))
    pa, pb = Counter(a), Counter(b)
    do = sum(abs(i - j) / 2 * c / n for (i, j), c in obs.items())
    de = sum(abs(i - j) / 2 * pa[i] / n * pb[j] / n for i in range(3) for j in range(3))
    assert math.isclose(L.weighted_kappa(a, b), 1 - do / de)


def test_paired_bootstrap_deterministic_and_centered():
    d = {"g1": [0.2, 0.0, 0.4], "g2": [0.1, -0.1]}
    r1, r2 = L.paired_bootstrap(d, b=2000), L.paired_bootstrap(d, b=2000)
    assert r1 == r2 and r1[1] <= r1[0] <= r1[2] and math.isclose(r1[0], 0.6 / 5)


# ---------------- freeze ----------------
def test_protocol_manifest_hashes_match():
    m = OUT / "protocol_manifest.json"
    if not m.exists():
        pytest.skip("manifest not yet written")
    man = json.loads(m.read_text())
    for f, v in man["files"].items():
        if f.endswith("test_protocol.py"):
            continue
        assert sha(ROOT / f) == v["sha256"], f
