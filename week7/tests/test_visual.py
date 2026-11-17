"""Tests for the Week 7 visual pipeline.
Unit/integration tests use REAL H&M images (small subset) and in-memory arrays only for pure math.
Output tests validate the produced files and are skipped until the run exists.
Run: python3 -I -B -m pytest -q week7/tests/test_visual.py -p no:cacheprovider
"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "week7" / "src"))
import extract_embeddings as X  # noqa: E402
import visual_retrieval as V  # noqa: E402

OUT = ROOT / "week7/outputs/hm"
UNI = X.load_universe()
SUBSET = UNI[:8]


@pytest.fixture(scope="module")
def model():
    return X.load_model()


@pytest.fixture(scope="module")
def sub_emb(model):
    return X.l2_normalize(X.extract(SUBSET, model))


# ---- ID mapping / images ----
def test_id_mapping_matches_articles_csv_and_path_rule():
    arts = {}
    with open(ROOT / "dataset_sources/hm-dataset/articles.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            arts[r["article_id"]] = r
    for r in UNI:
        a = r["article_id"]
        assert len(a) == 10 and a in arts and arts[a]["product_code"] == r["product_code"]
        assert r["image_relpath"] == f"dataset_sources/hm-dataset/images/{a[:3]}/{a}.jpg"
        assert (ROOT / r["image_relpath"]).is_file()


def test_all_universe_images_readable_with_stated_rule():
    res = [X.check_image(r) for r in UNI]
    assert all(ok for ok, _ in res), [r for r in res if not r[0]]


def test_image_check_rules_flag_bad_inputs(tmp_path):
    r = dict(UNI[0])
    assert X.check_image(r) == (True, "ok")
    assert X.check_image({**r, "image_sha256": "0" * 64}) == (False, "sha256_mismatch")
    assert X.check_image({**r, "image_relpath": "dataset_sources/hm-dataset/images/xxx/0000000000.jpg"})[1] == "id_path_mismatch"


# ---- preprocessing ----
def test_preprocess_shape_aspect_and_fill_rule():
    im = Image.new("RGB", (100, 200), (200, 100, 50))           # in-memory unit input for the pure function
    out = X.preprocess_pil(im)
    assert out.size == (224, 224)
    a = np.asarray(out)
    assert tuple(a[0, 0]) == (200, 100, 50) and tuple(a[112, 112]) == (200, 100, 50)  # fill = border mean
    # resized to 112x224, centred: left padding 56 px, content (same colour here) -> check geometry via a marked image
    m = Image.new("RGB", (100, 200), (255, 255, 255))
    m.paste((0, 0, 0), (0, 80, 100, 120))
    o = np.asarray(X.preprocess_pil(m))
    cols_dark = np.where((o.sum(axis=2) < 30).any(axis=0))[0]
    assert cols_dark.min() >= 56 - 1 and cols_dark.max() <= 168 + 1


def test_to_tensor_normalization_values():
    t = X.to_tensor(Image.new("RGB", (224, 224), (255, 255, 255)))
    assert tuple(t.shape) == (1, 3, 224, 224)
    for c, (mu, sd) in enumerate(zip(X.PREPROCESSING_SPEC["normalize_mean"], X.PREPROCESSING_SPEC["normalize_std"])):
        assert abs(float(t[0, c, 0, 0]) - (1 - mu) / sd) < 1e-5


# ---- embeddings ----
def test_weights_are_imagenet1k_v2():
    assert X.sha_file(X.WEIGHTS_FILE).startswith("11ad3fa6")


def test_embedding_dimension_and_l2(sub_emb):
    assert sub_emb.shape == (len(SUBSET), 2048) and sub_emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(sub_emb, axis=1), 1.0, atol=1e-6)
    assert len({tuple(v[:8]) for v in sub_emb}) == len(SUBSET)       # different images -> different vectors


def test_l2_normalize_handles_zero_vector():
    e = X.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(e[0], [0.6, 0.8]) and np.all(np.isfinite(e)) and np.allclose(e[1], 0)


def test_extraction_is_reproducible_bitwise(model, sub_emb):
    again = X.l2_normalize(X.extract(SUBSET, model))
    assert again.tobytes() == sub_emb.tobytes()


# ---- cosine / ranking ----
def test_cosine_equals_float64_reference(sub_emb):
    ids = [r["article_id"] for r in SUBSET]
    q = ids[0]
    got = {c: s for c, s in V.top_k(q, ids, sub_emb, k=len(ids) - 1)}
    e = sub_emb.astype(np.float64)
    for i, c in enumerate(ids[1:], 1):
        ref = float(e[0] @ e[i] / (np.linalg.norm(e[0]) * np.linalg.norm(e[i])))
        assert abs(got[c] - ref) < 1e-6


def test_topk_excludes_query_no_duplicates_sorted(sub_emb):
    ids = [r["article_id"] for r in SUBSET]
    for q in ids:
        res = V.top_k(q, ids, sub_emb, k=5)
        cs = [c for c, _ in res]
        assert q not in cs and len(cs) == len(set(cs)) == 5
        s = [x for _, x in res]
        assert s == sorted(s, reverse=True)


def test_ranking_tie_break_is_deterministic_hash_order():
    ids = ["q", "a", "b", "c"]
    v = np.array([[1, 0], [0, 1], [0, 1], [0.5, 0.5]], dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    q = np.array([[1, 0], [0, 1], [0, 1], [0, 1]], dtype=np.float32)   # a and b get an identical score
    res = [c for c, _ in V.top_k("q", ids, q, k=3)]
    assert res[:3] == sorted(["a", "b", "c"], key=lambda c: V.tie_key("q", c))  # all score 0 -> pure hash order
    p = np.array([[1, 0], [0.6, 0.8], [0.6, 0.8], [0, 1]], dtype=np.float32)
    r2 = V.top_k("q", ids, p, k=3)
    assert [c for c, _ in r2] == sorted(["a", "b"], key=lambda c: V.tie_key("q", c)) + ["c"]
    assert V.top_k("q", ids, p, k=3) == r2


# ---- cache invalidation ----
def test_cache_key_changes_with_every_input():
    base = X.cache_key("U", "W")
    assert base == X.cache_key("U", "W")
    assert base != X.cache_key("U2", "W")
    assert base != X.cache_key("U", "W2")
    spec = dict(X.PREPROCESSING_SPEC, version="pp-other")
    assert base != X.cache_key("U", "W", spec=spec)
    assert base != X.cache_key("U", "W", model_version="resnet50-other")


def test_stale_cache_is_not_reused(tmp_path):
    rows = UNI[:3]
    sub_csv = tmp_path / "u.csv"
    with open(sub_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    r1 = X.run(cache_dir=tmp_path / "c", universe_csv=sub_csv)
    assert r1["status"] == "recomputed"
    r2 = X.run(cache_dir=tmp_path / "c", universe_csv=sub_csv)
    assert r2["status"] == "cache_hit" and r2["emb"].tobytes() == r1["emb"].tobytes()
    meta = json.loads((r1["npy"].with_suffix(".json")).read_text())
    meta["cache_key"] = "stale"
    r1["npy"].with_suffix(".json").write_text(json.dumps(meta))
    assert X.run(cache_dir=tmp_path / "c", universe_csv=sub_csv)["status"] == "recomputed"
    r1["npy"].write_bytes(b"corrupt")                                  # tampered npy -> hash mismatch -> recompute
    assert X.run(cache_dir=tmp_path / "c", universe_csv=sub_csv)["status"] == "recomputed"


def test_example_query_rule_is_label_free():
    Q = list(csv.DictReader(open(OUT / "query_manifest.csv", encoding="utf-8")))
    sel = V.example_queries(Q)
    assert len(sel) == 12
    assert {q["product_group_name"] for q in sel} == {q["product_group_name"] for q in Q}
    assert all(sum(s["product_group_name"] == g for s in sel) == 2 for g in {q["product_group_name"] for q in Q})


# ---- output validation (skipped until the run exists) ----
needs_run = pytest.mark.skipif(not (OUT / "visual_top5.csv").exists(), reason="run not executed yet")


@needs_run
def test_outputs_complete_and_consistent():
    rows = list(csv.DictReader(open(OUT / "visual_top5.csv", encoding="utf-8")))
    Q = list(csv.DictReader(open(OUT / "query_manifest.csv", encoding="utf-8")))
    U = {r["article_id"]: r for r in UNI}
    assert len(rows) == 5 * len(Q) == 250
    by = {}
    for r in rows:
        by.setdefault(r["query_id"], []).append(r)
    assert set(by) == {q["article_id"] for q in Q}
    for q, lst in by.items():
        assert [int(x["rank"]) for x in lst] == [1, 2, 3, 4, 5]
        c = [x["candidate_id"] for x in lst]
        assert q not in c and len(set(c)) == 5 and all(x in U for x in c)
        assert U[q]["product_code"] not in {U[x]["product_code"] for x in c}      # no same-design variant
        s = [float(x["visual_score"]) for x in lst]
        assert s == sorted(s, reverse=True) and all(-1.0001 <= v <= 1.0001 for v in s)
        assert {x["model_version"] for x in lst} == {X.MODEL_VERSION} and {x["preprocessing_version"] for x in lst} == {X.PREPROCESSING_VERSION}


@needs_run
def test_outputs_scores_match_cached_embeddings_independently():
    man = json.loads((OUT / "embeddings_manifest.json").read_text())
    emb = np.load(ROOT / man["cache_file"])
    ids = man["article_ids_order"]
    assert emb.shape == (145, 2048) and X.sha_file(ROOT / man["cache_file"]) == man["cache_file_sha256"]
    e64 = emb.astype(np.float64)
    rows = list(csv.DictReader(open(OUT / "visual_top5.csv", encoding="utf-8")))
    for r in rows:
        i, j = ids.index(r["query_id"]), ids.index(r["candidate_id"])
        assert abs(float(r["visual_score"]) - float(e64[i] @ e64[j])) < 1e-6
    # independent full ranking per query (float64) must give the same Top-5 set unless scores tie within 1e-6
    for q in {r["query_id"] for r in rows}:
        i = ids.index(q)
        sc = e64 @ e64[i]; sc[i] = -9
        top = [ids[k] for k in np.argsort(-sc)[:5]]
        got = [r["candidate_id"] for r in rows if r["query_id"] == q]
        assert set(got) == set(top) or sorted(sc)[-6] - sorted(sc)[-5] < 1e-6


@needs_run
def test_json_and_log_counts():
    js = json.loads((OUT / "visual_top5.json").read_text())
    log = json.loads((OUT / "visual_run_log.json").read_text())
    assert len(js["queries"]) == log["counts"]["queries_run"] == 50
    assert log["counts"]["queries_with_full_top5"] == 50 and log["counts"]["images_excluded"] == 0
    assert js["candidate_pool_size_per_query"] == 144 and log["protocol_manifest_verified"] is True
    assert log["not_used"] and log["device"] == "cpu"
    assert len(list((OUT / "visual_examples").glob("*.jpg"))) == 12
