"""Week 7: Top-5 visual retrieval (V-R50-FULLPAD) over the frozen H&M candidate pool.

One command from the repo root:
    python3 -I -B week7/src/visual_retrieval.py
Produces visual_top5.csv/.json, embeddings_manifest.json, visual_run_log.json, visual_examples/.
Visual scores only; no popularity, no metadata, no relevance judgments.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import extract_embeddings as X  # noqa: E402

ROOT, OUT = X.ROOT, X.OUT
SEED, K = 42, 5


def tie_key(q: str, c: str) -> str:
    return hashlib.sha256(f"{SEED}:{q}:{c}".encode()).hexdigest()


def top_k(query_id: str, ids: list[str], emb: np.ndarray, k: int = K) -> list[tuple[str, float]]:
    """Cosine = dot of L2-normalised float32 vectors; descending by exact float32 score;
    ties -> ascending sha256('42:'+query+':'+candidate); the query itself is removed first."""
    qi = ids.index(query_id)
    scores = (emb @ emb[qi]).astype(np.float32)
    cand = [(ids[i], np.float32(scores[i])) for i in range(len(ids)) if i != qi]
    cand.sort(key=lambda t: (-float(t[1]), tie_key(query_id, t[0])))
    return [(c, float(s)) for c, s in cand[:k]]


def example_queries(queries: list[dict]) -> list[dict]:
    """Fixed rule (declared before looking at results): per product group, the 2 queries with the smallest
    sha256('42:example:'+article_id). Not selected by score or by how good the list looks."""
    by = {}
    for q in queries:
        by.setdefault(q["product_group_name"], []).append(q)
    sel = []
    for g in sorted(by):
        sel += sorted(by[g], key=lambda q: hashlib.sha256(f"42:example:{q['article_id']}".encode()).hexdigest())[:2]
    return sel


def render_examples(sel, res, rows_by_id, out_dir: Path):
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    t = 180
    for q in sel:
        qid = q["article_id"]
        items = [("QUERY", qid, None)] + [(f"rank {i + 1}", c, s) for i, (c, s) in enumerate(res[qid])]
        sheet = Image.new("RGB", (t * len(items), t + 34), "white")
        d = ImageDraw.Draw(sheet)
        for i, (lab, a, s) in enumerate(items):
            with Image.open(ROOT / rows_by_id[a]["image_relpath"]) as im:
                im = im.convert("RGB"); im.thumbnail((t, t))
                sheet.paste(im, (i * t + (t - im.width) // 2, 34 + (t - im.height) // 2))
            d.text((i * t + 4, 2), f"{lab}  {a}", fill="black")
            d.text((i * t + 4, 16), f"{rows_by_id[a]['product_type_name']}" + (f"  cos={s:.4f}" if s is not None else ""), fill="gray")
        sheet.save(out_dir / f"{q['product_group_name'].replace(' ', '_')}_{qid}.jpg", quality=88)


def main() -> None:
    t_start = time.time()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    man = X.verify_protocol_manifest()
    r = X.run()
    rows, emb = r["rows"], r["emb"]
    ids = [x["article_id"] for x in rows]
    by_id = {x["article_id"]: x for x in rows}
    queries = list(csv.DictReader(open(OUT / "query_manifest.csv", newline="", encoding="utf-8")))
    assert emb.shape == (len(ids), X.EMBED_DIM) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)

    t0 = time.time()
    res, run_queries, skipped = {}, [], []
    for q in queries:
        if q["article_id"] not in by_id:
            skipped.append({"query": q["article_id"], "reason": "query image invalid/excluded"})
            continue
        res[q["article_id"]] = top_k(q["article_id"], ids, emb)
        run_queries.append(q)
    t_rank = time.time() - t0

    rows_csv = []
    for q in run_queries:
        for rk, (c, s) in enumerate(res[q["article_id"]], 1):
            rows_csv.append({"query_id": q["article_id"], "candidate_id": c, "rank": rk, "visual_score": f"{s:.8f}",
                             "model_version": X.MODEL_VERSION, "preprocessing_version": X.PREPROCESSING_VERSION,
                             "query_product_code": q["product_code"], "candidate_product_code": by_id[c]["product_code"],
                             "query_group": q["product_group_name"], "candidate_group": by_id[c]["product_group_name"],
                             "candidate_type": by_id[c]["product_type_name"]})
    with open(OUT / "visual_top5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)

    pool = len(ids) - 1
    js = {"method": "V-R50-FULLPAD", "model_version": X.MODEL_VERSION, "preprocessing_version": X.PREPROCESSING_VERSION,
          "protocol": "W7-HM-SIMILAR-PRODUCTS-v1", "candidate_pool_size_per_query": pool,
          "candidate_ids_valid": ids, "candidates_excluded_invalid_image": r["excluded"],
          "queries": [{"query_id": q["article_id"], "query_product_code": q["product_code"], "product_group": q["product_group_name"],
                       "excluded_from_pool": [{"article_id": q["article_id"], "reason": "query itself"}],
                       "pool_size": pool,
                       "top5": [{"rank": i + 1, "candidate_id": c, "visual_score": s} for i, (c, s) in enumerate(res[q["article_id"]])]}
                      for q in run_queries], "queries_skipped": skipped}
    (OUT / "visual_top5.json").write_text(json.dumps(js, indent=1))

    sel = example_queries(run_queries)
    ex_dir = OUT / "visual_examples"
    for old in ex_dir.glob("*.jpg"):
        old.unlink()
    render_examples(sel, res, by_id, ex_dir)

    import torch
    (OUT / "embeddings_manifest.json").write_text(json.dumps({
        "model_version": X.MODEL_VERSION, "weights_file": str(X.WEIGHTS_FILE.name), "weights_sha256": r["weights_sha256"],
        "preprocessing_spec": X.PREPROCESSING_SPEC, "embedding_dim": X.EMBED_DIM, "dtype": "float32", "l2_normalized": True,
        "n_embeddings": int(emb.shape[0]), "article_ids_order": ids,
        "dataset": {"universe_csv": "week7/outputs/hm/product_universe.csv", "universe_sha256": r["universe_sha256"]},
        "cache_key": r["key"], "cache_file": str(r["npy"].relative_to(ROOT)), "cache_file_sha256": X.sha_file(r["npy"]),
        "cache_status_this_run": r["status"], "excluded_invalid": r["excluded"]}, indent=1))

    complete = sum(1 for q in run_queries if len(res[q["article_id"]]) == K and len({c for c, _ in res[q["article_id"]]}) == K)
    (OUT / "visual_run_log.json").write_text(json.dumps({
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_manifest_frozen_utc": man["frozen_utc"], "protocol_manifest_verified": True,
        "device": "cpu", "torch_threads": 4, "deterministic_algorithms": True, "seed": 42, "batch_size": 1,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "pillow": __import__("PIL").__version__, "torchvision": __import__("torchvision").__version__,
                     "platform": platform.platform()},
        "counts": {"images_input": r["n_input"], "images_valid": len(ids), "images_excluded": len(r["excluded"]),
                   "embeddings": int(emb.shape[0]), "queries_in_manifest": len(queries), "queries_run": len(run_queries),
                   "queries_with_full_top5": complete, "queries_skipped": len(skipped), "top5_rows": len(rows_csv),
                   "example_images": len(sel)},
        "cache": {"status": r["status"], "key": r["key"]},
        "timing_s": {"embedding_extraction": round(r["t_extract_s"], 2), "load_validate_total": round(r["t_total_s"], 2),
                     "ranking": round(t_rank, 4), "wall_total": round(time.time() - t_start, 2)},
        "not_used": ["popularity", "metadata", "relevance labels", "fine-tuning"],
        "outputs_sha256": {n: X.sha_file(OUT / n) for n in ("visual_top5.csv", "visual_top5.json")}}, indent=1))
    print(json.dumps({"images_input": r["n_input"], "valid": len(ids), "excluded": len(r["excluded"]), "embeddings": int(emb.shape[0]),
                      "queries_run": len(run_queries), "full_top5": complete, "cache": r["status"],
                      "t_extract_s": round(r["t_extract_s"], 1)}, indent=1))


if __name__ == "__main__":
    main()
