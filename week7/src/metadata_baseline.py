"""Week 7: metadata baseline M-ATTR-TFIDF over the frozen H&M candidate pool.

Implements, without any change, week7/config/experiment.json -> methods.M-ATTR-TFIDF:
fixed a-priori weights, missing = no credit (no renormalisation), TF-IDF fitted on detail_desc of the 145
representatives, rounding to 9 digits, sha256 tie-break identical to the visual method, query excluded.
Uses no CNN output, no visual score and no relevance label.

Run (repo root):  python3 -I -B week7/src/metadata_baseline.py
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_embeddings as X  # noqa: E402  (manifest check + paths only; no torch import)
import protocol_lib as L  # noqa: E402

ROOT, OUT = X.ROOT, X.OUT
EXP = json.loads((ROOT / "week7/config/experiment.json").read_text())
CFG = EXP["methods"]["M-ATTR-TFIDF"]
WEIGHTS = CFG["weights"]
K = 5
SEED = EXP["seed"]
METHOD_VERSION = "M-ATTR-TFIDF-v1"


def load_universe_with_desc() -> list[dict]:
    arts = {}
    with open(ROOT / EXP["data"]["articles_csv"], encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            arts[r["article_id"]] = r
    rows = X.load_universe()
    for r in rows:
        a = arts[r["article_id"]]
        assert a["product_code"] == r["product_code"]
        r["detail_desc"] = a["detail_desc"]
        r["colour_group_name"] = a["colour_group_name"]
        r["graphical_appearance_name"] = a["graphical_appearance_name"]
    return rows


def score_all(query: dict, rows: list[dict], tfidf: dict) -> list[dict]:
    out = []
    for c in rows:
        if c["article_id"] == query["article_id"]:
            continue
        matched = [f for f in WEIGHTS if f != "detail_desc_tfidf"
                   and not L.is_missing(query.get(f)) and not L.is_missing(c.get(f)) and query[f] == c[f]]
        cos = L.cosine_sparse(tfidf[query["article_id"]], tfidf[c["article_id"]])
        s = sum(WEIGHTS[f] for f in matched) + WEIGHTS["detail_desc_tfidf"] * cos
        out.append({"cand": c, "score": s, "matched": matched, "tfidf_cos": cos})
    return out


def compute(query_ids: list[str], rows: list[dict] | None = None) -> dict:
    """Return {query_id: [top5 dicts]} plus tie diagnostics. Pool = all universe products except the query."""
    rows = rows or load_universe_with_desc()
    by_id = {r["article_id"]: r for r in rows}
    tfidf = L.tfidf_vectors({r["article_id"]: r["detail_desc"] for r in rows})
    res, ties = {}, {}
    for q in query_ids:
        scored = score_all(by_id[q], rows, tfidf)
        for s in scored:
            s["score_r"] = round(s["score"], 9)
        scored.sort(key=lambda s: (-s["score_r"], L.tie_key(q, s["cand"]["article_id"], SEED)))
        res[q] = scored[:K]
        ties[q] = {"pool": len(scored), "tied_with_rank5": sum(1 for s in scored if s["score_r"] == scored[K - 1]["score_r"]),
                   "boundary_tie": scored[K - 1]["score_r"] == scored[K]["score_r"]}
    return {"top": res, "ties": ties, "rows": rows, "by_id": by_id}


FIELDS = ["query_id", "candidate_id", "rank", "metadata_score", "method_version", "query_product_code", "candidate_product_code",
          "query_group", "candidate_group", "candidate_type", "candidate_colour", "matched_fields", "tfidf_cosine"]


def to_rows(query_rows: list[dict], res: dict, by_id: dict) -> list[dict]:
    out = []
    for q in query_rows:
        for i, s in enumerate(res[q["article_id"]], 1):
            c = s["cand"]
            out.append({"query_id": q["article_id"], "candidate_id": c["article_id"], "rank": i,
                        "metadata_score": f"{s['score_r']:.9f}", "method_version": METHOD_VERSION,
                        "query_product_code": q["product_code"], "candidate_product_code": c["product_code"],
                        "query_group": by_id[q["article_id"]]["product_group_name"], "candidate_group": c["product_group_name"],
                        "candidate_type": c["product_type_name"], "candidate_colour": c["colour_group_name"],
                        "matched_fields": "|".join(s["matched"]), "tfidf_cosine": f"{s['tfidf_cos']:.6f}"})
    return out


def main() -> None:
    t0 = time.time()
    man = X.verify_protocol_manifest()
    queries = list(csv.DictReader(open(OUT / "query_manifest.csv", newline="", encoding="utf-8")))
    r = compute([q["article_id"] for q in queries])
    rows, by_id = r["rows"], r["by_id"]
    out_rows = to_rows(queries, r["top"], by_id)
    with open(OUT / "metadata_top5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(out_rows)

    # ---- comparison with the visual run (descriptive; NOT a quality measure) ----
    vis = {}
    for x in csv.DictReader(open(OUT / "visual_top5.csv", newline="", encoding="utf-8")):
        vis.setdefault(x["query_id"], []).append(x["candidate_id"])
    vjs = json.loads((OUT / "visual_top5.json").read_text())
    visual_pool = set(vjs["candidate_ids_valid"])
    meta_pool = {c["article_id"] for c in rows}
    comp = []
    for q in queries:
        qid = q["article_id"]
        m = [s["cand"]["article_id"] for s in r["top"][qid]]
        v = vis[qid]
        pool_v = visual_pool - {qid}
        pool_m = {c["article_id"] for c in rows} - {qid}
        comp.append({"query_id": qid, "query_group": q["product_group_name"], "pool_visual": len(pool_v), "pool_metadata": len(pool_m),
                     "pools_identical": pool_v == pool_m, "visual_top5_in_pool": all(c in pool_v for c in v),
                     "metadata_top5_in_pool": all(c in pool_m for c in m), "query_in_visual_top5": qid in v, "query_in_metadata_top5": qid in m,
                     "overlap_count": len(set(v) & set(m)), "jaccard": round(len(set(v) & set(m)) / len(set(v) | set(m)), 4),
                     "top1_same": v[0] == m[0], "union_size": len(set(v) | set(m)),
                     "metadata_boundary_tie_rank5_6": r["ties"][qid]["boundary_tie"], "metadata_tied_with_rank5": r["ties"][qid]["tied_with_rank5"],
                     "visual_top5": "|".join(v), "metadata_top5": "|".join(m)})
    with open(OUT / "method_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0])); w.writeheader(); w.writerows(comp)

    full = sum(1 for q in queries if len({s["cand"]["article_id"] for s in r["top"][q["article_id"]]}) == K)
    import numpy as np
    (OUT / "metadata_run_log.json").write_text(json.dumps({
        "method": "M-ATTR-TFIDF", "method_version": METHOD_VERSION, "protocol": EXP["experiment_id"],
        "started_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec="seconds"),
        "protocol_manifest_frozen_utc": man["frozen_utc"], "protocol_manifest_verified": True,
        "weights": WEIGHTS, "weights_source": "week7/config/experiment.json (fixed a priori; not tuned; no visual/label information used)",
        "universe_products": len(rows), "candidate_pool_per_query": len(rows) - 1,
        "universe_with_missing_values": {f: sum(L.is_missing(x[f]) for x in rows) for f in list(WEIGHTS) if f != "detail_desc_tfidf"} | {"detail_desc": sum(L.is_missing(x["detail_desc"]) for x in rows)},
        "counts": {"queries": len(queries), "queries_with_full_top5": full, "top5_rows": len(out_rows),
                   "duplicate_candidate_in_query": sum(1 for q in queries if len({s["cand"]["article_id"] for s in r["top"][q["article_id"]]}) != K),
                   "queries_boundary_tie_rank5_6": sum(1 for q in queries if r["ties"][q["article_id"]]["boundary_tie"]),
                   "pools_identical_all_queries": all(c["pools_identical"] for c in comp)},
        "tie_break": "sha256('42:'+query_article_id+':'+candidate_article_id) ascending (same function as visual)",
        "not_used": ["CNN embeddings", "visual scores", "popularity", "relevance labels", "weight tuning"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "wall_s": round(time.time() - t0, 3), "output_sha256": {"metadata_top5.csv": X.sha_file(OUT / "metadata_top5.csv"),
                                                              "method_comparison.csv": X.sha_file(OUT / "method_comparison.csv")}}, indent=1))
    print(json.dumps({"queries": len(queries), "full_top5": full, "rows": len(out_rows),
                      "boundary_ties": sum(c["metadata_boundary_tie_rank5_6"] for c in comp),
                      "pools_identical": all(c["pools_identical"] for c in comp),
                      "overlap_mean": round(sum(c["overlap_count"] for c in comp) / len(comp), 3)}, indent=1))


if __name__ == "__main__":
    main()
