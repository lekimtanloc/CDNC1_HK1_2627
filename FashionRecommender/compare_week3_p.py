#!/usr/bin/env python3
"""Compare Week 3 Top-5 with the paper's main P weights on identical inputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from week3_verify_and_smoke import category_score, digest, read_csv


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "repro_outputs_resnet50_full")
    parser.add_argument("--week3-dir", type=Path, default=ROOT.parent / "results/week3/FashionRecommender")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=10)
    args = parser.parse_args()

    manifest = json.loads((args.cache_dir / "run_manifest.json").read_text())
    for name in ("features_resnet50.pt", "metadata.csv", "purchases.csv"):
        expected = next(row["sha256"] for row in manifest["outputs"] if row["path"] == name)
        if digest(args.cache_dir / name) != expected:
            raise ValueError(f"Cached input changed: {name}")
    features = torch.load(args.cache_dir / "features_resnet50.pt", map_location="cpu", weights_only=True)
    metadata = read_csv(args.cache_dir / "metadata.csv")
    purchases = read_csv(args.cache_dir / "purchases.csv")
    if len(features) != len(metadata) or args.top_k >= len(metadata):
        raise ValueError("Feature/metadata length or Top-K invalid")

    week3 = json.loads((args.week3_dir / "ket_qua_kiem_thu.json").read_text())
    baseline_rows = read_csv(args.week3_dir / "ket_qua_goi_y_top_k.csv")
    if week3["seed"] != args.seed or week3["top_k"] != args.top_k or week3["queries"] != args.max_queries:
        raise ValueError("Requested inputs differ from the recorded Week 3 run")
    if week3["source_cache_manifest_sha256"] != digest(args.cache_dir / "run_manifest.json"):
        raise ValueError("Cache differs from the recorded Week 3 run")
    if week3["topk_csv_sha256"] != digest(args.week3_dir / "ket_qua_goi_y_top_k.csv"):
        raise ValueError("Recorded Week 3 Top-K CSV changed")

    rng = np.random.default_rng(args.seed)
    popularity = rng.integers(0, 101, len(metadata)).astype(float)
    popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    trendiness = {int(row["user_id"]): float(rng.random()) for row in purchases}
    blocked: dict[int, set[int]] = defaultdict(set)
    for row in purchases:
        blocked[int(row["user_id"])].add(int(row["item_index"]))

    queries = [(int(row["user_id"]), int(row["item_index"])) for row in purchases[:args.max_queries]]
    if len(baseline_rows) != len(queries) * args.top_k:
        raise ValueError("Week 3 Top-K row count differs from query count")
    comparisons = []
    all_baseline = []
    all_p = []
    p_rows = []
    for query_number, (user_id, query_index) in enumerate(queries):
        visual = (features @ features[query_index]).numpy()
        category = np.array([
            category_score(metadata[query_index]["category"], row["category"])
            for row in metadata
        ])
        distance = np.abs(popularity - trendiness[user_id])
        baseline_score = .6 * visual + .1 * category + .3 * (1 - distance ** 4)
        # Section 4.2 of the paper states w_v=.7, w_p=.3, gamma=2.
        # It gives no numeric category weight for this main configuration.
        p_score = .7 * visual + .3 * (1 - distance ** 2)
        baseline_score[list(blocked[user_id])] = -np.inf
        p_score[list(blocked[user_id])] = -np.inf
        baseline_indices = np.argsort(-baseline_score, kind="stable")[:args.top_k]
        p_indices = np.argsort(-p_score, kind="stable")[:args.top_k]
        recorded = baseline_rows[query_number * args.top_k:(query_number + 1) * args.top_k]
        baseline_ids = [metadata[index]["item_id"] for index in baseline_indices]
        if baseline_ids != [row["candidate_item"] for row in recorded]:
            raise ValueError(f"Recomputed Week 3 ranking differs at query {query_number + 1}")
        p_ids = [metadata[index]["item_id"] for index in p_indices]
        overlap = len(set(baseline_ids) & set(p_ids))
        comparisons.append({
            "query_number": query_number + 1,
            "user_id": user_id,
            "query_item": metadata[query_index]["item_id"],
            "week3_top5": baseline_ids,
            "p_top5": p_ids,
            "common_items": overlap,
            "same_top1": baseline_ids[0] == p_ids[0],
            "same_order": baseline_ids == p_ids,
        })
        for rank, index in enumerate(p_indices, 1):
            p_rows.append({
                "user_id": user_id,
                "query_item": metadata[query_index]["item_id"],
                "rank": rank,
                "candidate_item": metadata[index]["item_id"],
                "visual_score": round(float(visual[index]), 6),
                "popularity_score": round(float(1 - distance[index] ** 2), 6),
                "final_score": round(float(p_score[index]), 6),
            })
        all_baseline.extend(int(index) for index in baseline_indices)
        all_p.extend(int(index) for index in p_indices)

    output = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": "same cached embeddings, synthetic popularity/trendiness, queries and candidates as Week 3 smoke run",
        "paper_reference": "https://arxiv.org/pdf/2506.07773 section 4.2",
        "week3_configuration": {"visual": .6, "category": .1, "popularity": .3, "popularity_exponent": 4},
        "p_configuration": {"visual": .7, "category": 0.0, "popularity": .3, "popularity_exponent": 2},
        "p_category_note": "The paper's main numeric configuration does not specify a category weight; P comparison uses only the two stated weights.",
        "queries": len(queries),
        "candidate_items": len(metadata),
        "top_k": args.top_k,
        "seed": args.seed,
        "source_cache_manifest_sha256": digest(args.cache_dir / "run_manifest.json"),
        "week3_topk_csv_sha256": digest(args.week3_dir / "ket_qua_goi_y_top_k.csv"),
        "summary": {
            "total_common_top5_items": sum(row["common_items"] for row in comparisons),
            "total_top5_slots": len(queries) * args.top_k,
            "queries_same_top1": sum(row["same_top1"] for row in comparisons),
            "queries_same_top5_order": sum(row["same_order"] for row in comparisons),
            "week3_category_exact_match": round(float(np.mean([
                metadata[index]["category"] == metadata[int(queries[n // args.top_k][1])]["category"]
                for n, index in enumerate(all_baseline)
            ])), 6),
            "p_category_exact_match": round(float(np.mean([
                metadata[index]["category"] == metadata[int(queries[n // args.top_k][1])]["category"]
                for n, index in enumerate(all_p)
            ])), 6),
            "week3_popularity_mae": round(float(np.mean([
                abs(popularity[index] - trendiness[queries[n // args.top_k][0]])
                for n, index in enumerate(all_baseline)
            ])), 6),
            "p_popularity_mae": round(float(np.mean([
                abs(popularity[index] - trendiness[queries[n // args.top_k][0]])
                for n, index in enumerate(all_p)
            ])), 6),
        },
        "per_query": comparisons,
        "p_topk_rows": p_rows,
        "limitations": [
            "This compares ranking on 10 cached queries, not the paper's full experimental results.",
            "The adjusted archive has no gender labels; gender filtering and alignment are not evaluated.",
            "Popularity, trendiness and purchases are synthetic, not real website events.",
            "The cache has one embedding per product, not one per view; this is not a multi-view retrieval test.",
        ],
    }
    path = args.week3_dir / "doi_chieu_cau_hinh_P.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "summary": output["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
