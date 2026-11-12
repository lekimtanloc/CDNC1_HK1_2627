#!/usr/bin/env python3
"""Audit the frozen retrieval protocol and smoke-test trend-aware ranking.

This is an integration check, not an evaluation of user preference. The cached
ResNet-50 vectors have one masked image per item, so they cannot evaluate the
multi-view retrieval ground truth in the frozen split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "valid", "test")
GROUPS = {
    "Tees_Tanks": "A", "Jackets_Coats": "C", "Pants": "B",
    "Leggings": "B", "Skirts": "D", "Dresses": "D",
}
GROUP_SCORES = {
    ("A", "B"): .2, ("A", "C"): .3, ("A", "D"): .1,
    ("B", "C"): .1, ("B", "D"): .1, ("C", "D"): .1,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit(protocol_dir: Path, archive: Path) -> dict:
    report = json.loads((protocol_dir / "validation_report.json").read_text())
    errors: list[str] = []
    checks: dict[str, object] = {}
    checks["archive_sha256_matches"] = digest(archive) == report["archive_sha256"]
    if not checks["archive_sha256_matches"]:
        errors.append("Archive SHA-256 differs from frozen data build")
    bad_artifacts = [
        entry["path"] for entry in report["outputs"]
        if not (protocol_dir / entry["path"]).is_file()
        or digest(protocol_dir / entry["path"]) != entry["sha256"]
    ]
    checks["artifact_hashes_match"] = not bad_artifacts
    if bad_artifacts:
        errors.append(f"Artifact hash mismatch: {bad_artifacts}")

    all_items: set[str] = set()
    all_images: set[str] = set()
    split_counts = {}
    for split in SPLITS:
        rows = read_csv(protocol_dir / f"{split}.csv")
        truths = read_csv(protocol_dir / f"ground_truth_{split}.csv")
        items = {row["item_id"] for row in rows}
        images = {row["image_path"] for row in rows}
        if len(images) != len(rows) or items & all_items or images & all_images:
            errors.append(f"Duplicate image or leaked item in {split}")
        all_items.update(items)
        all_images.update(images)
        by_item: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_item[row["item_id"]].append(row)
        expected = {
            (query["image_path"], gallery["image_path"], item_id)
            for item_id, views in by_item.items()
            for query in views if query["role"] == "query"
            for gallery in views if gallery["role"] == "gallery"
        }
        observed = {
            (row["query_image_path"], row["relevant_gallery_path"], row["item_id"])
            for row in truths
        }
        if observed != expected or len(observed) != len(truths):
            errors.append(f"Ground truth incomplete or inconsistent in {split}")
        if any(query == gallery for query, gallery, _ in observed):
            errors.append(f"Self match in {split}")
        counts = {
            "items": len(items), "images": len(rows),
            "queries": sum(row["role"] == "query" for row in rows),
            "gallery_images": sum(row["role"] == "gallery" for row in rows),
            "ground_truth_pairs": len(truths),
        }
        split_counts[split] = counts
        if any(report["split_stats"][split][key] != value for key, value in counts.items()):
            errors.append(f"Count differs from validation report in {split}")
    checks["split_counts"] = split_counts
    checks["no_item_leakage"] = not any("leaked item" in e for e in errors)
    checks["ground_truth_exact"] = not any("Ground truth" in e for e in errors)

    with sqlite3.connect(protocol_dir / report["tracking_database"]) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        db_items = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        db_images = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        db_truths = connection.execute("SELECT COUNT(*) FROM retrieval_ground_truth").fetchone()[0]
    checks["sqlite_integrity"] = integrity
    checks["sqlite_foreign_key_errors"] = len(foreign_keys)
    checks["sqlite_counts"] = {"items": db_items, "images": db_images, "ground_truth_pairs": db_truths}
    if integrity != "ok" or foreign_keys or db_items != len(all_items) or db_images != len(all_images) or db_truths != sum(x["ground_truth_pairs"] for x in split_counts.values()):
        errors.append("SQLite lineage does not match CSVs")
    if report["status"] != "passed" or report["errors"]:
        errors.append("Original validation report did not pass")
    return {"status": "passed" if not errors else "failed", "build_id": report["build_id"], "checks": checks, "errors": errors}


def category_score(query: str, candidate: str) -> float:
    left, right = GROUPS.get(query), GROUPS.get(candidate)
    if left is None or right is None:
        return 0.0
    if left == right:
        return 1.0 if query == candidate else .8
    return GROUP_SCORES.get(tuple(sorted((left, right))), 0.0)


def smoke(cache_dir: Path, seed: int, top_k: int, max_queries: int) -> dict:
    manifest = json.loads((cache_dir / "run_manifest.json").read_text())
    required = ["features_resnet50.pt", "metadata.csv", "purchases.csv"]
    stale = [row["path"] for row in manifest["outputs"] if row["path"] in required and digest(cache_dir / row["path"]) != row["sha256"]]
    if stale:
        raise ValueError(f"Cached input checksum mismatch: {stale}")
    features = torch.load(cache_dir / "features_resnet50.pt", map_location="cpu", weights_only=True)
    metadata = read_csv(cache_dir / "metadata.csv")
    purchases = read_csv(cache_dir / "purchases.csv")
    if len(features) != len(metadata) or top_k >= len(metadata):
        raise ValueError("Feature/metadata length or Top-K invalid")
    rng = np.random.default_rng(seed)
    popularity = rng.integers(0, 101, len(metadata)).astype(float)
    popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    trendiness = {int(row["user_id"]): float(rng.random()) for row in purchases}
    blocked: dict[int, set[int]] = defaultdict(set)
    for row in purchases:
        blocked[int(row["user_id"])].add(int(row["item_index"]))
    queries = [(int(row["user_id"]), int(row["item_index"])) for row in purchases[:max_queries]]
    if not queries:
        raise ValueError("No cached queries")
    weights = {"visual": .6, "category": .1, "popularity": .3}
    output = []
    for user_id, query_index in queries:
        visual = (features @ features[query_index]).numpy()
        category = np.array([category_score(metadata[query_index]["category"], row["category"]) for row in metadata])
        pop = 1 - np.abs(popularity - trendiness[user_id]) ** 4
        score = weights["visual"] * visual + weights["category"] * category + weights["popularity"] * pop
        score[list(blocked[user_id])] = -np.inf
        ranked = np.argsort(-score, kind="stable")[:top_k]
        for rank, index in enumerate(ranked, 1):
            output.append({
                "user_id": user_id, "query_item": metadata[query_index]["item_id"],
                "rank": rank, "candidate_item": metadata[index]["item_id"],
                "visual_score": round(float(visual[index]), 6),
                "category_score": round(float(category[index]), 6),
                "popularity_score": round(float(pop[index]), 6),
                "final_score": round(float(score[index]), 6),
            })
    return {
        "status": "passed", "label": "P-style integration smoke, illustrative category weight",
        "queries": len(queries), "top_k": top_k, "candidate_items": len(metadata),
        "backbone": manifest["backbone"], "weights": manifest["weights"],
        "masked_embeddings": True, "seed": seed, "weights_fusion": weights,
        "popularity_source": "new deterministic synthetic values; independent of retrieval labels",
        "gender_score": "omitted: no gender labels in adjusted archive",
        "source_cache_manifest_sha256": digest(cache_dir / "run_manifest.json"),
        "mean_topk_scores": {
            key: round(sum(row[key] for row in output) / len(output), 6)
            for key in ("visual_score", "category_score", "popularity_score", "final_score")
        },
        "rows": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT.parent / "archive.zip")
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=ROOT.parent / "results/week2/FashionRecommender/data_protocol",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "repro_outputs_resnet50_full")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "results/week3/FashionRecommender",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=10)
    args = parser.parse_args()
    audit_result = audit(args.protocol_dir, args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_report = args.output_dir / "bao_cao_kiem_tra_du_lieu.json"
    audit_report.write_text(json.dumps(audit_result, indent=2) + "\n")
    if audit_result["status"] != "passed":
        raise SystemExit(f"Data audit failed; see {audit_report.name}")
    smoke_result = smoke(args.cache_dir, args.seed, args.top_k, args.max_queries)
    rows = smoke_result.pop("rows")
    topk_output = args.output_dir / "ket_qua_goi_y_top_k.csv"
    with topk_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    smoke_result["topk_csv_sha256"] = digest(topk_output)
    smoke_result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "ket_qua_kiem_thu.json").write_text(
        json.dumps(smoke_result, indent=2) + "\n"
    )
    print(json.dumps({"audit": audit_result["status"], "smoke": smoke_result}, indent=2))


if __name__ == "__main__":
    main()
