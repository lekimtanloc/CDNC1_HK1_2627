#!/usr/bin/env python3
"""Score same-item image retrieval on the frozen Week 2 test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from direction_a_repro import load_masked_image
from week3_retrieval_metrics import evaluate


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT.parent / "archive.zip")
    parser.add_argument("--protocol-dir", type=Path, default=ROOT.parent / "results/week2/FashionRecommender/data_protocol")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "results/week3/FashionRecommender")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    validation = json.loads((args.protocol_dir / "validation_report.json").read_text())
    if digest(args.archive) != validation["archive_sha256"]:
        raise ValueError("Archive differs from the frozen Week 2 dataset")
    for name in ("test.csv", "ground_truth_test.csv"):
        expected = next(row["sha256"] for row in validation["outputs"] if row["path"] == name)
        if digest(args.protocol_dir / name) != expected:
            raise ValueError(f"Frozen protocol file changed: {name}")

    rows = read_csv(args.protocol_dir / "test.csv")
    truths = read_csv(args.protocol_dir / "ground_truth_test.csv")
    queries = [row for row in rows if row["role"] == "query"]
    gallery = [row for row in rows if row["role"] == "gallery"]
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in truths:
        relevant[row["query_image_path"]].add(row["relevant_gallery_path"])
    if set(relevant) != {row["image_path"] for row in queries}:
        raise ValueError("Ground truth does not cover each query exactly")
    if len(queries) != validation["split_stats"]["test"]["queries"]:
        raise ValueError("Query count differs from the frozen Week 2 split")
    if len(gallery) != validation["split_stats"]["test"]["gallery_images"]:
        raise ValueError("Gallery count differs from the frozen Week 2 split")

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])
    started = time.monotonic()
    vectors = []
    all_rows = queries + gallery
    black_crop_paths = []
    with zipfile.ZipFile(args.archive) as archive:
        for offset in range(0, len(all_rows), args.batch_size):
            batch = all_rows[offset:offset + args.batch_size]
            tensors = []
            for row in batch:
                masked, _, _ = load_masked_image(archive, row["image_path"], row["mask_path"])
                cropped = transforms.CenterCrop(224)(transforms.Resize(256)(masked))
                if not np.asarray(cropped).any():
                    black_crop_paths.append(row["image_path"])
                tensors.append(preprocessing(cropped))
            with torch.inference_mode():
                vectors.append(model(torch.stack(tensors)).cpu())
            done = min(offset + len(batch), len(all_rows))
            if done % 200 < args.batch_size or done == len(all_rows):
                print(f"features: {done}/{len(all_rows)}", flush=True)
    embeddings = F.normalize(torch.cat(vectors, dim=0), dim=1)
    query_count = len(queries)
    similarities = (embeddings[:query_count] @ embeddings[query_count:].T).numpy()
    gallery_paths = [row["image_path"] for row in gallery]
    gallery_items = [row["item_id"] for row in gallery]
    rankings: dict[str, list[str]] = {}
    topk_rows = []
    top1_hits = 0
    for index, query in enumerate(queries):
        positions = np.argsort(-similarities[index], kind="stable")[:args.top_k]
        ranked = [gallery_paths[int(pos)] for pos in positions]
        rankings[query["image_path"]] = ranked
        top1_hits += int(ranked[0] in relevant[query["image_path"]])
        for rank, pos in enumerate(positions, 1):
            topk_rows.append({
                "query_image_path": query["image_path"],
                "query_item_id": query["item_id"],
                "rank": rank,
                "gallery_image_path": gallery_paths[int(pos)],
                "gallery_item_id": gallery_items[int(pos)],
                "cosine_similarity": round(float(similarities[index, int(pos)]), 6),
                "same_item": int(gallery_paths[int(pos)] in relevant[query["image_path"]]),
            })
    metrics = evaluate(rankings, relevant, args.top_k)
    report = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Week 2 frozen test, query image vs other-view gallery image from same item",
        "archive_sha256": validation["archive_sha256"],
        "test_csv_sha256": digest(args.protocol_dir / "test.csv"),
        "ground_truth_test_csv_sha256": digest(args.protocol_dir / "ground_truth_test.csv"),
        "model": "ResNet-50 pretrained IMAGENET1K_V2, masked garment, Resize(256), CenterCrop(224), ImageNet normalization",
        "training": "none; pretrained weights used only for feature extraction",
        "queries": query_count,
        "gallery_images": len(gallery),
        "ground_truth_pairs": len(truths),
        "top_k": args.top_k,
        "top1_accuracy": top1_hits / query_count,
        "recall_at_5": metrics["recall"],
        "ndcg_at_5": metrics["ndcg"],
        "mrr_at_5": metrics["reciprocal_rank"],
        "all_black_center_crops": len(black_crop_paths),
        "all_black_center_crop_examples": black_crop_paths[:10],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "This is image-to-image retrieval on the DeepFashion test split, not ranking on the website's H&M catalog.",
            "The existing CenterCrop preprocessing is retained; fully black crops are counted and may lower retrieval scores.",
            "Popularity and category fusion are not used in this visual-retrieval test.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "ket_qua_test_nhieu_goc_nhin.json"
    ranking_path = args.output_dir / "xep_hang_test_top5.csv"
    if result_path.exists() or ranking_path.exists():
        raise FileExistsError("Refusing to overwrite an existing test evaluation")
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(topk_rows[0]))
        writer.writeheader()
        writer.writerows(topk_rows)
    report["top5_csv_sha256"] = digest(ranking_path)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "metrics": {k: report[k] for k in ("top1_accuracy", "recall_at_5", "ndcg_at_5", "mrr_at_5", "all_black_center_crops")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
