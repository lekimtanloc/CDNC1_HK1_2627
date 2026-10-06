#!/usr/bin/env python3
"""Bước 3 — Tạo ground truth độc lập (Tuần 3).

Ground truth được xây dựng dựa trên **category tương đồng** (category-based relevance),
KHÔNG dùng popularity rule, KHÔNG dùng hành vi tổng hợp.

Nguyên tắc:
  - Với mỗi item trong tập test, tất cả item trong tập train có cùng category
    được coi là relevant (binary relevance = 1), khác category = 0.
  - Đây là ground truth tối giản nhưng độc lập và không circular.
  - Dùng để tính Recall@K, NDCG@K, MRR trên kết quả retrieval của pipeline A.

Đầu ra:
  ground_truth.json  — dict: {test_item_id: {relevant_item_id: 1, ...}}
  gt_manifest.json   — metadata: số query, avg relevant per query, category coverage

Chạy:
    python make_ground_truth.py --split-dir . --output-dir .
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_split(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo ground truth độc lập từ split")
    parser.add_argument("--split-dir", type=Path, default=Path("."),
                        help="Thư mục chứa split_train.json, split_test.json")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    train_path = args.split_dir / "split_train.json"
    test_path = args.split_dir / "split_test.json"
    for p in (train_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy: {p} — hãy chạy split_items.py trước")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_items = load_split(train_path)
    test_items = load_split(test_path)

    # Index train items theo category
    train_by_cat: dict[str, list[str]] = defaultdict(list)
    for item in train_items:
        train_by_cat[item["category"]].append(item["item_id"])

    # Tạo ground truth
    ground_truth: dict[str, dict[str, int]] = {}
    stats_relevant_counts = []

    for item in test_items:
        qid = item["item_id"]
        cat = item["category"]
        relevant = {rid: 1 for rid in train_by_cat.get(cat, [])}
        ground_truth[qid] = relevant
        stats_relevant_counts.append(len(relevant))

    # Tính metrics summary
    n_queries = len(ground_truth)
    n_zero = sum(1 for c in stats_relevant_counts if c == 0)
    avg_relevant = sum(stats_relevant_counts) / n_queries if n_queries else 0
    max_relevant = max(stats_relevant_counts) if stats_relevant_counts else 0
    min_relevant = min(stats_relevant_counts) if stats_relevant_counts else 0

    # Lưu ground truth
    gt_path = args.output_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Lưu manifest
    manifest = {
        "method": "category-based binary relevance (không dùng popularity, không circular)",
        "train_items": len(train_items),
        "test_items": len(test_items),
        "n_queries": n_queries,
        "queries_with_zero_relevant": n_zero,
        "avg_relevant_per_query": round(avg_relevant, 2),
        "max_relevant_per_query": max_relevant,
        "min_relevant_per_query": min_relevant,
        "categories_in_train": {cat: len(ids) for cat, ids in sorted(train_by_cat.items())},
        "note": (
            "Relevant = cùng category trong tập train. "
            "Ground truth này độc lập với popularity và hành vi tổng hợp. "
            "Dùng để tính Recall@K, NDCG@K, MRR."
        ),
    }
    manifest_path = args.output_dir / "gt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"=== Ground truth ===")
    print(f"  Queries (test items)       : {n_queries:,}")
    print(f"  Queries không có relevant  : {n_zero}")
    print(f"  Avg relevant / query       : {avg_relevant:.1f}")
    print(f"  Min / Max relevant         : {min_relevant} / {max_relevant}")
    print(f"\n  ground_truth.json → {gt_path.resolve()}")
    print(f"  gt_manifest.json  → {manifest_path.resolve()}")
    print(f"\nPhân phối train theo category:")
    for cat, ids in sorted(train_by_cat.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<20} {len(ids):>5} items")


if __name__ == "__main__":
    main()
