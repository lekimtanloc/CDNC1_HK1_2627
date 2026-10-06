#!/usr/bin/env python3
"""Bước 3 — Tạo split train/val/test cố định (Tuần 3).

Đọc toàn bộ cặp ảnh/mask hợp lệ từ archive, lấy category từ dominant mask label,
rồi chia thành 3 split theo tỷ lệ 70/15/15 (stratified theo category, seed cố định).

Đầu ra:
  split_train.json   — danh sách item_id + category cho tập train
  split_val.json     — tập validation
  split_test.json    — tập test (dùng để đánh giá cuối)
  split_manifest.json — metadata về split (counts, seed, tỷ lệ)

Chạy:
    python split_items.py --archive ../archive.zip
"""
from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MASK_LABEL_TO_CATEGORY = {
    1: "Tees_Tanks",
    2: "Skirts",
    3: "Leggings",
    4: "Dresses",
    5: "Jackets_Coats",
    6: "Pants",
    7: "Bags",
    8: "Neckwear",
    9: "Headwear",
    10: "Eyeglass",
    11: "Belts",
    12: "Footwear",
}


def dominant_category(mask_bytes: bytes) -> tuple[int, str]:
    from collections import Counter
    arr = np.asarray(Image.open(io.BytesIO(mask_bytes)).convert("L"))
    garment = arr[(arr >= 1) & (arr <= 12)]
    if garment.size == 0:
        return 0, "Unknown"
    label = int(Counter(garment.reshape(-1).tolist()).most_common(1)[0][0])
    return label, MASK_LABEL_TO_CATEGORY.get(label, "Unknown")


def stratified_split(
    items: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Chia stratified theo category, đảm bảo mỗi category có ít nhất 1 item/split nếu đủ."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_cat[item["category"]].append(item)

    train, val, test = [], [], []
    for cat, cat_items in by_cat.items():
        shuffled = cat_items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n > 1 else 0
        # Nếu tổng vượt quá n thì điều chỉnh
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    # Shuffle lại toàn bộ mỗi split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo split train/val/test cố định")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0,
                        help="Giới hạn số item để chạy nhanh (0 = toàn bộ)")
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đọc archive: {args.archive}")
    items_with_category: list[dict] = []

    with zipfile.ZipFile(args.archive) as zf:
        names = set(zf.namelist())
        image_names = sorted(
            n for n in names if n.startswith("images/") and n.endswith(".jpg")
        )
        seen_item_ids: set[str] = set()
        paired = []
        for img in image_names:
            stem = Path(img).stem
            item_id = stem.split("_", 1)[0]
            if item_id in seen_item_ids:
                continue
            seen_item_ids.add(item_id)
            mask = f"masks/{stem}.png"
            if mask in names:
                paired.append((item_id, img, mask))

        if args.sample > 0:
            rng_sample = random.Random(args.seed)
            paired = rng_sample.sample(paired, min(args.sample, len(paired)))

        print(f"Cặp hợp lệ: {len(paired)} — đang đọc category từ mask...")
        for i, (item_id, img_name, mask_name) in enumerate(paired, 1):
            mask_bytes = zf.read(mask_name)
            label, category = dominant_category(mask_bytes)
            items_with_category.append({
                "item_id": item_id,
                "image_path": img_name,
                "mask_path": mask_name,
                "mask_label": label,
                "category": category,
            })
            if i % 500 == 0 or i == len(paired):
                print(f"  [{i}/{len(paired)}]", flush=True)

    train, val, test = stratified_split(
        items_with_category,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = args.output_dir / f"split_{split_name}.json"
        path.write_text(json.dumps(split_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  {split_name}: {len(split_data)} items → {path}")

    # Category distribution per split
    def cat_dist(split_data: list[dict]) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(d["category"] for d in split_data).most_common())

    manifest = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 4),
        "total_items": len(items_with_category),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "train_category_dist": cat_dist(train),
        "val_category_dist": cat_dist(val),
        "test_category_dist": cat_dist(test),
        "archive": str(args.archive.resolve()),
        "note": "Split cố định dùng cho toàn bộ thí nghiệm — không thay đổi sau khi tạo",
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n=== Kết quả split ===")
    print(f"  Train : {len(train):,}")
    print(f"  Val   : {len(val):,}")
    print(f"  Test  : {len(test):,}")
    print(f"  Total : {len(items_with_category):,}")
    print(f"  Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
