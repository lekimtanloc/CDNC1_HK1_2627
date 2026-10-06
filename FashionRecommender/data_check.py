#!/usr/bin/env python3
"""Bước 3 — Kiểm tra dữ liệu (Tuần 3).

Kiểm kê toàn bộ archive.zip:
  - Số cặp ảnh/mask hợp lệ
  - Phân phối category (từ dominant mask label)
  - Phát hiện cặp thiếu mask hoặc thiếu ảnh
  - Ghi kết quả ra data_check_report.json

Chạy:
    python data_check.py --archive ../archive.zip
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
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
    arr = np.asarray(Image.open(io.BytesIO(mask_bytes)).convert("L"))
    garment = arr[(arr >= 1) & (arr <= 12)]
    if garment.size == 0:
        return 0, "Unknown"
    label = int(Counter(garment.reshape(-1).tolist()).most_common(1)[0][0])
    return label, MASK_LABEL_TO_CATEGORY.get(label, "Unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data_check_report.json"))
    parser.add_argument("--sample", type=int, default=0,
                        help="Nếu > 0: chỉ đọc N cặp để kiểm tra nhanh (0 = toàn bộ)")
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)

    print(f"Đọc archive: {args.archive} ({args.archive.stat().st_size:,} byte)")

    with zipfile.ZipFile(args.archive) as zf:
        names = set(zf.namelist())

        # Thu thập tất cả image path và item_id duy nhất
        image_names = sorted(
            n for n in names if n.startswith("images/") and n.endswith(".jpg")
        )
        print(f"Tổng file ảnh trong archive: {len(image_names)}")

        # Ghép cặp ảnh–mask theo item_id (lấy item_id đầu tiên của stem)
        seen_item_ids: set[str] = set()
        paired: list[tuple[str, str, str]] = []   # (item_id, image_name, mask_name)
        missing_mask: list[str] = []

        for img in image_names:
            stem = Path(img).stem
            item_id = stem.split("_", 1)[0]
            if item_id in seen_item_ids:
                continue
            seen_item_ids.add(item_id)
            mask = f"masks/{stem}.png"
            if mask in names:
                paired.append((item_id, img, mask))
            else:
                missing_mask.append(img)

        print(f"Cặp hợp lệ (item_id duy nhất có cả ảnh + mask): {len(paired)}")
        print(f"Ảnh thiếu mask: {len(missing_mask)}")

        # Giới hạn sample nếu cần
        to_scan = paired if args.sample == 0 else paired[: args.sample]

        # Đọc dominant category từ mask
        category_counter: Counter = Counter()
        label_counter: Counter = Counter()
        unknown_count = 0
        size_mismatches = 0

        for i, (item_id, img_name, mask_name) in enumerate(to_scan, 1):
            img_bytes = zf.read(img_name)
            mask_bytes = zf.read(mask_name)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
            if img_pil.size != mask_pil.size:
                size_mismatches += 1
            label, category = dominant_category(mask_bytes)
            category_counter[category] += 1
            label_counter[label] += 1
            if category == "Unknown":
                unknown_count += 1
            if i % 500 == 0 or i == len(to_scan):
                print(f"  [{i}/{len(to_scan)}] scanned...", flush=True)

    # Kết quả
    scanned = len(to_scan)
    report = {
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "total_image_files": len(image_names),
        "unique_item_ids": len(seen_item_ids),
        "valid_pairs": len(paired),
        "missing_mask_count": len(missing_mask),
        "missing_mask_examples": missing_mask[:5],
        "scanned_for_category": scanned,
        "size_mismatch_count": size_mismatches,
        "unknown_category_count": unknown_count,
        "category_distribution": dict(category_counter.most_common()),
        "label_distribution": {str(k): v for k, v in sorted(label_counter.items())},
        "category_count": len([c for c in category_counter if c != "Unknown"]),
    }

    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n=== Kết quả ===")
    print(f"Cặp hợp lệ  : {report['valid_pairs']:,}")
    print(f"Thiếu mask  : {report['missing_mask_count']}")
    print(f"Size mismatch: {report['size_mismatch_count']}")
    print(f"Unknown cat : {report['unknown_category_count']}")
    print(f"\nPhân phối category ({scanned} mẫu scan):")
    for cat, count in category_counter.most_common():
        pct = 100 * count / scanned if scanned else 0
        print(f"  {cat:<20} {count:>5}  ({pct:.1f}%)")
    print(f"\nBáo cáo đã lưu: {args.output.resolve()}")


if __name__ == "__main__":
    main()
