#!/usr/bin/env python3
"""Build reproducible, leakage-safe retrieval splits from the adjusted archive.

The archive contains ``images/<item_id>_<view>.jpg`` and matching
``masks/<item_id>_<view>.png`` members.  Item IDs are assigned to exactly one
of train/valid/test.  Within each split, one view of every multi-view item is a
query and its remaining views are relevant gallery images.  This makes the
retrieval ground truth independent from model scores and synthetic popularity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from experiment_tracking import create_data_build_database


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
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class PairRecord:
    item_id: str
    image_path: str
    mask_path: str
    image_bytes: int
    mask_bytes: int
    width: int
    height: int
    mask_label: int
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the adjusted DeepFashion ZIP and create Week-2 splits"
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results/week2/FashionRecommender/data_protocol",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Read image headers only. Masks are always decoded to derive labels.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero after writing the report when validation errors exist.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dominant_category(mask: np.ndarray) -> tuple[int, str]:
    garment = mask[(mask >= 1) & (mask <= 12)]
    if garment.size == 0:
        return 0, "Unknown"
    counts = np.bincount(garment.reshape(-1), minlength=13)
    label = int(np.argmax(counts[1:13]) + 1)
    return label, MASK_LABEL_TO_CATEGORY[label]


def item_id_from_image(image_path: str) -> str:
    stem = Path(image_path).stem
    if "_" not in stem:
        raise ValueError(f"Image name has no item/view separator: {image_path}")
    return stem.split("_", 1)[0]


def inspect_pair(
    archive: zipfile.ZipFile,
    image_info: zipfile.ZipInfo,
    mask_info: zipfile.ZipInfo,
    verify_image: bool,
) -> PairRecord:
    image_bytes = archive.read(image_info)
    mask_bytes = archive.read(mask_info)
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        if verify_image:
            image.verify()
    with Image.open(io.BytesIO(mask_bytes)) as mask_image:
        mask_image.load()
        mask_width, mask_height = mask_image.size
        mask = np.asarray(mask_image.convert("L"))
    if (width, height) != (mask_width, mask_height):
        raise ValueError(
            f"Image/mask dimensions differ: {image_info.filename} {(width, height)} "
            f"vs {mask_info.filename} {(mask_width, mask_height)}"
        )
    label, category = dominant_category(mask)
    return PairRecord(
        item_id=item_id_from_image(image_info.filename),
        image_path=image_info.filename,
        mask_path=mask_info.filename,
        image_bytes=image_info.file_size,
        mask_bytes=mask_info.file_size,
        width=width,
        height=height,
        mask_label=label,
        category=category,
    )


def inventory_archive(
    archive_path: Path, verify_image: bool
) -> tuple[list[PairRecord], list[str], list[str], dict]:
    records: list[PairRecord] = []
    errors: list[str] = []
    warnings: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        member_counts = Counter(info.filename for info in infos)
        duplicate_members = sorted(name for name, count in member_counts.items() if count > 1)
        if duplicate_members:
            errors.append(f"Duplicate ZIP members: {len(duplicate_members)}")

        info_by_name = {info.filename: info for info in infos}
        image_names = sorted(
            name
            for name in info_by_name
            if name.startswith("images/") and name.lower().endswith(".jpg")
        )
        mask_names = {
            name
            for name in info_by_name
            if name.startswith("masks/") and name.lower().endswith(".png")
        }
        expected_masks = {f"masks/{Path(name).stem}.png" for name in image_names}
        missing_masks = sorted(expected_masks - mask_names)
        orphan_masks = sorted(mask_names - expected_masks)
        if missing_masks:
            errors.append(f"Images without masks: {len(missing_masks)}")
        if orphan_masks:
            warnings.append(f"Masks without images: {len(orphan_masks)}")

        paired_names = [
            name for name in image_names if f"masks/{Path(name).stem}.png" in mask_names
        ]
        total = len(paired_names)
        for index, image_name in enumerate(paired_names, start=1):
            mask_name = f"masks/{Path(image_name).stem}.png"
            try:
                records.append(
                    inspect_pair(
                        archive,
                        info_by_name[image_name],
                        info_by_name[mask_name],
                        verify_image,
                    )
                )
            except (OSError, ValueError, zipfile.BadZipFile, UnidentifiedImageError) as exc:
                errors.append(f"{image_name}: {exc}")
            if index % 1000 == 0 or index == total:
                print(f"validated pairs: {index}/{total}", flush=True)

    archive_stats = {
        "zip_members": len(infos),
        "image_members": len(image_names),
        "mask_members": len(mask_names),
        "paired_members": len(paired_names),
        "valid_pairs": len(records),
        "missing_masks": len(missing_masks),
        "orphan_masks": len(orphan_masks),
        "duplicate_members": len(duplicate_members),
    }
    return records, errors, warnings, archive_stats


def allocate_counts(size: int, ratios: Sequence[float]) -> list[int]:
    raw = [size * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = size - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def majority_category(records: Iterable[PairRecord]) -> str:
    categories = [record.category for record in records if record.category != "Unknown"]
    if not categories:
        return "Unknown"
    counts = Counter(categories)
    return min(counts, key=lambda category: (-counts[category], category))


def stratified_item_split(
    by_item: dict[str, list[PairRecord]], ratios: Sequence[float], seed: int
) -> dict[str, str]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for item_id, records in sorted(by_item.items()):
        by_category[majority_category(records)].append(item_id)

    assignments: dict[str, str] = {}
    for category in sorted(by_category):
        item_ids = by_category[category]
        random.Random(f"{seed}:{category}").shuffle(item_ids)
        counts = allocate_counts(len(item_ids), ratios)
        offset = 0
        for split, count in zip(SPLITS, counts):
            for item_id in item_ids[offset : offset + count]:
                assignments[item_id] = split
            offset += count
    return assignments


def build_rows(
    records: Sequence[PairRecord], assignments: dict[str, str], seed: int
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[dict]]:
    by_item: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        by_item[record.item_id].append(record)

    split_rows = {split: [] for split in SPLITS}
    ground_truth = {split: [] for split in SPLITS}
    inventory_rows = []
    for item_id in sorted(by_item, key=lambda value: (len(value), value)):
        item_records = sorted(by_item[item_id], key=lambda record: record.image_path)
        shuffled = list(item_records)
        random.Random(f"{seed}:{item_id}:views").shuffle(shuffled)
        query_path = shuffled[0].image_path if len(shuffled) >= 2 else None
        split = assignments[item_id]
        item_category = majority_category(item_records)

        for record in item_records:
            role = "query" if record.image_path == query_path else "gallery"
            row = {
                "split": split,
                "role": role,
                "item_id": item_id,
                "image_path": record.image_path,
                "mask_path": record.mask_path,
                "image_bytes": record.image_bytes,
                "mask_bytes": record.mask_bytes,
                "width": record.width,
                "height": record.height,
                "mask_label": record.mask_label,
                "view_category": record.category,
                "item_category": item_category,
            }
            inventory_rows.append(row)
            split_rows[split].append({key: value for key, value in row.items() if key != "split"})

        if query_path is not None:
            for relevant in shuffled[1:]:
                ground_truth[split].append(
                    {
                        "query_image_path": query_path,
                        "relevant_gallery_path": relevant.image_path,
                        "item_id": item_id,
                        "item_category": item_category,
                    }
                )
    return split_rows, ground_truth, inventory_rows


def validate_protocol(
    split_rows: dict[str, list[dict]], ground_truth: dict[str, list[dict]]
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    item_sets = {
        split: {row["item_id"] for row in rows} for split, rows in split_rows.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = item_sets[left] & item_sets[right]
            if overlap:
                errors.append(f"Item leakage {left}/{right}: {len(overlap)}")

    all_images: list[str] = []
    for split, rows in split_rows.items():
        images = [row["image_path"] for row in rows]
        all_images.extend(images)
        roles = {row["image_path"]: row["role"] for row in rows}
        items = {row["image_path"]: row["item_id"] for row in rows}
        for gt in ground_truth[split]:
            query = gt["query_image_path"]
            gallery = gt["relevant_gallery_path"]
            if query == gallery:
                errors.append(f"Self-match in {split}: {query}")
            if roles.get(query) != "query" or roles.get(gallery) != "gallery":
                errors.append(f"Invalid query/gallery role in {split}: {query} -> {gallery}")
            if items.get(query) != gt["item_id"] or items.get(gallery) != gt["item_id"]:
                errors.append(f"Ground-truth item mismatch in {split}: {query} -> {gallery}")
        if split in ("valid", "test") and not ground_truth[split]:
            errors.append(f"No retrieval ground truth in {split}")
        singleton_count = sum(
            1
            for item_id in item_sets[split]
            if sum(row["item_id"] == item_id for row in rows) == 1
        )
        if singleton_count:
            warnings.append(
                f"{split}: {singleton_count} single-view items are gallery negatives only"
            )
    if len(all_images) != len(set(all_images)):
        errors.append("An image occurs more than once across generated splits")
    return errors, warnings


def output_hashes(output_dir: Path, filenames: Sequence[str]) -> list[dict]:
    outputs = []
    for filename in filenames:
        path = output_dir / filename
        outputs.append(
            {"path": filename, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    return outputs


def artifact_kind(filename: str) -> str:
    if filename.startswith("ground_truth_"):
        return "ground_truth"
    if Path(filename).stem in SPLITS:
        return "split"
    if filename == "inventory.csv":
        return "inventory"
    return "documentation"


def write_readme(output_dir: Path, archive_path: Path, seed: int, ratios: Sequence[float]) -> None:
    text = f"""# Week 2 data protocol

This directory was generated from `{archive_path.name}` with seed `{seed}` and
train/valid/test ratios `{ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}`.

- Item IDs are disjoint across train, validation, and test.
- One view of each multi-view item is the query; its other views are relevant gallery images.
- Single-view items remain gallery negatives and do not create a query.
- Relevance uses item identity only. Category is secondary metadata derived from masks.
- Synthetic popularity and user behaviour are not used to build ground truth.

Regenerate from `FashionRecommender/`:

```bash
python3 prepare_week2_splits.py --archive ../{archive_path.name} --output-dir {output_dir.name} --seed {seed}
```

`validation_report.json` records archive checks, split statistics, limitations,
and SHA-256 hashes of all generated CSV files. `tracking.sqlite3` stores the
same data lineage and provides tables for future experiment runs and metrics.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if any(ratio <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("train/valid/test ratios must be positive and sum to 1")

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, errors, warnings, archive_stats = inventory_archive(
        args.archive, verify_image=not args.skip_image_verify
    )
    by_item: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        by_item[record.item_id].append(record)
    assignments = stratified_item_split(by_item, ratios, args.seed)
    split_rows, ground_truth, inventory_rows = build_rows(records, assignments, args.seed)
    protocol_errors, protocol_warnings = validate_protocol(split_rows, ground_truth)
    errors.extend(protocol_errors)
    warnings.extend(protocol_warnings)

    inventory_fields = [
        "split", "role", "item_id", "image_path", "mask_path", "image_bytes",
        "mask_bytes", "width", "height", "mask_label", "view_category", "item_category",
    ]
    split_fields = inventory_fields[1:]
    gt_fields = [
        "query_image_path", "relevant_gallery_path", "item_id", "item_category"
    ]
    filenames: list[str] = []
    write_csv(args.output_dir / "inventory.csv", inventory_rows, inventory_fields)
    filenames.append("inventory.csv")
    for split in SPLITS:
        split_name = f"{split}.csv"
        gt_name = f"ground_truth_{split}.csv"
        write_csv(args.output_dir / split_name, split_rows[split], split_fields)
        write_csv(args.output_dir / gt_name, ground_truth[split], gt_fields)
        filenames.extend((split_name, gt_name))
    write_readme(args.output_dir, args.archive, args.seed, ratios)
    filenames.append("README.md")

    split_stats = {}
    for split in SPLITS:
        rows = split_rows[split]
        item_ids = {row["item_id"] for row in rows}
        query_ids = {row["item_id"] for row in rows if row["role"] == "query"}
        categories = Counter(row["item_category"] for row in rows)
        split_stats[split] = {
            "items": len(item_ids),
            "images": len(rows),
            "queries": sum(row["role"] == "query" for row in rows),
            "gallery_images": sum(row["role"] == "gallery" for row in rows),
            "retrievable_items": len(query_ids),
            "ground_truth_pairs": len(ground_truth[split]),
            "item_category_image_counts": dict(sorted(categories.items())),
        }

    created_at = datetime.now(timezone.utc).isoformat()
    archive_digest = sha256(args.archive)
    status = "passed" if not errors else "failed"
    elapsed_seconds = time.perf_counter() - started
    protocol = {
        "split_unit": "item_id",
        "query_gallery_rule": "one deterministic query view; remaining views are positives",
        "primary_relevance": "same item_id, different image",
        "uses_popularity_for_ground_truth": False,
        "image_content_verified": not args.skip_image_verify,
    }
    artifact_rows = output_hashes(args.output_dir, filenames)
    for artifact in artifact_rows:
        artifact["kind"] = artifact_kind(artifact["path"])
    dataset_id = f"adjusted-deepfashion-{archive_digest[:12]}"
    build_id = f"week2-seed{args.seed}-{archive_digest[:12]}"
    database_name = "tracking.sqlite3"
    create_data_build_database(
        args.output_dir / database_name,
        dataset={
            "dataset_id": dataset_id,
            "name": "DeepFashion In-shop Clothes Retrieval Adjusted",
            "archive_path": str(args.archive.resolve()),
            "archive_bytes": args.archive.stat().st_size,
            "archive_sha256": archive_digest,
            "image_count": len(records),
            "item_count": len(by_item),
            "created_at_utc": created_at,
        },
        build={
            "build_id": build_id,
            "created_at_utc": created_at,
            "seed": args.seed,
            "ratios": dict(zip(SPLITS, ratios)),
            "status": status,
            "elapsed_seconds": elapsed_seconds,
            "protocol": protocol,
        },
        inventory_rows=inventory_rows,
        ground_truth=ground_truth,
        errors=errors,
        warnings=warnings,
        artifacts=artifact_rows,
    )

    report = {
        "status": status,
        "created_at_utc": created_at,
        "dataset_id": dataset_id,
        "build_id": build_id,
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": archive_digest,
        "seed": args.seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "protocol": protocol,
        "archive_stats": archive_stats,
        "dataset_stats": {
            "items": len(by_item),
            "images": len(records),
            "multi_view_items": sum(len(rows) >= 2 for rows in by_item.values()),
            "single_view_items": sum(len(rows) == 1 for rows in by_item.values()),
        },
        "split_stats": split_stats,
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            "Same-item retrieval evaluates visual identity retrieval, not real user preference.",
            "The adjusted archive has no real user behaviour or gender labels.",
            "Categories are coarse labels derived from masks and are secondary metrics only.",
        ],
        "elapsed_seconds": elapsed_seconds,
        "tracking_database": database_name,
        "outputs": artifact_rows,
    }
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "dataset_stats": report["dataset_stats"],
        "split_stats": report["split_stats"],
        "errors": errors,
        "warnings": warnings,
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2, ensure_ascii=False), flush=True)
    if errors and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
