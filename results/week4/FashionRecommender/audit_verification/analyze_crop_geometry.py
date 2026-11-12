#!/usr/bin/env python3
"""Measure bbox/resize/padding geometry for the five audit configurations."""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / "archive.zip"
TEST = ROOT / "results/week2/FashionRecommender/data_protocol/test.csv"
OUT = HERE / "artifacts"
CONFIGS = {
    "A": tuple(range(1, 13)),
    "B": (1, 2, 3, 5, 6, 7, 8, 10, 11, 12),
    "C": (1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13),
    "D": (1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 15),
    "E": (1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15),
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read(TEST)
    details = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for row in rows:
            mask = np.asarray(Image.open(io.BytesIO(archive.read(row["mask_path"]))).convert("L"))
            height, width = mask.shape
            for config, labels in CONFIGS.items():
                selected = np.isin(mask, labels)
                y, x = np.where(selected)
                if len(x):
                    bbox_width = int(x.max()) - int(x.min()) + 1
                    bbox_height = int(y.max()) - int(y.min()) + 1
                    scale = min(224 / bbox_width, 224 / bbox_height)
                    resized_width = min(224, max(1, round(bbox_width * scale)))
                    resized_height = min(224, max(1, round(bbox_height * scale)))
                    bbox_fraction = bbox_width * bbox_height / (width * height)
                    padding_fraction = 1 - resized_width * resized_height / (224 * 224)
                    fill_fraction = int(selected.sum()) / (bbox_width * bbox_height)
                else:
                    bbox_width = bbox_height = resized_width = resized_height = 0
                    bbox_fraction = 0.0
                    padding_fraction = 1.0
                    fill_fraction = 0.0
                details.append(
                    {
                        "config": config,
                        "image_path": row["image_path"],
                        "role": row["role"],
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "bbox_area_fraction": bbox_fraction,
                        "resized_width": resized_width,
                        "resized_height": resized_height,
                        "padding_fraction": padding_fraction,
                        "selected_fill_in_bbox": fill_fraction,
                    }
                )
    write(OUT / "crop_geometry.csv", details)
    summary = []
    for config in CONFIGS:
        values = [row for row in details if row["config"] == config]
        nonempty = [row for row in values if row["bbox_width"]]
        for role in ("all", "query", "gallery"):
            subset = nonempty if role == "all" else [row for row in nonempty if row["role"] == role]
            summary.append(
                {
                    "config": config,
                    "role": role,
                    "images": len(values) if role == "all" else sum(row["role"] == role for row in values),
                    "nonempty_images": len(subset),
                    "mean_bbox_area_fraction": float(np.mean([row["bbox_area_fraction"] for row in subset])),
                    "median_bbox_area_fraction": float(np.median([row["bbox_area_fraction"] for row in subset])),
                    "mean_padding_fraction": float(np.mean([row["padding_fraction"] for row in subset])),
                    "mean_selected_fill_in_bbox": float(np.mean([row["selected_fill_in_bbox"] for row in subset])),
                    "mean_resized_width": float(np.mean([row["resized_width"] for row in subset])),
                    "mean_resized_height": float(np.mean([row["resized_height"] for row in subset])),
                }
            )
    write(OUT / "crop_geometry_summary.csv", summary)
    print(json.dumps([row for row in summary if row["role"] == "all"], indent=2))


if __name__ == "__main__":
    main()
