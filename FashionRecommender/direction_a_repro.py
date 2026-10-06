#!/usr/bin/env python3
"""Reproducible, corrected small run for Direction A.

The Kaggle "Adjusted" archive contains paired ``images/*.jpg`` and
``masks/*.png`` but removes the original gender/category path metadata. This
runner reads pairs directly from the ZIP (no full extraction), derives a coarse
category from the dominant garment-mask label, extracts pretrained CNN features,
and compares visual-only, popularity-only, and fused recommendation variants.

This is a feasibility/reproduction run. Synthetic user behaviour is not real
personalisation ground truth, and gender alignment is deliberately not reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import resource
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected Direction-A reproduction run")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs"))
    parser.add_argument("--sample-items", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--max-purchases", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backbone", choices=("resnet50", "densenet121"), default="resnet50")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_entries(
    archive: zipfile.ZipFile, limit: int, seed: int
) -> list[tuple[str, str, str]]:
    names = set(archive.namelist())
    available = []
    seen_item_ids = set()
    for image_name in sorted(name for name in names if name.startswith("images/") and name.endswith(".jpg")):
        stem = Path(image_name).stem
        item_id = stem.split("_", 1)[0]
        mask_name = f"masks/{stem}.png"
        if mask_name not in names or item_id in seen_item_ids:
            continue
        available.append((item_id, image_name, mask_name))
        seen_item_ids.add(item_id)
    if len(available) < limit:
        raise RuntimeError(f"Requested {limit} unique items, found only {len(available)}")
    selected = random.Random(seed).sample(available, limit)
    return sorted(selected, key=lambda row: int(row[0]))


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if sys.platform == "darwin" else value * 1024


def dominant_category(mask_array: np.ndarray) -> tuple[int, str]:
    garment = mask_array[(mask_array >= 1) & (mask_array <= 12)]
    if garment.size == 0:
        return 0, "Unknown"
    label = int(Counter(garment.reshape(-1).tolist()).most_common(1)[0][0])
    return label, MASK_LABEL_TO_CATEGORY.get(label, "Unknown")


def load_masked_image(
    archive: zipfile.ZipFile, image_name: str, mask_name: str
) -> tuple[Image.Image, int, str]:
    image = Image.open(io.BytesIO(archive.read(image_name))).convert("RGB")
    mask = Image.open(io.BytesIO(archive.read(mask_name))).convert("L")
    if image.size != mask.size:
        raise ValueError(f"Image/mask size mismatch: {image_name} vs {mask_name}")
    image_array = np.asarray(image)
    mask_array = np.asarray(mask)
    label, category = dominant_category(mask_array)
    garment = (mask_array >= 1) & (mask_array <= 12)
    masked = np.where(garment[..., None], image_array, 0).astype(np.uint8)
    return Image.fromarray(masked, mode="RGB"), label, category


def build_model(name: str, pretrained: bool) -> tuple[torch.nn.Module, str]:
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        weight_name = str(weights) if weights else "random"
    else:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = torch.nn.Identity()
        weight_name = str(weights) if weights else "random"
    model.eval()
    return model, weight_name


def extract_features(
    archive: zipfile.ZipFile,
    entries: list[tuple[str, str, str]],
    model: torch.nn.Module,
    batch_size: int,
) -> tuple[torch.Tensor, list[dict]]:
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    feature_batches = []
    metadata = []
    started = time.perf_counter()
    for offset in range(0, len(entries), batch_size):
        tensors = []
        for item_id, image_name, mask_name in entries[offset : offset + batch_size]:
            image, label, category = load_masked_image(archive, image_name, mask_name)
            tensors.append(preprocess(image))
            metadata.append(
                {
                    "item_id": item_id,
                    "image_path": image_name,
                    "mask_path": mask_name,
                    "mask_label": label,
                    "category": category,
                    "gender": "unavailable_in_adjusted_archive",
                }
            )
        with torch.inference_mode():
            feature_batches.append(model(torch.stack(tensors)).cpu())
        done = min(offset + batch_size, len(entries))
        print(f"features: {done}/{len(entries)}", flush=True)
    features = F.normalize(torch.cat(feature_batches, dim=0), dim=1)
    elapsed = time.perf_counter() - started
    print(f"feature extraction: {elapsed:.2f}s", flush=True)
    return features, metadata


def minmax(values: np.ndarray) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def simulate_users(
    rng: np.random.Generator, item_count: int, num_users: int, max_purchases: int
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    popularity = rng.integers(0, 101, size=item_count).astype(np.float64)
    trendiness = rng.random(num_users)
    purchases = []
    for user_index in range(num_users):
        count = int(rng.integers(1, max_purchases + 1))
        chosen = []
        for _ in range(count):
            normalized = minmax(popularity)
            distance = np.abs(normalized - trendiness[user_index])
            weights = np.exp(-5.0 * distance)
            if chosen:
                weights[chosen] = 0.0
            weights /= weights.sum()
            item_index = int(rng.choice(item_count, p=weights))
            chosen.append(item_index)
            popularity[item_index] += 1
            purchases.append({"user_id": user_index + 1, "item_index": item_index})
    return purchases, popularity, trendiness


def recommend_and_evaluate(
    features: torch.Tensor,
    metadata: list[dict],
    purchases: list[dict],
    popularity: np.ndarray,
    trendiness: np.ndarray,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    popularity_norm = minmax(popularity)
    by_user: dict[int, list[int]] = {}
    for row in purchases:
        by_user.setdefault(row["user_id"], []).append(row["item_index"])

    variants = {
        "popularity_only": (0.0, 1.0),
        "visual_only": (1.0, 0.0),
        "visual_plus_popularity": (0.7, 0.3),
    }
    results = []
    metrics = []
    item_count = len(metadata)

    for variant, (visual_weight, popularity_weight) in variants.items():
        exact_matches = []
        popularity_errors = []
        relevance_values = []
        for user_id, purchased in sorted(by_user.items()):
            blocked = set(purchased)
            user_trend = float(trendiness[user_id - 1])
            pop_score = 1.0 - np.abs(popularity_norm - user_trend) ** 4
            for purchased_index in purchased:
                visual_score = (features @ features[purchased_index]).numpy()
                score = visual_weight * visual_score + popularity_weight * pop_score
                for index in blocked:
                    score[index] = -np.inf
                ranked = np.argsort(-score)[: min(top_k, item_count - len(blocked))]
                purchased_category = metadata[purchased_index]["category"]
                for rank, candidate_index in enumerate(ranked, start=1):
                    candidate_category = metadata[int(candidate_index)]["category"]
                    exact = float(candidate_category == purchased_category)
                    error = abs(float(popularity_norm[int(candidate_index)]) - user_trend)
                    relevance = float(score[int(candidate_index)])
                    exact_matches.append(exact)
                    popularity_errors.append(error)
                    relevance_values.append(relevance)
                    results.append(
                        {
                            "variant": variant,
                            "user_id": user_id,
                            "purchased_item": metadata[purchased_index]["item_id"],
                            "purchased_category": purchased_category,
                            "rank": rank,
                            "recommended_item": metadata[int(candidate_index)]["item_id"],
                            "recommended_category": candidate_category,
                            "category_exact_match": exact,
                            "user_trendiness": user_trend,
                            "recommended_popularity": float(popularity_norm[int(candidate_index)]),
                            "popularity_absolute_error": error,
                            "relevance": relevance,
                        }
                    )
        metrics.append(
            {
                "variant": variant,
                "recommendation_rows": len(exact_matches),
                "category_exact_match": float(np.mean(exact_matches)),
                "popularity_mae": float(np.mean(popularity_errors)),
                "mean_relevance": float(np.mean(relevance_values)),
            }
        )
    return results, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.sample_items <= args.top_k + args.max_purchases:
        raise ValueError("sample-items must exceed top-k + max-purchases")
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model, weight_name = build_model(args.backbone, pretrained=not args.no_pretrained)
    with zipfile.ZipFile(args.archive) as archive:
        entries = select_entries(archive, args.sample_items, args.seed)
        features, metadata = extract_features(archive, entries, model, args.batch_size)

    rng = np.random.default_rng(args.seed)
    purchases, popularity, trendiness = simulate_users(
        rng, len(metadata), args.num_users, args.max_purchases
    )
    results, metrics = recommend_and_evaluate(
        features, metadata, purchases, popularity, trendiness, args.top_k
    )

    torch.save(features, args.output_dir / f"features_{args.backbone}.pt")
    write_csv(args.output_dir / "metadata.csv", metadata)
    write_csv(args.output_dir / "purchases.csv", purchases)
    write_csv(args.output_dir / "recommendations.csv", results)
    write_csv(args.output_dir / "metrics.csv", metrics)

    output_files = []
    for filename in (
        f"features_{args.backbone}.pt",
        "metadata.csv",
        "purchases.csv",
        "recommendations.csv",
        "metrics.csv",
    ):
        path = args.output_dir / filename
        output_files.append(
            {"path": filename, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )

    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "small feasibility/reproduction run",
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256(args.archive),
        "paired_entries_in_archive": 13752,
        "sample_items": len(metadata),
        "seed": args.seed,
        "num_users": args.num_users,
        "max_purchases": args.max_purchases,
        "top_k": args.top_k,
        "backbone": args.backbone,
        "weights": weight_name,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "limitations": [
            "Adjusted Kaggle archive removed original gender labels; Gender Alignment is not reported.",
            "Category is derived from the dominant garment mask label, not the original category path.",
            "User purchases and trendiness are synthetic, so this is not real personalisation evidence.",
            "Results on this sample must not be presented as reproduction of the paper's full-dataset numbers.",
        ],
        "metrics": metrics,
        "outputs": output_files,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"outputs: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
