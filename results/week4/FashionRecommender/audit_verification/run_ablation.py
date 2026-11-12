#!/usr/bin/env python3
"""Controlled Week-4 mask-label ablation on the frozen test protocol.

This is an audit-only runner. It never modifies week1--week4 artifacts and
refuses to reuse an existing output directory.
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
import shlex
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms


HERE = Path(__file__).resolve().parent
WEEK4 = HERE.parent
ROOT = WEEK4.parents[2]
PROTOCOL = ROOT / "results/week2/FashionRecommender/data_protocol"
ARCHIVE = ROOT / "archive.zip"
V1_DIR = WEEK4 / "artifacts_completed"
V2_DIR = WEEK4 / "v2/artifacts_v2"
METRIC_SOURCE = ROOT / "FashionRecommender/week3_retrieval_metrics.py"
sys.path.insert(0, str(METRIC_SOURCE.parent))
from week3_retrieval_metrics import evaluate, score_query  # noqa: E402


CONFIGS = {
    "A": tuple(range(1, 13)),
    "B": (1, 2, 3, 5, 6, 7, 8, 10, 11, 12),
    "C": (1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13),
    "D": (1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 15),
    "E": (1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15),
}

OFFICIAL_LABELS = {
    0: "background",
    1: "top",
    2: "skirt",
    3: "leggings",
    4: "dress",
    5: "outer",
    6: "pants",
    7: "bag",
    8: "neckwear",
    9: "headwear",
    10: "eyeglass",
    11: "belt",
    12: "footwear",
    13: "hair",
    14: "skin",
    15: "face",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not rows and not fields:
        raise ValueError(f"No rows or fields for {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pair(archive: zipfile.ZipFile, row: dict[str, str]) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(io.BytesIO(archive.read(row["image_path"]))).convert("RGB")
    mask = np.asarray(Image.open(io.BytesIO(archive.read(row["mask_path"]))).convert("L"))
    assert image.size == (mask.shape[1], mask.shape[0])
    return image, mask


def crop_mask_bbox(image: Image.Image, mask: np.ndarray, labels: tuple[int, ...]) -> tuple[Image.Image, int]:
    selected = np.isin(mask, labels)
    y, x = np.where(selected)
    output = Image.new("RGB", (224, 224))
    if len(x) == 0:
        return output, 0
    masked = Image.fromarray(np.where(selected[..., None], np.asarray(image), 0).astype(np.uint8))
    bbox = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
    fitted = ImageOps.contain(masked.crop(bbox), (224, 224), Image.Resampling.BILINEAR)
    output.paste(fitted, ((224 - fitted.width) // 2, (224 - fitted.height) // 2))
    return output, int(selected.sum())


def group_name(positive_count: int) -> str:
    if positive_count == 1:
        return "1 positive"
    if positive_count <= 4:
        return "2-4 positives"
    return "5+ positives"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    start = time.monotonic()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    test_rows = read_csv(PROTOCOL / "test.csv")
    queries = [row for row in test_rows if row["role"] == "query"]
    gallery = [row for row in test_rows if row["role"] == "gallery"]
    ordered = queries + gallery
    assert (len(test_rows), len(queries), len(gallery)) == (2129, 586, 1543)

    ground_truth: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(PROTOCOL / "ground_truth_test.csv"):
        ground_truth[row["query_image_path"]].add(row["relevant_gallery_path"])
    assert len(ground_truth) == 586
    assert sum(map(len, ground_truth.values())) == 1439

    # A and E are existing deterministic full runs. Validate and slice them by path.
    v1_index = read_csv(V1_DIR / "fixed_all_index.csv")
    v2_index = read_csv(V2_DIR / "fixed_all_v2_index.csv")
    assert [row["image_path"] for row in v1_index] == [row["image_path"] for row in v2_index]
    source_index = {row["image_path"]: index for index, row in enumerate(v1_index)}
    ordered_indices = [source_index[row["image_path"]] for row in ordered]
    embeddings: dict[str, np.ndarray] = {
        "A": np.asarray(np.load(V1_DIR / "fixed_all.npy", mmap_mode="r")[ordered_indices]),
        "E": np.asarray(np.load(V2_DIR / "fixed_all_v2.npy", mmap_mode="r")[ordered_indices]),
    }
    assert embeddings["A"].shape == embeddings["E"].shape == (2129, 2048)

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "platform": platform.platform(),
    }
    v1_manifest = json.loads((V1_DIR / "manifest.json").read_text(encoding="utf-8"))
    v2_manifest = json.loads((V2_DIR / "manifest_v2.json").read_text(encoding="utf-8"))
    assert versions == v1_manifest["versions"] == v2_manifest["versions"]
    assert v1_manifest["weights_sha256"] == v2_manifest["weights_sha256"]

    weights_path = Path(torch.hub.get_dir()) / "checkpoints/resnet50-11ad3fa6.pth"
    assert sha256(weights_path) == v1_manifest["weights_sha256"]
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    timings: dict[str, float] = {}
    crop_rows: list[dict] = []
    # Extract only the three missing ablations. A/E are exact prior controlled runs.
    with zipfile.ZipFile(ARCHIVE) as archive:
        for config in ("B", "C", "D"):
            stage_start = time.monotonic()
            batches: list[np.ndarray] = []
            labels = CONFIGS[config]
            for offset in range(0, len(ordered), args.batch_size):
                tensors = []
                for row in ordered[offset : offset + args.batch_size]:
                    image, mask = load_pair(archive, row)
                    cropped, kept_pixels = crop_mask_bbox(image, mask, labels)
                    tensors.append(preprocess(cropped))
                    crop_rows.append(
                        {
                            "config": config,
                            "image_path": row["image_path"],
                            "role": row["role"],
                            "product_id": row["item_id"],
                            "kept_pixels": kept_pixels,
                            "all_black": int(not np.asarray(cropped).any()),
                        }
                    )
                with torch.inference_mode():
                    batch = F.normalize(model(torch.stack(tensors)), dim=1).numpy()
                batches.append(batch)
                if offset % 400 == 0:
                    print(f"extract {config}: {offset}/{len(ordered)}", flush=True)
            array = np.concatenate(batches)
            assert array.shape == (2129, 2048)
            assert np.isfinite(array).all()
            assert np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-5)
            embeddings[config] = array
            np.save(output_dir / f"test_embeddings_{config}.npy", array)
            timings[f"extract_{config}"] = time.monotonic() - stage_start

        # Recompute the crop-only evidence for A/E in the same pass and code path.
        for config in ("A", "E"):
            labels = CONFIGS[config]
            for index, row in enumerate(ordered):
                image, mask = load_pair(archive, row)
                cropped, kept_pixels = crop_mask_bbox(image, mask, labels)
                crop_rows.append(
                    {
                        "config": config,
                        "image_path": row["image_path"],
                        "role": row["role"],
                        "product_id": row["item_id"],
                        "kept_pixels": kept_pixels,
                        "all_black": int(not np.asarray(cropped).any()),
                    }
                )
                if index % 800 == 0:
                    print(f"audit crop {config}: {index}/{len(ordered)}", flush=True)

    write_csv(output_dir / "crop_audit.csv", crop_rows)
    black_counts = {
        config: sum(row["all_black"] for row in crop_rows if row["config"] == config)
        for config in CONFIGS
    }
    black_queries = {
        config: sum(
            row["all_black"]
            for row in crop_rows
            if row["config"] == config and row["role"] == "query"
        )
        for config in CONFIGS
    }

    summary_rows: list[dict] = []
    group_rows: list[dict] = []
    per_query_rows: list[dict] = []
    top10_rows: list[dict] = []
    positive_rows: list[dict] = []
    query_scores: dict[str, dict[str, dict[str, float]]] = {}
    query_rankings: dict[str, dict[str, list[str]]] = {}

    query_count = len(queries)
    for config in CONFIGS:
        vectors = embeddings[config]
        similarities = vectors[:query_count] @ vectors[query_count:].T
        rankings: dict[str, list[str]] = {}
        scores_for_config: dict[str, dict[str, float]] = {}
        for query_index, query in enumerate(queries):
            query_path = query["image_path"]
            order = np.argsort(-similarities[query_index], kind="stable")
            ranked_paths = [gallery[int(index)]["image_path"] for index in order]
            rankings[query_path] = ranked_paths
            query_metric = score_query(ranked_paths, ground_truth[query_path], 5)
            query_metric["top1"] = float(ranked_paths[0] in ground_truth[query_path])
            query_metric["hit_at_5"] = float(any(path in ground_truth[query_path] for path in ranked_paths[:5]))
            scores_for_config[query_path] = query_metric
            for rank, gallery_index in enumerate(order[:10], 1):
                candidate = gallery[int(gallery_index)]
                top10_rows.append(
                    {
                        "config": config,
                        "query_id": query_path,
                        "rank": rank,
                        "candidate_id": candidate["image_path"],
                        "product_id": candidate["item_id"],
                        "score": float(similarities[query_index, gallery_index]),
                        "is_correct": int(candidate["image_path"] in ground_truth[query_path]),
                    }
                )
            gallery_lookup = {row["image_path"]: index for index, row in enumerate(gallery)}
            for positive_path in sorted(ground_truth[query_path]):
                positive_rows.append(
                    {
                        "config": config,
                        "query_id": query_path,
                        "positive_id": positive_path,
                        "positive_count": len(ground_truth[query_path]),
                        "positive_group": group_name(len(ground_truth[query_path])),
                        "cosine": float(similarities[query_index, gallery_lookup[positive_path]]),
                    }
                )
        query_scores[config] = scores_for_config
        query_rankings[config] = rankings

        checked_top1 = evaluate(rankings, ground_truth, 1)["reciprocal_rank"]
        checked_recall = evaluate(rankings, ground_truth, 5)["recall"]
        top1 = float(np.mean([row["top1"] for row in scores_for_config.values()]))
        recall5 = float(np.mean([row["recall"] for row in scores_for_config.values()]))
        hit5 = float(np.mean([row["hit_at_5"] for row in scores_for_config.values()]))
        assert np.isclose(top1, checked_top1)
        assert np.isclose(recall5, checked_recall)
        positive_cosines = [row["cosine"] for row in positive_rows if row["config"] == config]
        summary_rows.append(
            {
                "config": config,
                "keep_ids": " ".join(map(str, CONFIGS[config])),
                "official_regions": "; ".join(OFFICIAL_LABELS[label] for label in CONFIGS[config]),
                "top1": top1,
                "recall_at_5": recall5,
                "hit_at_5": hit5,
                "black_test_images": black_counts[config],
                "black_queries": black_queries[config],
                "mean_positive_cosine": float(np.mean(positive_cosines)),
                "queries_gain_vs_A": 0,
                "queries_loss_vs_A": 0,
                "queries_same_vs_A": len(queries),
            }
        )

    # Fill per-query transitions and aggregate group diagnostics.
    for config in CONFIGS:
        gain = loss = same = 0
        for query in queries:
            query_path = query["image_path"]
            current = query_scores[config][query_path]
            baseline = query_scores["A"][query_path]
            delta = current["recall"] - baseline["recall"]
            transition = "gain" if delta > 1e-12 else "loss" if delta < -1e-12 else "same"
            gain += transition == "gain"
            loss += transition == "loss"
            same += transition == "same"
            per_query_rows.append(
                {
                    "config": config,
                    "query_id": query_path,
                    "positive_count": len(ground_truth[query_path]),
                    "positive_group": group_name(len(ground_truth[query_path])),
                    "top1": current["top1"],
                    "recall_at_5": current["recall"],
                    "hit_at_5": current["hit_at_5"],
                    "recall_delta_vs_A": delta,
                    "transition_vs_A": transition,
                }
            )
        summary = next(row for row in summary_rows if row["config"] == config)
        summary["queries_gain_vs_A"] = gain
        summary["queries_loss_vs_A"] = loss
        summary["queries_same_vs_A"] = same

        for group in ("1 positive", "2-4 positives", "5+ positives"):
            members = [
                query["image_path"]
                for query in queries
                if group_name(len(ground_truth[query["image_path"]])) == group
            ]
            group_positive_cosines = [
                row["cosine"]
                for row in positive_rows
                if row["config"] == config and row["positive_group"] == group
            ]
            transitions = [
                row["transition_vs_A"]
                for row in per_query_rows
                if row["config"] == config and row["positive_group"] == group
            ]
            group_rows.append(
                {
                    "config": config,
                    "positive_group": group,
                    "queries": len(members),
                    "top1": float(np.mean([query_scores[config][path]["top1"] for path in members])),
                    "recall_at_5": float(np.mean([query_scores[config][path]["recall"] for path in members])),
                    "hit_at_5": float(np.mean([query_scores[config][path]["hit_at_5"] for path in members])),
                    "mean_positive_cosine": float(np.mean(group_positive_cosines)),
                    "queries_gain_vs_A": transitions.count("gain"),
                    "queries_loss_vs_A": transitions.count("loss"),
                    "queries_same_vs_A": transitions.count("same"),
                }
            )

    write_csv(output_dir / "ablation_summary.csv", summary_rows)
    write_csv(output_dir / "ablation_groups.csv", group_rows)
    write_csv(output_dir / "ablation_per_query.csv", per_query_rows)
    write_csv(output_dir / "ablation_top10.csv", top10_rows)
    write_csv(output_dir / "positive_pair_scores.csv", positive_rows)

    # A/E must reproduce the already saved rankings exactly.
    def read_saved_ranking(path: Path) -> dict[str, list[str]]:
        value: dict[str, list[str]] = defaultdict(list)
        for row in read_csv(path):
            value[row["query_id"]].append(row["candidate_id"])
        return value

    saved_a = read_saved_ranking(V1_DIR / "visual_only_top10.csv")
    saved_e = read_saved_ranking(V2_DIR / "visual_only_v2_top10.csv")
    reuse_checks = {
        "A_top10_exact_match_v1": all(query_rankings["A"][key][:10] == value for key, value in saved_a.items()),
        "E_top10_exact_match_v2": all(query_rankings["E"][key][:10] == value for key, value in saved_e.items()),
        "A_embedding_source_sha256": sha256(V1_DIR / "fixed_all.npy"),
        "E_embedding_source_sha256": sha256(V2_DIR / "fixed_all_v2.npy"),
        "A_source_rows": len(v1_index),
        "E_source_rows": len(v2_index),
    }
    assert reuse_checks["A_top10_exact_match_v1"]
    assert reuse_checks["E_top10_exact_match_v2"]
    dump_json(output_dir / "reuse_checks.json", reuse_checks)

    inputs = [
        ARCHIVE,
        PROTOCOL / "test.csv",
        PROTOCOL / "ground_truth_test.csv",
        V1_DIR / "fixed_all.npy",
        V1_DIR / "fixed_all_index.csv",
        V1_DIR / "visual_only_top10.csv",
        V1_DIR / "manifest.json",
        V2_DIR / "fixed_all_v2.npy",
        V2_DIR / "fixed_all_v2_index.csv",
        V2_DIR / "visual_only_v2_top10.csv",
        V2_DIR / "manifest_v2.json",
        METRIC_SOURCE,
        Path(__file__).resolve(),
        weights_path,
    ]
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full frozen test split",
        "seed": seed,
        "device": "cpu",
        "threads": 4,
        "batch_size": args.batch_size,
        "deterministic_algorithms": True,
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "weights_sha256": sha256(weights_path),
        "versions": versions,
        "configurations": {key: list(value) for key, value in CONFIGS.items()},
        "official_label_mapping": OFFICIAL_LABELS,
        "embedding_execution": {
            "A": "reused exact deterministic Week-4 v1 full-run artifact",
            "B": "extracted in this audit run",
            "C": "extracted in this audit run",
            "D": "extracted in this audit run",
            "E": "reused exact deterministic Week-4 v2 full-run artifact",
        },
        "timings_seconds": timings,
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "command": "python3 -B " + shlex.join(sys.argv),
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
    }
    manifest_path = output_dir / "manifest.json"
    dump_json(manifest_path, manifest)
    output_files = sorted(path for path in output_dir.iterdir() if path != manifest_path)
    manifest["outputs"] = [
        {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": len(read_csv(path)) if path.suffix == ".csv" else (int(np.load(path, mmap_mode="r").shape[0]) if path.suffix == ".npy" else None),
        }
        for path in output_files
    ]
    # Manifest is intentionally rewritten before its own hash is included.
    manifest_path.unlink()
    dump_json(manifest_path, manifest)

    print(json.dumps({"summary": summary_rows, "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
