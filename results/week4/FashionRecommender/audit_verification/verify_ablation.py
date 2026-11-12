#!/usr/bin/env python3
"""Independent arithmetic/integrity checks for the audit ablation outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = HERE / "artifacts"
PROTOCOL = ROOT / "results/week2/FashionRecommender/data_protocol"
sys.path.insert(0, str(ROOT / "FashionRecommender"))
from week3_retrieval_metrics import evaluate  # noqa: E402


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: object = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    for output in manifest["outputs"]:
        path = ARTIFACTS / output["path"]
        check(f"hash:{output['path']}", sha256(path) == output["sha256"], output["sha256"])

    truth: dict[str, set[str]] = defaultdict(set)
    for row in read(PROTOCOL / "ground_truth_test.csv"):
        truth[row["query_image_path"]].add(row["relevant_gallery_path"])
    top10 = read(ARTIFACTS / "ablation_top10.csv")
    summary = {row["config"]: row for row in read(ARTIFACTS / "ablation_summary.csv")}
    positives = read(ARTIFACTS / "positive_pair_scores.csv")
    crops = read(ARTIFACTS / "crop_audit.csv")
    per_query = read(ARTIFACTS / "ablation_per_query.csv")
    groups = read(ARTIFACTS / "ablation_groups.csv")

    check("top10_rows", len(top10) == 29300, len(top10))
    check("positive_rows", len(positives) == 7195, len(positives))
    check("crop_rows", len(crops) == 10645, len(crops))
    check("per_query_rows", len(per_query) == 2930, len(per_query))
    check("group_rows", len(groups) == 15, len(groups))

    for config in "ABCDE":
        rankings: dict[str, list[str]] = defaultdict(list)
        config_rows = [row for row in top10 if row["config"] == config]
        for row in config_rows:
            rankings[row["query_id"]].append(row["candidate_id"])
        metric1 = evaluate(rankings, truth, 1)
        metric5 = evaluate(rankings, truth, 5)
        hit5 = np.mean(
            [any(candidate in truth[query] for candidate in ranked[:5]) for query, ranked in rankings.items()]
        )
        positive_mean = np.mean([float(row["cosine"]) for row in positives if row["config"] == config])
        black = sum(int(row["all_black"]) for row in crops if row["config"] == config)
        saved = summary[config]
        check(f"{config}:queries", len(rankings) == 586, len(rankings))
        check(f"{config}:top1", np.isclose(metric1["reciprocal_rank"], float(saved["top1"])), metric1)
        check(f"{config}:recall5", np.isclose(metric5["recall"], float(saved["recall_at_5"])), metric5)
        check(f"{config}:hit5", np.isclose(hit5, float(saved["hit_at_5"])), float(hit5))
        check(
            f"{config}:positive_cosine",
            np.isclose(positive_mean, float(saved["mean_positive_cosine"])),
            float(positive_mean),
        )
        check(f"{config}:black", black == int(saved["black_test_images"]), black)

    reuse = json.loads((ARTIFACTS / "reuse_checks.json").read_text(encoding="utf-8"))
    check("A_matches_saved_v1", reuse["A_top10_exact_match_v1"], reuse)
    check("E_matches_saved_v2", reuse["E_top10_exact_match_v2"], reuse)

    report = {
        "status": "passed",
        "checks_passed": len(checks),
        "checks_failed": 0,
        "checks": checks,
        "semantic_note": "The ablation manifest field official_label_mapping is the MMFashion documented color order used only for comparison; it is not asserted as the archive's semantic schema. See ../source_evidence.json and ../rgb_to_id.csv.",
    }
    output = ARTIFACTS / "verification.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks_passed": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
