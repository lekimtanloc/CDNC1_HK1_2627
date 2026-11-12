import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import prepare_week2_splits as week2
from experiment_tracking import (
    create_data_build_database,
    finish_experiment,
    log_metric,
    start_experiment,
)


class Week2ProtocolTests(unittest.TestCase):
    def make_archive(self, root: Path, items: int = 12) -> Path:
        archive_path = root / "fixture.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for item in range(items):
                for view in range(2):
                    stem = f"{1000 + item}_{view + 1:03d}"
                    image = Image.new("RGB", (8, 10), color=(item, view, 100))
                    mask = Image.fromarray(
                        np.full((10, 8), (item % 3) + 1, dtype=np.uint8), mode="L"
                    )
                    for folder, suffix, value in (
                        ("images", ".jpg", image), ("masks", ".png", mask)
                    ):
                        target = root / f"{stem}{suffix}"
                        value.save(target)
                        archive.write(target, f"{folder}/{stem}{suffix}")
        return archive_path

    def test_deterministic_item_disjoint_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = self.make_archive(root)
            records, errors, _, stats = week2.inventory_archive(archive_path, True)
            self.assertEqual(errors, [])
            self.assertEqual(stats["valid_pairs"], 24)
            by_item = {}
            for record in records:
                by_item.setdefault(record.item_id, []).append(record)

            ratios = (0.5, 0.25, 0.25)
            first = week2.stratified_item_split(by_item, ratios, seed=42)
            second = week2.stratified_item_split(by_item, ratios, seed=42)
            self.assertEqual(first, second)
            split_rows, ground_truth, _ = week2.build_rows(records, first, seed=42)
            protocol_errors, _ = week2.validate_protocol(split_rows, ground_truth)
            self.assertEqual(protocol_errors, [])

            split_items = {
                split: {row["item_id"] for row in rows}
                for split, rows in split_rows.items()
            }
            self.assertFalse(split_items["train"] & split_items["valid"])
            self.assertFalse(split_items["train"] & split_items["test"])
            self.assertFalse(split_items["valid"] & split_items["test"])
            for split in week2.SPLITS:
                self.assertTrue(ground_truth[split])
                for row in ground_truth[split]:
                    self.assertNotEqual(
                        row["query_image_path"], row["relevant_gallery_path"]
                    )

    def test_count_allocation_preserves_total(self):
        self.assertEqual(week2.allocate_counts(11, (0.7, 0.15, 0.15)), [8, 2, 1])
        self.assertEqual(sum(week2.allocate_counts(3, (0.7, 0.15, 0.15))), 3)

    def test_tracking_database_and_experiment_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = self.make_archive(root)
            records, errors, warnings, _ = week2.inventory_archive(archive_path, True)
            by_item = {}
            for record in records:
                by_item.setdefault(record.item_id, []).append(record)
            assignments = week2.stratified_item_split(
                by_item, (0.5, 0.25, 0.25), seed=42
            )
            split_rows, ground_truth, inventory = week2.build_rows(
                records, assignments, seed=42
            )
            errors, protocol_warnings = week2.validate_protocol(split_rows, ground_truth)
            warnings.extend(protocol_warnings)
            database = root / "tracking.sqlite3"
            build_id = "fixture-build"
            create_data_build_database(
                database,
                dataset={
                    "dataset_id": "fixture-dataset",
                    "name": "fixture",
                    "archive_path": str(archive_path),
                    "archive_bytes": archive_path.stat().st_size,
                    "archive_sha256": week2.sha256(archive_path),
                    "image_count": len(records),
                    "item_count": len(by_item),
                    "created_at_utc": "2026-01-01T00:00:00+00:00",
                },
                build={
                    "build_id": build_id,
                    "created_at_utc": "2026-01-01T00:00:00+00:00",
                    "seed": 42,
                    "ratios": {"train": 0.5, "valid": 0.25, "test": 0.25},
                    "status": "passed",
                    "elapsed_seconds": 1.0,
                    "protocol": {"split_unit": "item_id"},
                },
                inventory_rows=inventory,
                ground_truth=ground_truth,
                errors=errors,
                warnings=warnings,
                artifacts=[],
            )
            run_id = start_experiment(
                database,
                build_id=build_id,
                variant="visual_only",
                seed=42,
                backbone="resnet50",
                config={"masked": False},
            )
            log_metric(
                database,
                run_id=run_id,
                split="test",
                name="recall",
                k=5,
                value=0.5,
            )
            finish_experiment(database, run_id)
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM items").fetchone()[0], 12
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM images").fetchone()[0], 24
                )
                self.assertEqual(
                    connection.execute("SELECT status FROM experiment_runs").fetchone()[0],
                    "completed",
                )
                self.assertEqual(
                    connection.execute("SELECT value FROM metrics").fetchone()[0], 0.5
                )


if __name__ == "__main__":
    unittest.main()
