"""Small SQLite tracking layer for data builds and model experiments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("tracking_schema.sql")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    connection.execute(
        "INSERT OR IGNORE INTO schema_metadata(version, applied_at_utc) VALUES (?, ?)",
        (SCHEMA_VERSION, utc_now()),
    )


def create_data_build_database(
    database_path: Path,
    *,
    dataset: dict,
    build: dict,
    inventory_rows: Sequence[dict],
    ground_truth: dict[str, list[dict]],
    errors: Sequence[str],
    warnings: Sequence[str],
    artifacts: Sequence[dict],
) -> None:
    """Atomically create a database containing one validated data build."""
    temporary_path = database_path.with_suffix(database_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    connection = connect(temporary_path)
    try:
        initialize(connection)
        with connection:
            connection.execute(
                """INSERT INTO datasets(
                       dataset_id, name, archive_path, archive_bytes, archive_sha256,
                       image_count, item_count, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dataset["dataset_id"], dataset["name"], dataset["archive_path"],
                    dataset["archive_bytes"], dataset["archive_sha256"],
                    dataset["image_count"], dataset["item_count"], dataset["created_at_utc"],
                ),
            )
            connection.execute(
                """INSERT INTO data_builds(
                       build_id, dataset_id, created_at_utc, seed, train_ratio,
                       valid_ratio, test_ratio, status, elapsed_seconds, error_count,
                       warning_count, protocol_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    build["build_id"], dataset["dataset_id"], build["created_at_utc"],
                    build["seed"], build["ratios"]["train"], build["ratios"]["valid"],
                    build["ratios"]["test"], build["status"], build["elapsed_seconds"],
                    len(errors), len(warnings), json.dumps(build["protocol"], sort_keys=True),
                ),
            )

            by_item: dict[str, list[dict]] = {}
            for row in inventory_rows:
                by_item.setdefault(row["item_id"], []).append(row)
            connection.executemany(
                """INSERT INTO items(build_id, item_id, split, category, image_count)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        build["build_id"], item_id, rows[0]["split"],
                        rows[0]["item_category"], len(rows),
                    )
                    for item_id, rows in sorted(by_item.items())
                ],
            )
            connection.executemany(
                """INSERT INTO images(
                       build_id, image_path, item_id, mask_path, split, role,
                       width, height, mask_label, view_category
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        build["build_id"], row["image_path"], row["item_id"],
                        row["mask_path"], row["split"], row["role"], row["width"],
                        row["height"], row["mask_label"], row["view_category"],
                    )
                    for row in inventory_rows
                ],
            )
            connection.executemany(
                """INSERT INTO retrieval_ground_truth(
                       build_id, split, query_image_path, relevant_gallery_path, item_id
                   ) VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        build["build_id"], split, row["query_image_path"],
                        row["relevant_gallery_path"], row["item_id"],
                    )
                    for split, rows in ground_truth.items()
                    for row in rows
                ],
            )
            connection.executemany(
                "INSERT INTO validation_events(build_id, severity, message) VALUES (?, ?, ?)",
                [(build["build_id"], "error", message) for message in errors]
                + [(build["build_id"], "warning", message) for message in warnings],
            )
            connection.executemany(
                """INSERT INTO build_artifacts(build_id, path, kind, bytes, sha256)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        build["build_id"], artifact["path"], artifact["kind"],
                        artifact["bytes"], artifact["sha256"],
                    )
                    for artifact in artifacts
                ],
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    finally:
        connection.close()
    temporary_path.replace(database_path)


def start_experiment(
    database_path: Path,
    *,
    build_id: str,
    variant: str,
    seed: int,
    config: dict,
    backbone: str | None = None,
    code_revision: str | None = None,
    notes: str | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    with connect(database_path) as connection:
        connection.execute(
            """INSERT INTO experiment_runs(
                   run_id, build_id, started_at_utc, status, variant, backbone,
                   seed, config_json, code_revision, notes
               ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)""",
            (
                run_id, build_id, utc_now(), variant, backbone, seed,
                json.dumps(config, sort_keys=True), code_revision, notes,
            ),
        )
    return run_id


def log_metric(
    database_path: Path,
    *,
    run_id: str,
    split: str,
    name: str,
    value: float,
    k: int | None = None,
) -> None:
    metric_k = 0 if k is None else k
    if metric_k < 0:
        raise ValueError("Metric k must be non-negative")
    with connect(database_path) as connection:
        connection.execute(
            """INSERT INTO metrics(run_id, split, name, k, value, recorded_at_utc)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, split, name, k)
               DO UPDATE SET value=excluded.value, recorded_at_utc=excluded.recorded_at_utc""",
            (run_id, split, name, metric_k, value, utc_now()),
        )


def log_experiment_artifact(
    database_path: Path,
    *,
    run_id: str,
    path: Path,
    kind: str,
) -> None:
    """Attach an existing file and its checksum to an experiment run."""
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    with connect(database_path) as connection:
        connection.execute(
            """INSERT INTO experiment_artifacts(run_id, path, kind, bytes, sha256)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(run_id, path) DO UPDATE SET
                   kind=excluded.kind, bytes=excluded.bytes, sha256=excluded.sha256""",
            (run_id, str(path.resolve()), kind, path.stat().st_size, digest.hexdigest()),
        )


def finish_experiment(database_path: Path, run_id: str, status: str = "completed") -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("Final experiment status must be completed or failed")
    with connect(database_path) as connection:
        cursor = connection.execute(
            """UPDATE experiment_runs
               SET status = ?, completed_at_utc = ? WHERE run_id = ?""",
            (status, utc_now(), run_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown run_id: {run_id}")
