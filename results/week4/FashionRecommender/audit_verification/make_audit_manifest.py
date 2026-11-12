#!/usr/bin/env python3
"""Create a final checksum manifest for all verification deliverables."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "verification_report.md"
OUTPUT = HERE / "audit_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.reader(stream)) - 1


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = [
        path
        for path in HERE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path != OUTPUT
    ]
    files.append(REPORT)
    rows = []
    for path in sorted(files):
        rows.append(
            {
                "path": str(path.relative_to(HERE.parent)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": csv_rows(path) if path.suffix == ".csv" else None,
            }
        )
    value = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": rows,
    }
    OUTPUT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(rows), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
