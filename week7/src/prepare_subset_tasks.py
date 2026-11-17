"""Build the subset annotation interfaces from the output of `evaluate_recommendations_v2.py plan` (N2 of research_decisions.md).

    python3 -I -B week7/src/prepare_subset_tasks.py [--plan-dir week7/outputs/evaluation_results] [--evaldir week7/outputs/evaluation]
                                                   [--out-dir week7/outputs/subset_package]

Reads ONLY: labels_lock.json (proves `plan` ran after the primary lock), second_judgment_needed.csv, adjudication_needed.csv,
annotation_tasks.csv (blind-safe pair -> image names) and the image files. It never opens the private key and never writes to the
primary label files.

  * second_judgment/second_judgment_interface.html : blind second judgments for pairs with exactly ONE numeric primary label. The page contains
    NO label of anybody (tasks hold pair_id and image names only).
  * adjudication/adjudication_interface.html       : pairs where the two primary annotators gave different numeric labels. The two existing labels are
    shown (unattributed, ascending) and the adjudicator can pick only one of them (or U). Exports go to a separate file.

Each role folder is self-contained (page + guide + its own images) so the adjudication folder, which displays existing labels, can be
delivered only AFTER the third person has returned the blind second-judgment export (decision N1).
Every file written here is free of method / rank / score information.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
TEMPLATE = Path(__file__).parent / "subset_interface_template.html"
LEAK = re.compile(r"visual|metadata|cosine|resnet|anchor|top-?5|rank|score|method|baseline|\bcnn\b|embedding|tf-?idf|evaluator|amendment", re.I)
TITLES = {"second_judgment": "Chấm lần thứ hai (độc lập)", "adjudication": "Phân xử bất đồng"}


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(plan_dir: Path, evaldir: Path, out_dir: Path) -> dict:
    if not (plan_dir / "labels_lock.json").exists():
        raise SystemExit("no labels_lock.json: lock the primary files and run 'plan' first")
    for n in ("second_judgment_needed.csv", "adjudication_needed.csv", "plan_summary.json"):
        if not (plan_dir / n).exists():
            raise SystemExit(f"{n} missing: run 'plan' first")
    summary = json.loads((plan_dir / "plan_summary.json").read_text())
    second = _read(plan_dir / "second_judgment_needed.csv")
    adj = _read(plan_dir / "adjudication_needed.csv")
    if len(second) != summary["second_judgment_needed"] or len(adj) != summary["adjudication_needed"]:
        raise SystemExit("plan_summary.json does not match the plan CSV files")
    tasks = {r["pair_id"]: r for r in _read(evaldir / "annotation_tasks.csv") if r["set"] == "main"}
    ids_s = [r["pair_id"] for r in second]
    ids_a = [r["pair_id"] for r in adj]
    for name, ids in (("second_judgment_needed", ids_s), ("adjudication_needed", ids_a)):
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{name}: duplicate pair_id")
        unknown = [i for i in ids if i not in tasks]
        if unknown:
            raise SystemExit(f"{name}: pair_id not in annotation_tasks.csv: {unknown[:3]}")
    if set(ids_s) & set(ids_a):
        raise SystemExit("a pair cannot need both a second judgment and an adjudication")

    out_dir.mkdir(parents=True, exist_ok=True)
    tpl = TEMPLATE.read_text(encoding="utf-8")
    written, counts, image_counts = {}, {}, {}
    for mode, ids, offset in (("second_judgment", ids_s, 3), ("adjudication", ids_a, 4)):
        counts[mode] = len(ids)
        if not ids:
            continue
        pkg = out_dir / mode                                      # one self-contained folder per role: deliver them one after the other
        (pkg / "images").mkdir(parents=True, exist_ok=True)
        order = sorted(ids)
        random.Random(42 + offset).shuffle(order)
        pos = {p: i for i, p in enumerate(order)}
        items = []
        for p in sorted(ids):
            t = {"p": p, "q": tasks[p]["query_image"], "c": tasks[p]["candidate_image"], "o": pos[p]}
            if mode == "adjudication":
                row = next(r for r in adj if r["pair_id"] == p)
                a, b = int(row["label_annotator_1"]), int(row["label_annotator_2"])
                if a == b or {a, b} - {0, 1, 2}:
                    raise SystemExit(f"adjudication pair {p}: need two different labels in 0..2, got {a},{b}")
                t["opts"] = sorted([a, b])                         # unattributed, ascending
            for im in (t["q"], t["c"]):
                shutil.copyfile(evaldir / im, pkg / im)
            items.append(t)
        html = tpl.replace("__TITLE__", TITLES[mode]).replace("__CONFIG_JSON__", json.dumps({"mode": mode, "tasks": items}, separators=(",", ":")))
        hits = LEAK.findall(html)
        if hits:
            raise SystemExit(f"{mode} interface contains method-related words {hits[:3]}: refusing to write")
        if mode == "second_judgment" and re.search(r'"opts"|"label"', html):
            raise SystemExit("second-judgment interface must not carry any label")
        name = f"{mode}_interface.html"
        (pkg / name).write_text(html, encoding="utf-8")
        shutil.copyfile(evaldir / "annotation_guide.md", pkg / "annotation_guide.md")
        written[f"{mode}/{name}"] = sha_file(pkg / name)
        written[f"{mode}/annotation_guide.md"] = sha_file(pkg / "annotation_guide.md")
        image_counts[mode] = len(list((pkg / "images").glob("img_*.jpg")))
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "counts": counts,
                "files": written, "images": image_counts, "internal_only": "this manifest stays with the coordinator; send each role folder separately",
                "plan_inputs_sha256": {n: sha_file(plan_dir / n) for n in ("second_judgment_needed.csv", "adjudication_needed.csv", "plan_summary.json")},
                "role_rules": {"second_judgment": "blind; pairs with exactly one numeric primary label; no label displayed",
                               "adjudication": "pairs with two different numeric primary labels; choose one of the two shown labels or U; deliver only AFTER the second-judgment export is received"},
                "csv_columns": ["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc", "role"],
                "private_key_opened": False}
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"counts": counts, "files": sorted(written), "images": image_counts}, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-dir", type=Path, default=W7 / "outputs/evaluation_results")
    ap.add_argument("--evaldir", type=Path, default=W7 / "outputs/evaluation")
    ap.add_argument("--out-dir", type=Path, default=W7 / "outputs/subset_package")
    a = ap.parse_args(argv)
    build(a.plan_dir, a.evaldir, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
