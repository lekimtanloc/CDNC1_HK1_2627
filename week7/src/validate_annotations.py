"""Week 7: validate annotator CSV files against the annotation manifest. Computes NO quality metric.

    python3 -I -B week7/src/validate_annotations.py --set main annotator_01.csv [annotator_02.csv] [--require-complete] [--json out.json]

Errors (exit code 1): missing columns, empty/inconsistent annotator_id, wrong set, unknown or duplicate pair_id,
label outside {0,1,2,U}, bad 'uncertain' value, tampered tasks file vs manifest, (with --require-complete) unlabelled pairs.
Warnings: annotator_id differs from the file name, label 'U' counts, pairs not present in the file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALDIR = ROOT / "week7/outputs/evaluation"
REQUIRED = ["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc"]
VALID_LABELS = {"0", "1", "2", "U"}


def load_tasks(evaldir: Path = EVALDIR) -> dict[str, set[str]]:
    man = json.loads((evaldir / "annotation_manifest.json").read_text())
    h = hashlib.sha256((evaldir / "annotation_tasks.csv").read_bytes()).hexdigest()
    if h != man["hashes"]["annotation_tasks.csv"]:
        raise RuntimeError("annotation_tasks.csv does not match annotation_manifest.json (tampered or regenerated)")
    out: dict[str, set[str]] = {}
    with open(evaldir / "annotation_tasks.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["set"], set()).add(r["pair_id"])
    return out


def validate_file(path: Path, set_: str, tasks: dict[str, set[str]]) -> dict:
    errors, warnings = [], []
    expected = tasks[set_]
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    miss_cols = [c for c in REQUIRED if c not in cols]
    if miss_cols:
        return {"file": str(path), "ok": False, "errors": [f"missing columns: {miss_cols}"], "warnings": [], "annotator_id": None}
    annots = {r["annotator_id"].strip() for r in rows}
    if "" in annots:
        errors.append(f"{sum(1 for r in rows if not r['annotator_id'].strip())} rows without annotator_id")
    annots.discard("")
    if len(annots) > 1:
        errors.append(f"several annotator_id in one file: {sorted(annots)}")
    annot = next(iter(annots)) if len(annots) == 1 else None
    if annot and annot not in Path(path).stem:
        warnings.append(f"annotator_id '{annot}' differs from file name '{Path(path).name}'")
    bad_set = [r["pair_id"] for r in rows if r["set"].strip() != set_]
    if bad_set:
        errors.append(f"{len(bad_set)} rows are not in set '{set_}' (pilot and main must never be mixed)")
    seen, dups, unknown = set(), [], []
    for r in rows:
        pid = r["pair_id"].strip()
        if pid in seen:
            dups.append(pid)
        seen.add(pid)
        if pid not in expected and r["set"].strip() == set_:
            unknown.append(pid)
    if dups:
        errors.append(f"{len(dups)} duplicate pair_id (e.g. {dups[:3]})")
    if unknown:
        errors.append(f"{len(unknown)} pair_id not in the {set_} task list (e.g. {unknown[:3]})")
    labels = {"0": 0, "1": 0, "2": 0, "U": 0, "": 0}
    bad_labels = []
    for r in rows:
        lab = r["label"].strip().upper() if r["label"].strip().upper() == "U" else r["label"].strip()
        if lab == "":
            labels[""] += 1
        elif lab in VALID_LABELS:
            labels[lab] += 1
        else:
            bad_labels.append((r["pair_id"], r["label"]))
    if bad_labels:
        errors.append(f"{len(bad_labels)} labels outside {{0,1,2,U}} (e.g. {bad_labels[:3]})")
    bad_unc = [r["pair_id"] for r in rows if r["uncertain"].strip() not in ("", "0", "1")]
    if bad_unc:
        errors.append(f"{len(bad_unc)} invalid 'uncertain' values")
    present = {r["pair_id"].strip() for r in rows} & expected
    not_in_file = expected - {r["pair_id"].strip() for r in rows}
    if not_in_file:
        warnings.append(f"{len(not_in_file)} pairs of the {set_} set are absent from the file")
    answered = sum(labels[k] for k in ("0", "1", "2", "U"))
    if labels["U"]:
        warnings.append(f"{labels['U']} pairs labelled U (treated as missing, never 0)")
    return {"file": str(path), "annotator_id": annot, "set": set_, "rows": len(rows), "pairs_expected": len(expected),
            "pairs_present": len(present), "pairs_answered": answered, "pairs_unanswered": len(expected) - answered,
            "label_counts": labels, "complete": answered == len(expected) and not errors,
            "errors": errors, "warnings": warnings, "ok": not errors}


def validate_many(paths: list[Path], set_: str, tasks: dict[str, set[str]], require_complete: bool = False) -> dict:
    res = [validate_file(p, set_, tasks) for p in paths]
    errors = [e for r in res for e in r["errors"]]
    if require_complete:
        errors += [f"{r['file']}: {r['pairs_unanswered']} unanswered pairs" for r in res if r.get("pairs_unanswered")]
    cross = {}
    if len(res) >= 2 and all("annotator_id" in r for r in res):
        ids = [r["annotator_id"] for r in res]
        if len(set(ids)) != len(ids):
            errors.append(f"annotators are not distinct: {ids}")
        sets = [{r2["pair_id"].strip() for r2 in csv.DictReader(open(p, newline="", encoding="utf-8-sig"))} for p in paths]
        cross = {"pairs_in_all_files": len(set.intersection(*sets)), "pairs_in_any_file": len(set.union(*sets))}
    status = "INVALID" if errors else ("COMPLETE" if all(r.get("complete") for r in res) else "PARTIAL")
    return {"set": set_, "status": status, "errors": errors, "files": res, "cross_file": cross,
            "note": "validation only; no Precision/NDCG/kappa is computed here"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--set", choices=["main", "pilot"], default="main")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = validate_many(a.files, a.set, load_tasks(), a.require_complete)
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    if a.json:
        a.json.write_text(txt)
    print(txt)
    return 1 if out["status"] == "INVALID" else 0


if __name__ == "__main__":
    sys.exit(main())
