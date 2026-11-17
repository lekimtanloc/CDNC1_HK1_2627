"""Validate and merge the third annotator's files (second judgments + adjudications) into ONE file for `lock --third`.

    python3 -I -B week7/src/subset_labels.py validate --mode second_judgment --plan-dir DIR annotator_03_second_judgment.csv
    python3 -I -B week7/src/subset_labels.py validate --mode adjudication    --plan-dir DIR annotator_03_adjudication.csv
    python3 -I -B week7/src/subset_labels.py merge --plan-dir DIR [--second FILE] [--adjudication FILE] --out annotator_03.csv

Rules (research_decisions.md N1/N2, amendment A1/A2):
  * the author id must be annotator_03 or higher (never annotator_01/02) and identical in every file;
  * every row carries role = second_judgment | adjudication and the role must match the file/mode;
  * a second-judgment row must belong to second_judgment_needed.csv (exactly one numeric primary label);
  * an adjudication row must belong to adjudication_needed.csv and its label must be one of the two existing labels (or U/blank);
  * unknown pair_id, duplicates, labels outside {0,1,2,U}, pairs in the wrong role -> error;
  * the primary label files are never read or written here. The private key is never opened.
Exit code 1 on errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ID_RE = re.compile(r"^annotator_(0[3-9]|[1-9][0-9])$")
HEAD = ["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc", "role"]
MODES = ("second_judgment", "adjudication")


def read_plan(plan_dir: Path) -> dict:
    with open(plan_dir / "second_judgment_needed.csv", newline="", encoding="utf-8") as f:
        second = {r["pair_id"] for r in csv.DictReader(f)}
    with open(plan_dir / "adjudication_needed.csv", newline="", encoding="utf-8") as f:
        adj = {r["pair_id"]: sorted([r["label_annotator_1"], r["label_annotator_2"]]) for r in csv.DictReader(f)}
    if second & set(adj):
        raise ValueError("plan lists a pair in both roles")
    return {"second_judgment": second, "adjudication": adj}


def validate_file(path: Path, mode: str, plan: dict) -> dict:
    errors, warnings = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    missing = [c for c in HEAD if c not in cols]
    if missing:
        return {"file": str(path), "mode": mode, "ok": False, "errors": [f"missing columns: {missing}"], "warnings": [], "rows": [], "annotator_id": None}
    annots = {r["annotator_id"].strip() for r in rows}
    if "" in annots:
        errors.append("rows without annotator_id")
    annots.discard("")
    if len(annots) > 1:
        errors.append(f"several annotator_id in one file: {sorted(annots)}")
    annot = next(iter(annots)) if len(annots) == 1 else None
    if annot and not ID_RE.match(annot):
        errors.append(f"annotator_id '{annot}' not allowed here (need annotator_03 or higher; annotator_01/02 are the primary annotators)")
    bad_role = [r["pair_id"] for r in rows if r["role"].strip() != mode]
    if bad_role:
        errors.append(f"{len(bad_role)} rows with role != '{mode}'")
    if any(r["set"].strip() != "main" for r in rows):
        errors.append("rows not in set 'main'")
    seen, dups = set(), []
    for r in rows:
        p = r["pair_id"].strip()
        dups += [p] if p in seen else []
        seen.add(p)
    if dups:
        errors.append(f"{len(dups)} duplicate pair_id (e.g. {dups[:3]})")
    other = "adjudication" if mode == "second_judgment" else "second_judgment"
    expected = plan[mode]
    wrong_role_pairs = [p for p in seen if p in plan[other]]
    unknown = [p for p in seen if p not in expected and p not in plan[other]]
    if wrong_role_pairs:
        errors.append(f"{len(wrong_role_pairs)} pairs belong to the '{other}' list, not '{mode}' (e.g. {wrong_role_pairs[:3]})")
    if unknown:
        errors.append(f"{len(unknown)} pair_id not in the plan (e.g. {unknown[:3]})")
    answered = 0
    for r in rows:
        p, lab = r["pair_id"].strip(), r["label"].strip().upper() if r["label"].strip().upper() == "U" else r["label"].strip()
        if lab == "":
            continue
        if lab not in ("0", "1", "2", "U"):
            errors.append(f"label '{r['label']}' outside 0/1/2/U (pair {p})")
        elif mode == "adjudication" and lab != "U" and p in plan["adjudication"] and lab not in plan["adjudication"][p]:
            errors.append(f"adjudication label {lab} for {p} is not one of the two existing labels {plan['adjudication'][p]}")
        if lab in ("0", "1", "2", "U"):
            answered += 1
    if any(r["uncertain"].strip() not in ("", "0", "1") for r in rows):
        errors.append("invalid 'uncertain' value")
    absent = len(expected) - len(seen & set(expected))
    if absent:
        warnings.append(f"{absent} of the {len(expected)} pairs of this role are absent from the file")
    return {"file": str(path), "mode": mode, "ok": not errors, "errors": errors, "warnings": warnings, "annotator_id": annot,
            "rows": rows, "answered": answered, "needed": len(expected), "complete": not errors and answered >= len(expected) and not absent}


def merge(plan_dir: Path, second: Path | None, adjudication: Path | None, out: Path) -> dict:
    plan = read_plan(plan_dir)
    parts = [(m, p) for m, p in (("second_judgment", second), ("adjudication", adjudication)) if p]
    if not parts:
        raise SystemExit("nothing to merge")
    reports = [validate_file(p, m, plan) for m, p in parts]
    errors = [e for r in reports for e in r["errors"]]
    ids = {r["annotator_id"] for r in reports}
    if len(ids) > 1:
        errors.append(f"the two files come from different annotators: {sorted(str(i) for i in ids)}")
    if errors:
        raise SystemExit("merge refused:\n" + "\n".join(errors[:10]))
    rows = sorted((r for rep in reports for r in rep["rows"]), key=lambda r: r["pair_id"])
    pids = [r["pair_id"] for r in rows]
    assert len(pids) == len(set(pids))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        for r in rows:
            w.writerow([r[c] for c in HEAD])
    summary = {"out": str(out), "annotator_id": next(iter(ids)), "rows": len(rows),
               "second_judgment": {"needed": len(plan["second_judgment"]), "answered": next((r["answered"] for r in reports if r["mode"] == "second_judgment"), 0)},
               "adjudication": {"needed": len(plan["adjudication"]), "answered": next((r["answered"] for r in reports if r["mode"] == "adjudication"), 0)}}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--mode", choices=MODES, required=True)
    v.add_argument("--plan-dir", type=Path, required=True)
    v.add_argument("file", type=Path)
    m = sub.add_parser("merge")
    m.add_argument("--plan-dir", type=Path, required=True)
    m.add_argument("--second", type=Path)
    m.add_argument("--adjudication", type=Path)
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        r = validate_file(a.file, a.mode, read_plan(a.plan_dir))
        r.pop("rows", None)
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0 if r["ok"] else 1
    print(json.dumps(merge(a.plan_dir, a.second, a.adjudication, a.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
