"""Mutation check for the subset workflow: break rules in the validator/merge tool, the gate, the generator and the browser interface;
test_subset_workflow.py must fail each time. Every touched file is restored in a finally-block and its SHA-256 verified."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "week7/src"
NO_BROWSER = "not headless_chrome and not interface_exports and not refuses_a_label and not primary_ids"
M = [
    ("adjudication_may_pick_a_foreign_label", "subset_labels.py", [('elif mode == "adjudication" and lab != "U" and p in plan["adjudication"] and lab not in plan["adjudication"][p]:',
                                                                    'elif False and lab not in plan["adjudication"][p]:')], NO_BROWSER),
    ("annotator_01_accepted_as_third_person", "subset_labels.py", [("^annotator_(0[3-9]|[1-9][0-9])$", "^annotator_(0[1-9]|[1-9][0-9])$")], NO_BROWSER),
    ("role_column_not_checked", "subset_labels.py", [('if r["role"].strip() != mode]', 'if r["role"].strip() != mode and False]')], NO_BROWSER),
    ("gate_threshold_off_by_one", "reporting_gate.py", [('if len(valid) < th["min_valid_queries"]:', 'if len(valid) <= th["min_valid_queries"]:')], NO_BROWSER),
    ("gate_group_threshold_off_by_one", "reporting_gate.py", [('if n < th["min_per_group"]}', 'if n <= th["min_per_group"]}')], NO_BROWSER),
    ("adjudication_labels_attributed_by_order", "prepare_subset_tasks.py", [('t["opts"] = sorted([a, b])', 't["opts"] = [a, b]')], NO_BROWSER),
    ("interface_accepts_primary_ids", "subset_interface_template.html", [("^annotator_(0[3-9]|[1-9][0-9])$", "^annotator_(0[1-9]|[1-9][0-9])$")], "headless_chrome"),
    ("interface_import_not_strict", "subset_interface_template.html", [("if (errs.length) return {ok:false", "if (false) return {ok:false")], "headless_chrome"),
    ("interface_adjudication_allows_any_label", "subset_interface_template.html", [("function allowed(t){ return MODE === 'adjudication' ? [String(t.opts[0]), String(t.opts[1]), 'U'] : ['0','1','2','U']; }",
                                                                                   "function allowed(t){ return ['0','1','2','U']; }")], "headless_chrome"),
]


def run() -> dict:
    out = {}
    for name, fname, edits, k in M:
        f = SRC / fname
        orig = f.read_bytes()
        h0 = hashlib.sha256(orig).hexdigest()
        try:
            text = orig.decode("utf-8")
            for a, b in edits:
                assert a in text, f"mutation anchor not found: {name}"
                text = text.replace(a, b, 1)
            f.write_text(text, encoding="utf-8")
            r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-k", k,
                                str(ROOT / "week7/tests/test_subset_workflow.py")], capture_output=True, text=True, cwd=ROOT)
            out[name] = "CAUGHT" if r.returncode != 0 else "MISSED"
        finally:
            f.write_bytes(orig)
        assert hashlib.sha256(f.read_bytes()).hexdigest() == h0, f"{fname} not restored"
    return {"mutations": out, "all_caught": all(v == "CAUGHT" for v in out.values()), "files_restored_identical": True}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
