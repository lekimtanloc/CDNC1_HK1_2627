"""Mutation check for evaluator v2: break the amendment rules in 8 places; test_evaluation_v2.py must fail each time.
The file is restored in a finally-block and its SHA-256 verified. Run from the repo root."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "week7/src/evaluate_recommendations_v2.py"
MUTATIONS = {
    "A1_single_numeric_used_as_ground_truth": [('return None, "single_numeric_not_ground_truth"', 'return x, "single_numeric_not_ground_truth"')],
    "A2_third_label_overrides_single_numeric": [("if c == x:", "if True:"), ('return x, "second_judgment_agree"', 'return c, "second_judgment_agree"')],
    "A2_adjudicator_may_pick_foreign_label": [("if c not in (a, b):", "if False:")],
    "A3_RQ1_accepts_one_judged_anchor": [('len(vals[q]["anchors_judged"]) == 2', 'len(vals[q]["anchors_judged"]) >= 1')],
    "A6_RQ2_tolerates_one_missing_position": [('if vals[q]["visual"]["n_missing"] == 0 and vals[q]["metadata"]["n_missing"] == 0]',
                                              'if vals[q]["visual"]["n_missing"] <= 1 and vals[q]["metadata"]["n_missing"] <= 1]')],
    "A6_RQ1_also_requires_metadata_complete": [('rq1_visual = [q for q in sorted(vals) if vals[q]["visual"]["n_missing"] == 0]',
                                                'rq1_visual = [q for q in sorted(vals) if vals[q]["visual"]["n_missing"] == 0 and vals[q]["metadata"]["n_missing"] == 0]')],
    "numeric_tolerance_removed": [("TOL = 1e-12", "TOL = 0.0")],
    "mean_label_uses_single_numeric": [("        else:\n            out[p] = None\n    return out", "        else:\n            out[p] = a if a is not None else b\n    return out")],
}


def run() -> dict:
    orig = F.read_text()
    h0 = hashlib.sha256(orig.encode()).hexdigest()
    out = {}
    try:
        for name, edits in MUTATIONS.items():
            mutated = orig
            for a, b in edits:
                assert a in mutated, f"mutation anchor not found: {name}: {a[:40]}"
                mutated = mutated.replace(a, b, 1)
            F.write_text(mutated)
            r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                str(ROOT / "week7/tests/test_evaluation_v2.py")], capture_output=True, text=True, cwd=ROOT)
            out[name] = "CAUGHT" if r.returncode != 0 else "MISSED"
    finally:
        F.write_text(orig)
    restored = hashlib.sha256(F.read_text().encode()).hexdigest() == h0
    return {"mutations": out, "all_caught": all(v == "CAUGHT" for v in out.values()), "evaluator_restored_identical": restored}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
