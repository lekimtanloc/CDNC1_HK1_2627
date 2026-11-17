"""Mutation check: deliberately break the evaluator in 6 places and confirm test_evaluation.py fails each time.
The evaluator file is restored in a finally-block and its SHA-256 is verified. Run from the repo root.
Returns a JSON-able dict; used by freeze_evaluator.py."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "week7/src/evaluate_recommendations.py"
MUTATIONS = {
    "missing_treated_as_zero_in_P5": ('j = [r for r in rels[:5] if r is not None]\n    return sum(r >= 1 for r in j) / len(j) if j else None',
                                      'j = [(r or 0) for r in rels[:5]]\n    return sum(r >= 1 for r in j) / len(j) if j else None'),
    "min_rule_becomes_max": ('return min(a, b), "min_rule"', 'return max(a, b), "min_rule"'),
    "idcg_from_top5_only": ('ideal = sorted((x for x in pool_labels if x is not None), reverse=True)[:5]',
                            'ideal = sorted((x for x in r5 if x is not None), reverse=True)[:5]'),
    "drop_rule_off_by_one": ('v["visual"]["n_missing"] > 1 or v["metadata"]["n_missing"] > 1', 'v["visual"]["n_missing"] > 2 or v["metadata"]["n_missing"] > 2'),
    "kappa_linear_uses_quadratic_weights": ('w = (np.abs(i - j) / (k - 1)) ** (1 if weight == "linear" else 2)\n    po = M / n',
                                            'w = (np.abs(i - j) / (k - 1)) ** 2\n    po = M / n'),
    "bootstrap_seed_changed": ('def stratified_bootstrap(vals_by_stratum: dict, B=B_DEFAULT, seed=SEED):',
                               'def stratified_bootstrap(vals_by_stratum: dict, B=B_DEFAULT, seed=7):'),
}


def run() -> dict:
    orig = F.read_text()
    h0 = hashlib.sha256(orig.encode()).hexdigest()
    out = {}
    try:
        for name, (a, b) in MUTATIONS.items():
            assert a in orig, f"mutation anchor not found: {name}"
            F.write_text(orig.replace(a, b, 1))
            r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                str(ROOT / "week7/tests/test_evaluation.py")], capture_output=True, text=True, cwd=ROOT)
            out[name] = "CAUGHT" if r.returncode != 0 else "MISSED"
    finally:
        F.write_text(orig)
    restored = hashlib.sha256(F.read_text().encode()).hexdigest() == h0
    return {"mutations": out, "all_caught": all(v == "CAUGHT" for v in out.values()), "evaluator_restored_identical": restored}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
