"""Freeze the evaluator: run the tests and the mutation check for real, then write evaluator_manifest.json (SHA-256 of every file).
Refuses to freeze if a test fails, a mutation is missed, a protocol file changed, or a real label/result file exists.
Run from the repo root:  python3 -I -B week7/scripts/freeze_evaluator.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
OUT = W7 / "outputs/evaluation/evaluator_manifest.json"
FILES = ["week7/src/evaluate_recommendations.py", "week7/tests/test_evaluation.py", "week7/outputs/evaluation/evaluator_spec.md",
         "week7/src/validate_annotations.py", "week7/config/evaluation.json", "week7/config/experiment.json",
         "week7/scripts/protocol_lib.py", "week7/scripts/mutation_check_evaluator.py"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    assert not list(W7.rglob("annotator_0*.csv")), "a real label file exists: freeze must happen before labels"
    assert not (W7 / "outputs/evaluation_results").exists(), "results already exist: freeze must happen before evaluation"
    pm = json.loads((W7 / "outputs/hm/protocol_manifest.json").read_text())
    bad = [f for f, v in pm["files"].items() if sha(ROOT / f) != v["sha256"]]
    assert not bad, f"protocol changed: {bad}"
    r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(W7 / "tests/test_evaluation.py")],
                       capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) passed", r.stdout)
    assert r.returncode == 0 and m and "failed" not in r.stdout, "tests failed:\n" + r.stdout[-2000:]
    sys.path.insert(0, str(W7 / "scripts"))
    import mutation_check_evaluator as MC
    mut = MC.run()
    assert mut["all_caught"] and mut["evaluator_restored_identical"], mut
    import numpy, sklearn
    man = {"evaluator_version": "evaluator-v1", "experiment_id": "W7-HM-SIMILAR-PRODUCTS-v1",
           "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "state": "FROZEN_BEFORE_LABELS", "labels_read_or_created": False,
           "rule": "any change to a listed file requires a new evaluator version and a logged reason; evaluate refuses to run if a hash differs",
           "files": {f: {"sha256": sha(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in FILES},
           "protocol_files_verified_unchanged": len(pm["files"]),
           "tests": {"file": "week7/tests/test_evaluation.py", "passed": int(m.group(1)), "failed": 0,
                     "independent_references": ["scikit-learn ndcg_score", "scikit-learn cohen_kappa_score", "week7/scripts/protocol_lib.py paired_bootstrap"]},
           "mutation_check": mut,
           "bootstrap": {"B": 10000, "seed": 42},
           "ambiguities": "A1-A13, see evaluator_spec.md section 7 (printed in every results.json); items A1, A2, A3, A6 flagged for researcher confirmation",
           "environment": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scikit-learn": sklearn.__version__}}
    OUT.write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("state", "tests", "mutation_check")}, indent=1))


if __name__ == "__main__":
    main()
