"""Freeze evaluator v2 and the methodology amendment: run every relevant test and the mutation check for real, verify the annotator
package allowlist, then write evaluator_v2_manifest.json, test_report.{json,md} and methodology_manifest.json (SHA-256).
Refuses to freeze if anything fails, a frozen protocol/v1 file changed, or a real label/result file exists. Run from the repo root.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
MD = W7 / "outputs/methodology"
sys.path.insert(0, str(W7 / "src")); sys.path.insert(0, str(W7 / "scripts"))

EVALUATOR_FILES = ["week7/src/evaluate_recommendations_v2.py", "week7/src/evaluate_recommendations.py",
                   "week7/outputs/evaluation/evaluator_manifest.json", "week7/src/validate_annotations.py",
                   "week7/tests/test_evaluation_v2.py", "week7/outputs/methodology/evaluator_spec_v2.md",
                   "week7/outputs/methodology/methodology_amendment_v1.md", "week7/config/evaluation.json", "week7/config/experiment.json",
                   "week7/scripts/protocol_lib.py", "week7/scripts/mutation_check_evaluator_v2.py"]
SUITES = [("v1 evaluator (superseded, must still pass unchanged)", "week7/tests/test_evaluation.py", []),
          ("v2 evaluator (amendment A1/A2/A3/A6)", "week7/tests/test_evaluation_v2.py", []),
          ("annotation package, validator, interface", "week7/tests/test_annotation.py", []),
          ("visual pipeline", "week7/tests/test_visual.py", []),
          ("frozen protocol (round-2 guard deselected: initial-state test, not applicable after rankings exist)", "week7/tests/test_protocol.py",
           ["--deselect", "week7/tests/test_protocol.py::test_no_ranking_or_embedding_exists_yet"])]
LEAK = re.compile(r"visual|metadata|cosine|resnet|anchor|top-?5|rank|score|method|baseline|\bcnn\b|embedding|tf-?idf|evaluator|amendment", re.I)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pytest_suite(path, extra):
    r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra, str(ROOT / path)],
                       capture_output=True, text=True, cwd=ROOT)
    tail = r.stdout.strip().splitlines()[-1]
    m = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|deselected|skipped|error)", tail)}
    assert r.returncode == 0 and m.get("passed") and not m.get("failed"), f"{path} failed:\n{r.stdout[-1500:]}"
    return {"summary_line": tail, **m}


def package_check() -> dict:
    import make_annotator_package as MP
    ev = W7 / "outputs/evaluation"
    man = json.loads((ev / "annotation_manifest.json").read_text())
    assert all(sha(ev / n) == h for n, h in man["hashes"].items()), "annotation_manifest hashes changed"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "pkg.zip"
        MP.build(out)
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            images = [n for n in names if n.startswith("evaluation/images/")]
            others = sorted(n for n in names if not n.startswith("evaluation/images/"))
            assert others == sorted(f"evaluation/{n}" for n in MP.ALLOWED), others
            assert len(images) == man["images"]["count"] and all(re.fullmatch(r"evaluation/images/img_[0-9a-f]{12}\.jpg", n) for n in images)
            blob = b"".join(z.read(n) for n in names if n.endswith((".md", ".html", ".json", ".csv"))).decode("utf-8")
            assert not LEAK.findall(blob), LEAK.findall(blob)[:5]
            assert not [n for n in names if re.search(r"evaluator|methodology|private|key|result|label", n, re.I) and "annotation_" not in n and not n.startswith("evaluation/images/")]
    return {"files_in_zip": len(names), "allowlist": MP.ALLOWED, "images": len(images), "leak_words_found": 0,
            "annotation_manifest_hashes_match": True, "evaluator_or_methodology_files_in_zip": 0}


def main() -> None:
    assert not list(W7.rglob("annotator_0*.csv")), "a real label file exists: freeze must happen before labels"
    assert not (W7 / "outputs/evaluation_results").exists(), "results already exist"
    pm = json.loads((W7 / "outputs/hm/protocol_manifest.json").read_text())
    bad = [f for f, v in pm["files"].items() if sha(ROOT / f) != v["sha256"]]
    assert not bad, f"protocol changed: {bad}"
    v1 = json.loads((W7 / "outputs/evaluation/evaluator_manifest.json").read_text())
    bad1 = [f for f, v in v1["files"].items() if sha(ROOT / f) != v["sha256"]]
    assert not bad1, f"evaluator v1 frozen files changed: {bad1}"
    suites = []
    for title, path, extra in SUITES:
        suites.append({"title": title, "file": path, **pytest_suite(path, extra)})
    import mutation_check_evaluator_v2 as MC
    mut = MC.run()
    assert mut["all_caught"] and mut["evaluator_restored_identical"], mut
    pkg = package_check()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    import numpy, sklearn
    env = {"python": sys.version.split()[0], "numpy": numpy.__version__, "scikit-learn": sklearn.__version__}
    ev2 = {"evaluator_version": "evaluator-v2", "supersedes": "evaluator-v1", "experiment_id": "W7-HM-SIMILAR-PRODUCTS-v1",
           "frozen_utc": now, "state": "FROZEN_BEFORE_LABELS", "labels_read_or_created": False,
           "rule": "evaluate refuses to run if any listed file differs; any change requires a new evaluator version and a logged reason",
           "files": {f: {"sha256": sha(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in EVALUATOR_FILES},
           "tests": {"file": "week7/tests/test_evaluation_v2.py", **next(s for s in suites if s["file"].endswith("test_evaluation_v2.py"))},
           "mutation_check": mut, "bootstrap": {"B": 10000, "seed": 42}, "environment": env}
    (MD / "evaluator_v2_manifest.json").write_text(json.dumps(ev2, indent=1))
    report = {"generated_utc": now, "suites": suites, "mutation_check": mut, "annotator_package_check": pkg,
              "frozen_protocol_files_unchanged": len(pm["files"]), "evaluator_v1_files_unchanged": len(v1["files"]),
              "real_labels_present": False, "research_metrics_computed": False, "environment": env}
    (MD / "test_report.json").write_text(json.dumps(report, indent=1))
    total = sum(s["passed"] for s in suites)
    lines = ["# Test report — evaluator v2 và methodology amendment v1", "", f"Sinh bởi `week7/scripts/freeze_evaluator_v2.py` lúc {now} (kết quả chạy thật, không chỉnh tay).", "",
             "| Bộ test | File | Kết quả |", "|---|---|---|"]
    lines += [f"| {s['title']} | `{s['file']}` | **PASS** — {s['summary_line']} |" for s in suites]
    lines += ["", f"Tổng: **{total} test PASS, 0 FAIL**.", "", "## Mutation check (evaluator v2)", "", "| Đột biến | Kết quả |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in mut["mutations"].items()]
    lines += ["", f"Evaluator được khôi phục giống hệt sau mutation: {mut['evaluator_restored_identical']}.", "", "## Gói annotation (allowlist)", "",
              f"- {pkg['files_in_zip']} file trong zip = {len(pkg['allowlist'])} file allowlist + {pkg['images']} ảnh tên trung tính.",
              f"- Số từ gợi ý phương pháp/hạng/điểm/evaluator tìm thấy trong nội dung zip: {pkg['leak_words_found']}.",
              "- File `evaluator_*`, `methodology_*`, file khóa và kết quả trong zip: 0.", "- Hash trong `annotation_manifest.json` khớp.", "",
              "## Bất biến", "", f"- {len(pm['files'])} file protocol đóng băng: hash không đổi.", f"- {len(v1['files'])} file evaluator v1: hash không đổi.",
              "- Không có file nhãn thật; chưa có `evaluation_results/`; không metric nghiên cứu nào được tính.", "",
              "Lưu ý: `test_no_ranking_or_embedding_exists_yet` (vòng 2) bị loại có chủ đích — là test trạng thái ban đầu, không còn áp dụng khi ranking đã tồn tại; file test không được sửa."]
    (MD / "test_report.md").write_text("\n".join(lines) + "\n")
    # overall methodology manifest
    files = ["week7/outputs/methodology/methodology_amendment_v1.md", "week7/outputs/methodology/evaluator_spec_v2.md",
             "week7/outputs/methodology/evaluator_v2_manifest.json", "week7/outputs/methodology/test_report.json", "week7/outputs/methodology/test_report.md",
             "week7/src/evaluate_recommendations_v2.py", "week7/tests/test_evaluation_v2.py", "week7/scripts/freeze_evaluator_v2.py",
             "week7/scripts/mutation_check_evaluator_v2.py", "week7/src/make_annotator_package.py",
             "week7/outputs/hm/protocol_manifest.json", "week7/outputs/evaluation/evaluator_manifest.json",
             "week7/outputs/evaluation/annotation_manifest.json", "week7/outputs/evaluation/annotation_guide.md",
             "week7/outputs/evaluation/annotation_interface.html", "week7/outputs/evaluation/annotation_tasks.csv"]
    mm = {"manifest": "methodology_manifest", "created_utc": now, "state": "AMENDMENT_ADOPTED_PENDING_REVIEW",
          "evaluator_ready_for_labels": True, "evaluator_ready_conditions": "technical readiness only; N1-N4 are REVIEW_REQUIRED for the principal investigator",
          "supersedes": {"evaluator-v1": "week7/outputs/evaluation/evaluator_manifest.json (unchanged, still valid as a record)"},
          "original_freeze_untouched": {"protocol_files": len(pm["files"]), "evaluator_v1_files": len(v1["files"])},
          "files": {f: {"sha256": sha(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in files},
          "resolved_by_amendment": ["A1", "A2", "A3", "A6"], "moot": ["A4", "A11"], "revised": ["A9"],
          "open": ["A5", "A7", "A8", "A10", "A12", "A13", "N5"], "review_required": ["N1", "N2", "N3", "N4"]}
    (MD / "methodology_manifest.json").write_text(json.dumps(mm, indent=1))
    print(json.dumps({"suites": [(s["file"], s["summary_line"]) for s in suites], "mutations_caught": mut["all_caught"], "package": pkg["files_in_zip"]}, indent=1))


if __name__ == "__main__":
    main()
