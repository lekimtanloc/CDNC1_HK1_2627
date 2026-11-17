"""Freeze the subset-annotation workflow (decisions N1-N4): run ALL relevant test suites and the mutation check for real, re-verify every
previously frozen component, then write subset_workflow_manifest.json + subset_test_report.{json,md} (SHA-256).
Refuses to freeze if anything fails, a frozen file changed, a real label/result exists. Run from the repo root.
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
MD = W7 / "outputs/methodology"
sys.path.insert(0, str(W7 / "scripts"))

COMPONENTS = ["week7/outputs/methodology/research_decisions.md", "week7/outputs/methodology/research_decisions.json",
              "week7/src/subset_interface_template.html", "week7/src/subset_selftest.js", "week7/src/subset_autolabel.js",
              "week7/src/prepare_subset_tasks.py", "week7/src/subset_labels.py", "week7/src/reporting_gate.py",
              "week7/tests/test_subset_workflow.py", "week7/tests/test_annotation.py", "week7/scripts/mutation_check_subset.py",
              "week7/scripts/freeze_subset_workflow.py", "week7/outputs/instructions/huong_dan_annotator_01_02.md",
              "week7/outputs/instructions/huong_dan_annotator_03.md", "week7/outputs/instructions/huong_dan_dieu_phoi.md"]
PREVIOUSLY_FROZEN = ["week7/outputs/hm/protocol_manifest.json", "week7/outputs/evaluation/evaluator_manifest.json",
                     "week7/outputs/methodology/evaluator_v2_manifest.json", "week7/outputs/methodology/methodology_manifest.json"]
SUITES = [("evaluator v1 (superseded, unchanged)", "week7/tests/test_evaluation.py", []),
          ("evaluator v2 (amendment A1/A2/A3/A6)", "week7/tests/test_evaluation_v2.py", []),
          ("annotation package, validator, official interface", "week7/tests/test_annotation.py", []),
          ("visual pipeline", "week7/tests/test_visual.py", []),
          ("frozen protocol (round-2 initial-state guard deselected: not applicable once rankings exist)", "week7/tests/test_protocol.py",
           ["--deselect", "week7/tests/test_protocol.py::test_no_ranking_or_embedding_exists_yet"]),
          ("subset workflow: interfaces, second judgment, adjudication, merge, gate, N1-N4", "week7/tests/test_subset_workflow.py", [])]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def drift(manifest_rel: str) -> list:
    m = json.loads((ROOT / manifest_rel).read_text())
    return [f for f, v in m["files"].items() if sha(ROOT / f) != v["sha256"]]


def pytest_suite(path, extra):
    r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra, str(ROOT / path)], capture_output=True, text=True, cwd=ROOT)
    tail = r.stdout.strip().splitlines()[-1]
    m = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|deselected|skipped|error)", tail)}
    assert r.returncode == 0 and m.get("passed") and not m.get("failed") and not m.get("skipped"), f"{path}:\n{r.stdout[-1500:]}"
    return {"summary_line": tail, **m}


def main() -> None:
    assert not list(W7.rglob("annotator_0*.csv")), "a real label file exists"
    assert not (W7 / "outputs/evaluation_results").exists() and not (W7 / "outputs/subset_package").exists(), "results/subset package already exist"
    for m in PREVIOUSLY_FROZEN:
        assert not drift(m), f"{m}: frozen file(s) changed: {drift(m)}"
    am = json.loads((W7 / "outputs/evaluation/annotation_manifest.json").read_text())
    assert all(sha(W7 / "outputs/evaluation" / n) == h for n, h in am["hashes"].items()), "official annotation package files changed"
    dec = json.loads((MD / "research_decisions.json").read_text())
    assert dec["status"] == "APPROVED_BY_RESEARCH_OWNER"
    suites = [{"title": t, "file": p, **pytest_suite(p, e)} for t, p, e in SUITES]
    import mutation_check_subset as MC
    mut = MC.run()
    assert mut["all_caught"] and mut["files_restored_identical"], mut
    import freeze_evaluator_v2 as F2
    pkg = F2.package_check()
    for m in PREVIOUSLY_FROZEN:                                           # the test runs above must not have changed anything frozen
        assert not drift(m), f"{m}: changed by the test run: {drift(m)}"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    import numpy, sklearn
    env = {"python": sys.version.split()[0], "numpy": numpy.__version__, "scikit-learn": sklearn.__version__, "browser_tests": "Google Chrome headless"}
    total = sum(s["passed"] for s in suites)
    report = {"generated_utc": now, "suites": suites, "total_passed": total, "total_failed": 0, "mutation_check": mut, "official_annotator_package_check": pkg,
              "previously_frozen_manifests_verified_unchanged": PREVIOUSLY_FROZEN, "official_interface_unchanged": True,
              "real_labels_present": False, "private_key_opened": False, "research_metrics_computed": False, "environment": env}
    (MD / "subset_test_report.json").write_text(json.dumps(report, indent=1))
    L = ["# Test report — quy trình chấm tập con (N1–N4)", "", f"Sinh bởi `week7/scripts/freeze_subset_workflow.py` lúc {now}; kết quả chạy thật, không chỉnh tay.", "",
         "| Bộ test | File | Kết quả |", "|---|---|---|"]
    L += [f"| {s['title']} | `{s['file']}` | **PASS** — {s['summary_line']} |" for s in suites]
    L += ["", f"Tổng: **{total} test PASS, 0 FAIL, 0 SKIP** (gồm các test giao diện trong Chrome headless thật).", "", "## Mutation check (thành phần mới)", "", "| Đột biến | Kết quả |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in mut["mutations"].items()]
    L += ["", "Các file bị đột biến được khôi phục giống hệt (kiểm bằng SHA-256).", "", "## Gói annotator chính thức", "",
          f"- {pkg['files_in_zip']} file trong zip = {len(pkg['allowlist'])} file allowlist + {pkg['images']} ảnh; từ gợi ý phương pháp/hạng/điểm/evaluator: {pkg['leak_words_found']}.",
          "- Giao diện chính thức và `annotation_manifest.json` không đổi.", "", "## Bất biến", "",
          "- 4 manifest đã đóng băng (protocol, evaluator v1, evaluator v2, methodology) không đổi hash, **kể cả sau khi chạy toàn bộ test**.",
          "- Không có file nhãn thật, không có `evaluation_results/`, không có `subset_package/` thật, file khóa chưa được mở; không metric nghiên cứu nào được tính.",
          "- Mọi fixture nằm trong thư mục tạm của pytest và là dữ liệu tổng hợp; không fixture nào được dùng như nhãn nghiên cứu.", "",
          "Lưu ý: `test_no_ranking_or_embedding_exists_yet` (vòng 2) loại có chủ đích — test trạng thái ban đầu, không còn áp dụng khi ranking đã tồn tại; file test không bị sửa."]
    (MD / "subset_test_report.md").write_text("\n".join(L) + "\n")
    files = COMPONENTS + ["week7/outputs/methodology/subset_test_report.json", "week7/outputs/methodology/subset_test_report.md"]
    man = {"manifest": "subset_workflow_manifest", "created_utc": now, "state": "READY_FOR_HUMAN_ANNOTATION", "decision_record_status": dec["status"],
           "scope": "tools for second judgments, adjudication, subset interfaces, merge, reporting gate; instructions for annotators 01, 02, 03",
           "files": {f: {"sha256": sha(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in files},
           "previously_frozen_unchanged": {m: sha(ROOT / m) for m in PREVIOUSLY_FROZEN},
           "official_interface_sha256": sha(W7 / "outputs/evaluation/annotation_interface.html"),
           "tests": {"total_passed": total, "mutation_check_all_caught": True}, "environment": env,
           "note": "no human label exists yet; the evaluator and the interfaces have only been exercised on synthetic fixtures in temporary folders"}
    (MD / "subset_workflow_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"suites": [(s["file"], s["summary_line"]) for s in suites], "total_passed": total, "mutations_caught": mut["all_caught"], "state": man["state"]}, indent=1))


if __name__ == "__main__":
    main()
