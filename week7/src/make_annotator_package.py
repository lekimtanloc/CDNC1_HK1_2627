"""Build the zip that is sent to annotators from an explicit ALLOWLIST (blind-safe files only).

week7/outputs/evaluation/ also contains evaluator_spec.md / evaluator_manifest.json, which name the methods;
those must never reach annotators, so this script never zips the folder wholesale.

    python3 -I -B week7/src/make_annotator_package.py [--out week7/outputs/annotator_package.zip]
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "outputs/evaluation"
ALLOWED = ["annotation_guide.md", "annotation_interface.html", "annotation_manifest.json", "annotation_tasks.csv",
           "annotation_template.csv", "annotation_template_pilot.csv"]
LEAK = re.compile(r"visual|metadata|cosine|resnet|anchor|top-?5|rank|score|method|baseline|\bcnn\b|embedding|tf-?idf", re.I)


def build(out: Path, evaldir: Path = EVAL) -> list[str]:
    for n in ALLOWED:
        hit = LEAK.findall((evaldir / n).read_text(encoding="utf-8"))
        if hit:
            raise SystemExit(f"{n} contains method-related words {hit[:3]}: refusing to package")
    images = sorted((evaldir / "images").glob("img_*.jpg"))
    if not images:
        raise SystemExit("no images found")
    out.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in ALLOWED:
            z.write(evaldir / n, f"evaluation/{n}"); names.append(f"evaluation/{n}")
        for im in images:
            z.write(im, f"evaluation/images/{im.name}"); names.append(f"evaluation/images/{im.name}")
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EVAL.parent / "annotator_package.zip")
    a = ap.parse_args()
    n = build(a.out)
    print(f"{a.out}: {len(n)} files")
