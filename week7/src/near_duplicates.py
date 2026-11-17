"""Pre-extraction duplicate screen over the frozen universe (no CNN).
Flags pairs of DIFFERENT product_code whose images are byte-identical or perceptually near-identical.
Thresholds are fixed here BEFORE looking at any result: pHash-style dHash 16x16 (256 bits) Hamming <= 20,
or mean abs pixel difference on 64x64 grayscale <= 6/255 (after aspect-preserving resize+pad).
"""
import csv, json, sys
from pathlib import Path
from PIL import Image
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "week7/outputs/hm"
HAM, MAD = 20, 6.0

def gray(path, n):
    im = Image.open(path).convert("L")
    im.thumbnail((n, n), Image.BICUBIC)
    c = Image.new("L", (n, n), 255); c.paste(im, ((n - im.width) // 2, (n - im.height) // 2))
    return np.asarray(c, dtype=np.float32)

def dhash(path):
    a = np.asarray(Image.open(path).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).flatten()

def main():
    U = list(csv.DictReader(open(OUT / "product_universe.csv", encoding="utf-8")))
    paths = [ROOT / r["image_relpath"] for r in U]
    H = np.stack([dhash(p) for p in paths]); G = np.stack([gray(p, 64).flatten() for p in paths])
    pairs = []
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            if U[i]["product_code"] == U[j]["product_code"]:
                continue
            ham = int((H[i] != H[j]).sum()); mad = float(np.abs(G[i] - G[j]).mean())
            same = U[i]["image_sha256"] == U[j]["image_sha256"]
            if same or ham <= HAM or mad <= MAD:
                pairs.append({"article_a": U[i]["article_id"], "article_b": U[j]["article_id"],
                              "product_code_a": U[i]["product_code"], "product_code_b": U[j]["product_code"],
                              "group_a": U[i]["product_group_name"], "group_b": U[j]["product_group_name"],
                              "type_a": U[i]["product_type_name"], "type_b": U[j]["product_type_name"],
                              "byte_identical": same, "dhash_hamming_of_256": ham, "mean_abs_diff_64px": round(mad, 3)})
    qa = {r["article_id"] for r in csv.DictReader(open(OUT / "query_manifest.csv"))}
    for p in pairs:
        p["involves_query"] = p["article_a"] in qa or p["article_b"] in qa
    flds = list(pairs[0].keys()) if pairs else ["article_a"]
    with open(OUT / "near_duplicate_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds); w.writeheader(); w.writerows(pairs)
    allh = [int((H[i] != H[j]).sum()) for i in range(len(U)) for j in range(i + 1, len(U)) if U[i]["product_code"] != U[j]["product_code"]]
    s = {"universe": len(U), "cross_code_pairs_checked": len(allh), "thresholds": {"dhash_hamming_le": HAM, "mean_abs_diff_le": MAD},
         "flagged_pairs": len(pairs), "byte_identical": sum(p["byte_identical"] for p in pairs),
         "flagged_involving_query": sum(p["involves_query"] for p in pairs),
         "min_hamming_overall": min(allh), "hamming_percentiles_1_5_50": [float(np.percentile(allh, q)) for q in (1, 5, 50)]}
    (OUT / "near_duplicate_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    for p in pairs[:40]: print(p)
main()
