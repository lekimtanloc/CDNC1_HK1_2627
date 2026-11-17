"""Week 7: build the blind annotation package from the frozen protocol.

Pool per query (evaluation.json -> judgment_pool): union of Top-5 visual and Top-5 metadata (deduplicated)
+ 2 anchors = first 2 candidates of (pool - union) ordered by sha256('42:anchor:'+query+':'+candidate).
Main set = the 50 official queries; pilot set = the 5 pilot queries (kept in a separate set, never merged).
Pilot Top-5 lists are computed with the same code/cache (visual_top5.csv of the official run is NOT touched).

Blind-safety: everything written to week7/outputs/evaluation/ is free of method, rank, score and anchor
information. The key linking pair_id -> method/rank/anchor goes to week7/outputs/evaluation_private/.
No labels are created here.

Run (repo root):  python3 -I -B week7/src/prepare_annotations.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import extract_embeddings as X  # noqa: E402
import metadata_baseline as M  # noqa: E402
import visual_retrieval as V  # noqa: E402

ROOT, OUT = X.ROOT, X.OUT
EVALDIR = ROOT / "week7/outputs/evaluation"
PRIVATE = ROOT / "week7/outputs/evaluation_private"
SRC = Path(__file__).parent
N_ANCHOR, MAX_PER_QUERY, MAX_TOTAL = 2, 12, 600
EVAL = json.loads((ROOT / "week7/config/evaluation.json").read_text())


def sha_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def pair_id(set_: str, q: str, c: str) -> str:
    return "P" + sha_str(f"42:pair:{set_}:{q}:{c}")[:10]


def img_name(article_id: str) -> str:
    return f"img_{sha_str('42:img:' + article_id)[:12]}.jpg"


def visual_for(qids: list[str], rows: list[dict]) -> dict[str, list[str]]:
    man = json.loads((OUT / "embeddings_manifest.json").read_text())
    emb = np.load(ROOT / man["cache_file"])
    assert X.sha_file(ROOT / man["cache_file"]) == man["cache_file_sha256"], "embedding cache changed"
    ids = man["article_ids_order"]
    assert ids == [r["article_id"] for r in rows]
    return {q: [c for c, _ in V.top_k(q, ids, emb)] for q in qids}


def build_set(set_: str, queries: list[dict], vis: dict, meta: dict, universe_ids: list[str]):
    pairs, key = [], []
    for q in queries:
        qid = q["article_id"]
        v, m = vis[qid], meta[qid]
        union = list(dict.fromkeys(v + m))
        rest = sorted((c for c in universe_ids if c != qid and c not in union),
                      key=lambda c: sha_str(f"42:anchor:{qid}:{c}"))
        anchors = rest[:N_ANCHOR]
        cands = union + anchors
        assert qid not in cands and len(cands) == len(set(cands)) and len(cands) <= MAX_PER_QUERY
        for c in cands:
            pid = pair_id(set_, qid, c)
            pairs.append({"pair_id": pid, "set": set_, "query_id": qid, "candidate_id": c})
            key.append({"pair_id": pid, "set": set_, "query_id": qid, "candidate_id": c,
                        "in_visual_top5": c in v, "visual_rank": v.index(c) + 1 if c in v else "",
                        "in_metadata_top5": c in m, "metadata_rank": m.index(c) + 1 if c in m else "",
                        "is_anchor": c in anchors})
    return pairs, key


def main() -> None:
    man = X.verify_protocol_manifest()
    rows = M.load_universe_with_desc()
    universe_ids = [r["article_id"] for r in rows]
    by_id = {r["article_id"]: r for r in rows}
    main_q = list(csv.DictReader(open(OUT / "query_manifest.csv", newline="", encoding="utf-8")))
    pilot_q = list(csv.DictReader(open(OUT / "pilot_manifest.csv", newline="", encoding="utf-8")))
    for p in pilot_q:
        p["article_id"] = p["article_id"]
    assert len(main_q) == 50 and len(pilot_q) == 5
    assert not ({q["article_id"] for q in main_q} & {p["article_id"] for p in pilot_q}), "pilot must be disjoint from main"

    # ---- main set: reuse the already-produced official Top-5 files verbatim
    vis_main, meta_main = {}, {}
    for x in csv.DictReader(open(OUT / "visual_top5.csv", newline="", encoding="utf-8")):
        vis_main.setdefault(x["query_id"], []).append((int(x["rank"]), x["candidate_id"]))
    for x in csv.DictReader(open(OUT / "metadata_top5.csv", newline="", encoding="utf-8")):
        meta_main.setdefault(x["query_id"], []).append((int(x["rank"]), x["candidate_id"]))
    vis_main = {q: [c for _, c in sorted(v)] for q, v in vis_main.items()}
    meta_main = {q: [c for _, c in sorted(v)] for q, v in meta_main.items()}

    # ---- pilot set: same methods, same code, written to a separate folder
    pq = [p["article_id"] for p in pilot_q]
    vis_p = visual_for(pq, rows)
    mres = M.compute(pq, rows)
    meta_p = {q: [s["cand"]["article_id"] for s in mres["top"][q]] for q in pq}
    pdir = OUT / "pilot"
    pdir.mkdir(exist_ok=True)
    with open(pdir / "metadata_top5_pilot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=M.FIELDS); w.writeheader()
        w.writerows(M.to_rows([{"article_id": q, "product_code": by_id[q]["product_code"]} for q in pq], mres["top"], by_id))
    with open(pdir / "visual_top5_pilot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["query_id", "candidate_id", "rank"])
        for q in pq:
            for i, c in enumerate(vis_p[q], 1):
                w.writerow([q, c, i])

    pairs_m, key_m = build_set("main", main_q, vis_main, meta_main, universe_ids)
    pairs_p, key_p = build_set("pilot", pilot_q, vis_p, meta_p, universe_ids)
    assert len(pairs_m) <= MAX_TOTAL, len(pairs_m)
    all_ids = [p["pair_id"] for p in pairs_m + pairs_p]
    assert len(all_ids) == len(set(all_ids)), "pair_id collision"
    assert not ({(p["query_id"], p["candidate_id"]) for p in pairs_m} & {(p["query_id"], p["candidate_id"]) for p in pairs_p})

    # ---- stable per-annotator orders: random.Random(42 + annotator_index), per set, over pair_ids sorted
    orders = {}
    for s, pp in (("main", pairs_m), ("pilot", pairs_p)):
        ids = sorted(p["pair_id"] for p in pp)
        for i in (1, 2, 3):
            lst = ids[:]
            random.Random(42 + i).shuffle(lst)
            for pos, pid in enumerate(lst):
                orders.setdefault(pid, {})[i] = pos

    # ---- images: neutral-name derived copies (max side 900 px); originals untouched
    imgdir = EVALDIR / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    need = sorted({p["query_id"] for p in pairs_m + pairs_p} | {p["candidate_id"] for p in pairs_m + pairs_p})
    for a in need:
        dst = imgdir / img_name(a)
        if not dst.exists():
            with Image.open(ROOT / by_id[a]["image_relpath"]) as im:
                im = im.convert("RGB"); im.thumbnail((900, 900), Image.LANCZOS); im.save(dst, quality=90, optimize=True)

    # ---- tasks (sorted by pair_id => order carries no information), template, key
    tasks = sorted(pairs_m + pairs_p, key=lambda p: (p["set"], p["pair_id"]))
    with open(EVALDIR / "annotation_tasks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "set", "query_image", "candidate_image", "order_annotator_01", "order_annotator_02", "order_annotator_03"])
        for p in tasks:
            w.writerow([p["pair_id"], p["set"], f"images/{img_name(p['query_id'])}", f"images/{img_name(p['candidate_id'])}",
                        *[orders[p["pair_id"]][i] for i in (1, 2, 3)]])
    heads = ["pair_id", "set", "annotator_id", "label", "uncertain", "note", "timestamp_utc"]
    for fname, pp in (("annotation_template.csv", pairs_m), ("annotation_template_pilot.csv", pairs_p)):
        with open(EVALDIR / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(heads)
            for p in sorted(pp, key=lambda p: p["pair_id"]):
                w.writerow([p["pair_id"], p["set"], "", "", "", "", ""])
    PRIVATE.mkdir(exist_ok=True)
    kf = ["pair_id", "set", "query_id", "candidate_id", "in_visual_top5", "visual_rank", "in_metadata_top5", "metadata_rank", "is_anchor"]
    with open(PRIVATE / "annotation_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kf); w.writeheader(); w.writerows(sorted(key_m + key_p, key=lambda k: (k["set"], k["pair_id"])))
    (PRIVATE / "README.txt").write_text("DO NOT OPEN before all annotation CSV files are locked (hashed) and validated.\n"
                                        "annotation_key.csv links pair_id to method, rank and anchor status.\n")

    # ---- HTML (tasks embedded: works from file://, offline)
    js = [{"p": p["pair_id"], "s": p["set"], "q": f"images/{img_name(p['query_id'])}", "c": f"images/{img_name(p['candidate_id'])}",
           "o": [orders[p["pair_id"]][i] for i in (1, 2, 3)]} for p in tasks]
    tpl = (SRC / "annotation_interface_template.html").read_text(encoding="utf-8")
    (EVALDIR / "annotation_interface.html").write_text(tpl.replace("__TASKS_JSON__", json.dumps(js, separators=(",", ":"))), encoding="utf-8")

    per_q = {}
    for p in pairs_m:
        per_q[p["query_id"]] = per_q.get(p["query_id"], 0) + 1
    # PUBLIC manifest (shipped with the annotator folder): no method/pool/rank information at all
    manifest = {
        "experiment_id": EVAL["experiment_id"], "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": "AWAITING_HUMAN_LABELS", "labels_present": False,
        "sets": {"main": {"queries": len(main_q), "pairs": len(pairs_m), "max_pairs_per_query": max(per_q.values()),
                          "min_pairs_per_query": min(per_q.values())},
                 "pilot": {"queries": len(pilot_q), "pairs": len(pairs_p), "note": "training only; never merged into results"}},
        "rules": {"pair_unit": "(query article_id, candidate article_id); product identity = product_code",
                  "max_pairs_per_query": MAX_PER_QUERY, "max_pairs_total": MAX_TOTAL,
                  "pair_id": "'P' + sha256('42:pair:<set>:<query>:<candidate>')[:10]",
                  "image_names": "img_ + sha256('42:img:<article_id>')[:12]; derived copies, longest side <= 900 px",
                  "order": "random.Random(42 + annotator_index).shuffle over sorted pair_ids, per set; annotator_01..03"},
        "label_values": ["0", "1", "2", "U (cannot identify -> missing, never 0)"],
        "required_csv_columns": heads,
        "hashes": {n: X.sha_file(EVALDIR / n) for n in ("annotation_tasks.csv", "annotation_template.csv", "annotation_template_pilot.csv", "annotation_interface.html", "annotation_guide.md")},
        "images": {"count": len(need), "dir": "images/"},
    }
    (EVALDIR / "annotation_manifest.json").write_text(json.dumps(manifest, indent=1))
    # PRIVATE manifest: how the pool was built and which inputs it came from (open only after labels are locked)
    (PRIVATE / "pool_manifest.json").write_text(json.dumps({
        "pool": "union(Top-5 visual, Top-5 metadata) deduplicated + 2 anchors (first 2 of pool-minus-union by sha256('42:anchor:q:c'))",
        "protocol_manifest_frozen_utc": man["frozen_utc"], "annotation_key_sha256": X.sha_file(PRIVATE / "annotation_key.csv"),
        "public_tasks_sha256": manifest["hashes"]["annotation_tasks.csv"],
        "inputs_sha256": {"query_manifest.csv": X.sha_file(OUT / "query_manifest.csv"), "pilot_manifest.csv": X.sha_file(OUT / "pilot_manifest.csv"),
                          "product_universe.csv": X.sha_file(OUT / "product_universe.csv"), "visual_top5.csv": X.sha_file(OUT / "visual_top5.csv"),
                          "metadata_top5.csv": X.sha_file(OUT / "metadata_top5.csv")}}, indent=1))
    print(json.dumps({"main_pairs": len(pairs_m), "pilot_pairs": len(pairs_p), "main_per_query": [min(per_q.values()), max(per_q.values())],
                      "images": len(need)}, indent=1))


if __name__ == "__main__":
    main()
