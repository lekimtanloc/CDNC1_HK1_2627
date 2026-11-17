"""Week 7 round 2: H&M dataset audit, candidate universe, query/pilot manifests, checksums.

Read-only on all inputs. Runs no CNN and no ranking.
Usage (from repo root CDNC1_HK1_2627):
    python3 -I week7/scripts/build_hm_protocol.py data      # audit + manifests
    python3 -I week7/scripts/build_hm_protocol.py manifest  # checksums (after protocol md is final)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
OUT = W7 / "outputs" / "hm"
EXP = json.loads((W7 / "config" / "experiment.json").read_text())
EVAL = json.loads((W7 / "config" / "evaluation.json").read_text())
META_FIELDS = [f for f in EXP["methods"]["M-ATTR-TFIDF"]["weights"] if f != "detail_desc_tfidf"]
AUDIT_FIELDS = ["product_type_no", "product_type_name", "product_group_name", "graphical_appearance_no",
                "graphical_appearance_name", "colour_group_code", "colour_group_name", "perceived_colour_value_id",
                "perceived_colour_value_name", "perceived_colour_master_id", "perceived_colour_master_name",
                "department_no", "index_code", "index_group_no", "index_group_name", "section_no",
                "garment_group_no", "garment_group_name", "detail_desc", "prod_name"]
sys.path.insert(0, str(Path(__file__).parent))
from protocol_lib import is_missing  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def p(rel: str) -> Path:
    return (ROOT / rel).resolve()


def check_image(path: Path) -> tuple[bool, str, tuple[int, int] | None]:
    if not path.is_file():
        return False, "missing", None
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            size = im.size
    except Exception as e:  # noqa: BLE001
        return False, f"decode_error:{type(e).__name__}", None
    if min(size) < 300:
        return False, "too_small", size
    return True, "ok", size


def allocate(sizes: dict[str, int], total: int, floor: int) -> dict[str, int]:
    alloc = {g: min(floor, n) for g, n in sizes.items()}
    rest = total - sum(alloc.values())
    tot = sum(sizes.values())
    quota = {g: rest * sizes[g] / tot for g in sizes}
    for g in sizes:
        alloc[g] += int(quota[g])
    left = total - sum(alloc.values())
    for g in sorted(sizes, key=lambda g: (-(quota[g] - int(quota[g])), g))[:left]:
        alloc[g] += 1
    assert all(alloc[g] <= sizes[g] for g in sizes)
    return alloc


def stage_data() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = EXP["data"]
    articles = {}
    with p(d["articles_csv"]).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            articles[r["article_id"]] = r
    catalog = json.loads(p(d["catalog_manifest"]).read_text())
    cat_ids = [c["articleId"] for c in catalog]
    pub = p(d["catalog_public_images"])

    audit: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "inputs": {k: {"path": d[k], "sha256": sha(p(d[k]))} for k in ("articles_csv", "catalog_manifest")}}
    audit["articles_csv"] = {"rows": len(articles), "unique_article_id": len(set(articles)),
                             "article_id_all_10_digit_str": all(len(a) == 10 and a.isdigit() for a in articles)}
    rows, img_status, img_sha = [], Counter(), {}
    for a in cat_ids:
        r = articles.get(a)
        src = p(d["image_dir"]) / a[:3] / f"{a}.jpg"
        ok, why, size = check_image(src)
        same = ok and (pub / f"{a}.jpg").is_file() and sha(src) == sha(pub / f"{a}.jpg")
        img_status[why] += 1
        if ok:
            img_sha[a] = sha(src)
        rows.append({"a": a, "r": r, "ok": ok and r is not None and same, "why": why, "size": size, "same": same})
    valid = [x for x in rows if x["ok"]]
    audit["catalog"] = {
        "entries": len(cat_ids), "unique_articleId": len(set(cat_ids)),
        "articleId_in_articles_csv": sum(x["r"] is not None for x in rows),
        "image_check": dict(img_status), "public_copy_identical": sum(x["same"] for x in rows),
        "valid_articles": len(valid),
        "selection_mechanism": "fashion-store-main/scripts/prepare_hm_catalog.py: first rows of articles.csv (file order) per product_group that pass keyword/image-border filters, quota 178/110/70/80/22/40; CLIP not used (packshotScore null for 500/500). NOT a random sample.",
    }
    vr = [x["r"] for x in valid]
    audit["catalog"]["missing_like_values"] = {f: sum(is_missing(r[f]) for r in vr) for f in AUDIT_FIELDS}
    audit["catalog"]["distributions_articles"] = {f: dict(Counter(r[f] for r in vr).most_common())
                                                   for f in ("product_group_name", "product_type_name", "colour_group_name",
                                                             "graphical_appearance_name", "index_group_name", "garment_group_name")}
    dup_img = defaultdict(list)
    for a, h in img_sha.items():
        dup_img[h].append(a)
    pc = Counter(r["product_code"] for r in vr)
    pcc = Counter((r["product_code"], r["colour_group_code"]) for r in vr)
    audit["duplicates"] = {
        "byte_identical_images": [v for v in dup_img.values() if len(v) > 1],
        "distinct_product_code": len(pc),
        "product_codes_with_multiple_articles": sum(v > 1 for v in pc.values()),
        "largest_product_code_groups": pc.most_common(10),
        "rows_duplicating_product_code_and_colour": sum(v - 1 for v in pcc.values()),
        "detail_desc_constant_within_product_code": sum(len({r["detail_desc"] for r in vr if r["product_code"] == c}) == 1 for c in pc),
        "product_codes_spanning_two_product_groups": sorted(c for c in pc if len({r["product_group_name"] for r in vr if r["product_code"] == c}) > 1),
        "same_prod_name_across_different_product_code": sorted(n for n, cs in
            ((n, {r["product_code"] for r in vr if r["prod_name"] == n}) for n in {r["prod_name"] for r in vr}) if len(cs) > 1),
        "decision": "unit = product_code; one representative per code (smallest sha256('42:rep:'+article_id)); other articles of the code are dropped from both query and candidate sets",
    }
    reps = {}  # representative = article with the smallest sha256('42:rep:'+article_id) in its product_code
    rep_key = lambda x: hashlib.sha256(f"{EXP['seed']}:rep:{x['a']}".encode()).hexdigest()
    for x in sorted(valid, key=rep_key):
        reps.setdefault(x["r"]["product_code"], x)
    universe = sorted(reps.values(), key=lambda x: x["a"])
    U = [x["r"] for x in universe]
    audit["universe"] = {
        "products": len(U),
        "per_product_group": dict(Counter(r["product_group_name"] for r in U).most_common()),
        "per_product_type": dict(Counter(r["product_type_name"] for r in U).most_common()),
        "per_colour_group": dict(Counter(r["colour_group_name"] for r in U).most_common()),
        "representative_rule_revision": "v0 'lowest article_id' rejected before freezing: it selected Black for 84/145 products (58%) vs 138/500 articles (28%) because first variants are usually black. Decided from metadata only; no ranking or label existed.",
        "per_index_group": dict(Counter(r["index_group_name"] for r in U).most_common()),
        "missing_like_values_in_metadata_fields": {f: sum(is_missing(r[f]) for r in U) for f in META_FIELDS + ["detail_desc"]},
        "candidate_pool_per_query": len(U) - 1,
        "byte_identical_images_within_universe": sum(1 for v in Counter(img_sha[x["a"]] for x in universe).values() if v > 1),
    }
    ufields = ["article_id", "product_code", "prod_name", "product_group_name", "product_type_no", "product_type_name",
               "garment_group_no", "colour_group_code", "colour_group_name", "perceived_colour_master_id",
               "perceived_colour_value_id", "graphical_appearance_no", "index_group_no", "image_relpath",
               "image_sha256", "width", "height", "n_articles_in_code"]
    with (OUT / "product_universe.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ufields)
        w.writeheader()
        for x in universe:
            r = x["r"]
            w.writerow({**{k: r[k] for k in ufields if k in r}, "image_relpath": f"{d['image_dir']}/{x['a'][:3]}/{x['a']}.jpg",
                        "image_sha256": img_sha[x["a"]], "width": x["size"][0], "height": x["size"][1],
                        "n_articles_in_code": pc[r["product_code"]]})

    qs = EXP["query_sampling"]
    by_group = defaultdict(list)
    for r in U:
        by_group[r["product_group_name"]].append(r["article_id"])
    sizes = {g: len(v) for g, v in by_group.items()}
    n_q = min(qs["max_queries"], len(U))
    alloc = allocate(sizes, n_q, 5)
    rng = random.Random(EXP["seed"])
    queries = []
    for g in sorted(by_group):
        for a in sorted(rng.sample(sorted(by_group[g]), alloc[g])):
            queries.append((g, a))
    rest = sorted(set(r["article_id"] for r in U) - {a for _, a in queries})
    pilot = sorted(random.Random(qs["pilot"]["seed"]).sample(rest, qs["pilot"]["n"]))
    rowsU = {r["article_id"]: r for r in U}
    qfields = ["query_no", "article_id", "product_code", "product_group_name", "product_type_name", "colour_group_name",
               "image_relpath", "image_sha256", "stratum_size", "stratum_queries", "inclusion_prob"]
    with (OUT / "query_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qfields)
        w.writeheader()
        for i, (g, a) in enumerate(queries, 1):
            r = rowsU[a]
            w.writerow({"query_no": i, "article_id": a, "product_code": r["product_code"], "product_group_name": g,
                        "product_type_name": r["product_type_name"], "colour_group_name": r["colour_group_name"],
                        "image_relpath": f"{d['image_dir']}/{a[:3]}/{a}.jpg", "image_sha256": img_sha[a],
                        "stratum_size": sizes[g], "stratum_queries": alloc[g], "inclusion_prob": round(alloc[g] / sizes[g], 6)})
    with (OUT / "pilot_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pilot_no", "article_id", "product_code", "product_group_name", "image_sha256"])
        w.writeheader()
        for i, a in enumerate(pilot, 1):
            w.writerow({"pilot_no": i, "article_id": a, "product_code": rowsU[a]["product_code"],
                        "product_group_name": rowsU[a]["product_group_name"], "image_sha256": img_sha[a]})
    audit["queries"] = {"n": len(queries), "allocation": alloc, "stratum_sizes": sizes,
                        "per_product_type": dict(Counter(rowsU[a]["product_type_name"] for _, a in queries).most_common()),
                        "per_colour_group": dict(Counter(rowsU[a]["colour_group_name"] for _, a in queries).most_common()),
                        "pilot": pilot,
                        "max_annotation_pairs": len(queries) * 12}
    audit["feasibility"] = {
        "enough_products": len(U) >= 100, "enough_queries": len(queries) == qs["max_queries"],
        "smallest_stratum_candidates_same_group": min(sizes.values()) - 1,
        "status": "FEASIBLE_WITH_LIMITATIONS" if len(U) >= 100 else "INSUFFICIENT",
    }
    (OUT / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: audit[k] for k in ("universe", "queries", "feasibility")}, ensure_ascii=False, indent=1)[:4000])


def stage_manifest() -> None:
    files = ["week7/config/experiment.json", "week7/config/evaluation.json", "week7/outputs/hm/evaluation_protocol.md",
             "week7/outputs/hm/query_manifest.csv", "week7/outputs/hm/pilot_manifest.csv",
             "week7/outputs/hm/product_universe.csv", "week7/outputs/hm/dataset_audit.json",
             "week7/scripts/build_hm_protocol.py", "week7/scripts/protocol_lib.py", "week7/tests/test_protocol.py"]
    m = {"experiment_id": EXP["experiment_id"], "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "state": "FROZEN_DESIGN_NOT_RUN",
         "rule": "any change to a file below after freezing requires a new protocol version and a logged reason; no ranking has been computed at freeze time",
         "files": {f: {"sha256": sha(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in files},
         "inputs": {EXP["data"][k]: sha(p(EXP["data"][k])) for k in ("articles_csv", "catalog_manifest")}}
    (OUT / "protocol_manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    {"data": stage_data, "manifest": stage_manifest}[sys.argv[1]]()
