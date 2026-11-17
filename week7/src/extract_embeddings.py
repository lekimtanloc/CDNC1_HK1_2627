"""Week 7: ResNet-50 embeddings for the frozen H&M universe (config V-R50-FULLPAD).

Implements exactly week7/config/experiment.json -> methods.V-R50-FULLPAD.
No fine-tuning, no masks, no cropping. CPU, 4 threads, deterministic, batch size 1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "week7"
OUT = W7 / "outputs" / "hm"
CACHE_DIR = OUT / "embeddings"

MODEL_VERSION = "resnet50-IMAGENET1K_V2-11ad3fa6-fcIdentity"
PREPROCESSING_VERSION = "pp-fullpad-v1"
WEIGHTS_FILE = Path.home() / ".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
PREPROCESSING_SPEC = {
    "version": PREPROCESSING_VERSION,
    "color": "RGB",
    "resize": "longer side = 224, aspect kept, PIL BICUBIC, size = max(1, round(side * 224 / longer))",
    "pad": "224x224, resized image centred at ((224-w)//2, (224-h)//2)",
    "pad_fill": "per-channel mean of the union of 4 px strips (2% of 224, rounded) on the four sides of the resized image, rounded to int",
    "border_strip_px": 4,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
}
MIN_SIDE = 300
EMBED_DIM = 2048


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_protocol_manifest() -> dict:
    """Abort if any frozen protocol file changed (test_protocol.py excluded: it is a pre-run guard)."""
    man = json.loads((OUT / "protocol_manifest.json").read_text())
    bad = [f for f, v in man["files"].items() if not f.endswith("test_protocol.py") and sha_file(ROOT / f) != v["sha256"]]
    if bad:
        raise RuntimeError(f"Frozen protocol files changed -> BLOCKED, create a new protocol version: {bad}")
    return man


def load_universe() -> list[dict]:
    with open(OUT / "product_universe.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_image(row: dict) -> tuple[bool, str]:
    """Publicly stated rule: missing / sha mismatch / decode error / min side < 300 -> excluded and logged."""
    expected = f"dataset_sources/hm-dataset/images/{row['article_id'][:3]}/{row['article_id']}.jpg"
    if row["image_relpath"] != expected:
        return False, "id_path_mismatch"
    p = ROOT / row["image_relpath"]
    if not p.is_file():
        return False, "missing_file"
    if sha_file(p) != row["image_sha256"]:
        return False, "sha256_mismatch"
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            if min(im.convert("RGB").size) < MIN_SIDE:
                return False, "too_small"
    except Exception as e:  # noqa: BLE001
        return False, f"decode_error:{type(e).__name__}"
    return True, "ok"


def preprocess_pil(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    s = 224 / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    a = np.asarray(im)
    t = PREPROCESSING_SPEC["border_strip_px"]
    m = np.zeros(a.shape[:2], dtype=bool)
    m[:t, :] = m[-t:, :] = m[:, :t] = m[:, -t:] = True
    fill = tuple(int(round(float(c))) for c in a[m].reshape(-1, 3).mean(axis=0))
    canvas = Image.new("RGB", (224, 224), fill)
    canvas.paste(im, ((224 - nw) // 2, (224 - nh) // 2))
    return canvas


def to_tensor(im: Image.Image):
    import torch
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(PREPROCESSING_SPEC["normalize_mean"]).view(3, 1, 1)
    std = torch.tensor(PREPROCESSING_SPEC["normalize_std"]).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def l2_normalize(e: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    e = e.astype(np.float32)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), eps).astype(np.float32)


def cache_key(universe_sha: str, weights_sha: str, spec: dict | None = None, model_version: str = MODEL_VERSION) -> str:
    payload = {"model": model_version, "weights_sha256": weights_sha, "preprocessing": spec or PREPROCESSING_SPEC,
               "dataset_universe_sha256": universe_sha, "dim": EMBED_DIM}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_model():
    import torch
    import torchvision
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    model = torchvision.models.resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    return model.eval()


def extract(rows: list[dict], model=None) -> np.ndarray:
    import torch
    model = model or load_model()
    out = []
    with torch.inference_mode():
        for r in rows:
            with Image.open(ROOT / r["image_relpath"]) as im:
                out.append(model(to_tensor(preprocess_pil(im))).numpy()[0])
    return np.stack(out).astype(np.float32)


def run(cache_dir: Path = CACHE_DIR, universe_csv: Path | None = None, limit: int | None = None) -> dict:
    """Return {'rows', 'emb' (L2), 'status', 'key', ...}. Reuses a cache only when its key matches exactly."""
    import torch
    t0 = time.time()
    verify_protocol_manifest()
    universe_csv = universe_csv or (OUT / "product_universe.csv")
    rows_all = load_universe() if universe_csv == OUT / "product_universe.csv" else list(csv.DictReader(open(universe_csv, encoding="utf-8")))
    if limit:
        rows_all = rows_all[:limit]
    checks = [(r, *check_image(r)) for r in rows_all]
    valid = [r for r, ok, _ in checks if ok]
    excluded = [{"article_id": r["article_id"], "reason": why} for r, ok, why in checks if not ok]
    usha = sha_file(universe_csv) if not limit else hashlib.sha256(json.dumps([r["image_sha256"] for r in rows_all]).encode()).hexdigest()
    wsha = sha_file(WEIGHTS_FILE)
    assert wsha.startswith("11ad3fa6"), "weights file is not IMAGENET1K_V2 resnet50"
    key = cache_key(usha, wsha)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy, meta_p = cache_dir / f"emb_{key[:16]}.npy", cache_dir / f"emb_{key[:16]}.json"
    status, t_extract = "recomputed", 0.0
    if npy.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("cache_key") == key and meta.get("article_ids") == [r["article_id"] for r in valid] \
                and meta.get("npy_sha256") == sha_file(npy):
            emb, status = np.load(npy), "cache_hit"
    if status != "cache_hit":
        t1 = time.time()
        emb = l2_normalize(extract(valid))
        t_extract = time.time() - t1
        np.save(npy, emb)
        meta_p.write_text(json.dumps({"cache_key": key, "model_version": MODEL_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
                                      "weights_sha256": wsha, "dataset_universe_sha256": usha, "dim": EMBED_DIM,
                                      "article_ids": [r["article_id"] for r in valid], "npy_sha256": sha_file(npy),
                                      "torch": torch.__version__}, indent=1))
    return {"rows": valid, "emb": emb, "status": status, "key": key, "npy": npy, "excluded": excluded,
            "n_input": len(rows_all), "t_extract_s": t_extract, "t_total_s": time.time() - t0, "weights_sha256": wsha,
            "universe_sha256": usha, "python": platform.python_version()}
