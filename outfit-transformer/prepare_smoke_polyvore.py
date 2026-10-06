"""Build a tiny schema-compatible Polyvore fixture from NGNN's bundled images.

This fixture is only for an end-to-end execution check.  It is not suitable
for measuring model quality because train/valid/test intentionally reuse the
same tiny synthetic examples.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "NGNN" / "data" / "polyvore-images_smallsample"
OUTPUT = ROOT / "smoke_polyvore"
TEXT_DIM = 8


def main() -> None:
    source_images = sorted(
        path for path in SOURCE.glob("*/*.jpg") if path.stem != "0"
    )
    if len(source_images) < 24:
        raise RuntimeError(f"Need at least 24 item images, found {len(source_images)}")

    image_dir = OUTPUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    outfits = []
    metadata = {}
    descriptions = []
    image_index = 0
    for outfit_number in range(8):
        set_id = f"smoke{outfit_number:02d}"
        items = []
        for item_index in range(1, 4):
            item_id = f"item{image_index:03d}"
            description = f"smoke item {image_index:03d}"
            source = source_images[image_index]
            shutil.copy2(source, image_dir / f"{item_id}.jpg")
            category_id = item_index
            metadata[item_id] = {
                "semantic_category": f"category_{category_id}",
                "category_id": category_id,
                "title": description,
                "url_name": description,
            }
            descriptions.append(description)
            items.append({"item_id": item_id, "index": item_index})
            image_index += 1
        outfits.append({"set_id": set_id, "items": items})

    # Each split contains both classes so AUC can be computed.  Positive rows
    # use one outfit; negative rows combine items from adjacent outfits.
    compatibility = []
    for outfit_number in range(8):
        current = f"smoke{outfit_number:02d}"
        following = f"smoke{(outfit_number + 1) % 8:02d}"
        compatibility.append(f"1 {current}_1 {current}_2 {current}_3")
        compatibility.append(f"0 {current}_1 {following}_2 {current}_3")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (OUTPUT / f"{split}.json").write_text(
            json.dumps(outfits, indent=2), encoding="utf-8"
        )
        (OUTPUT / f"compatibility_{split}.txt").write_text(
            "\n".join(compatibility) + "\n", encoding="utf-8"
        )

    (OUTPUT / "item_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "categories_header.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category_id", "fine"])
        for category_id in range(1, 4):
            writer.writerow([category_id, f"category_{category_id}"])

    vector = "[" + ",".join(["0.0"] * TEXT_DIM) + "]"
    (OUTPUT / "train_bert768.txt").write_text(
        "\n".join(f"{description},{vector}" for description in descriptions) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "purpose": "execution-only smoke fixture; metrics are not scientific",
        "source": str(SOURCE),
        "outfits": len(outfits),
        "items": len(metadata),
        "compatibility_rows_per_split": len(compatibility),
        "text_feature_dim": TEXT_DIM,
    }
    (OUTPUT / "SMOKE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
