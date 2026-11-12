#!/usr/bin/env python3
"""Build report figures from the audited archive and Week 3 outputs."""

import csv
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = ROOT / "archive.zip"
INVENTORY = ROOT / "results/week2/FashionRecommender/data_protocol/inventory.csv"
GROUND_TRUTH = ROOT / "results/week2/FashionRecommender/data_protocol/ground_truth_test.csv"
TOP_K = HERE / "ket_qua_goi_y_top_k.csv"
OUT = HERE / "figures"
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def photo(archive, path, size):
    with archive.open(path) as stream:
        image = Image.open(BytesIO(stream.read())).convert("RGB")
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def image_row(archive, paths):
    """Keep the dataset photographs untouched; identification belongs in the caption."""
    slot_width, slot_height, gap, margin = 335, 435, 30, 20
    width = margin * 2 + len(paths) * slot_width + (len(paths) - 1) * gap
    image = Image.new("RGB", (width, slot_height + 2 * margin), "white")
    for index, path in enumerate(paths):
        pic = photo(archive, path, (slot_width, slot_height))
        x = margin + index * (slot_width + gap) + (slot_width - pic.width) // 2
        y = margin + (slot_height - pic.height) // 2
        image.paste(pic, (x, y))
    return image


def dataset_pairs():
    rows = read_csv(GROUND_TRUTH)
    pairs = [("1014", "images/1014_011.jpg", "images/1014_012.jpg"),
             ("1023", "images/1023_032.jpg", "images/1023_031.jpg")]
    for item_id, query, gallery in pairs:
        assert any(row["item_id"] == item_id and
                   row["query_image_path"] == query and
                   row["relevant_gallery_path"] == gallery for row in rows)

    with ZipFile(ARCHIVE) as archive:
        image = image_row(archive, [path for _, query, gallery in pairs for path in (query, gallery)])
    image.save(OUT / "hinh_01_cap_anh_test.png", optimize=True)


def top_five():
    run = [row for row in read_csv(TOP_K) if row["query_item"] == "4901"]
    assert len(run) == 5 and [int(row["rank"]) for row in run] == [1, 2, 3, 4, 5]
    inventory = read_csv(INVENTORY)

    def item_path(item_id):
        matches = [row for row in inventory if row["item_id"] == item_id and row["role"] == "gallery"]
        assert matches, f"Missing gallery image for {item_id}"
        return matches[0]["image_path"]

    with ZipFile(ARCHIVE) as archive:
        image = image_row(archive, [item_path("4901")] +
                          [item_path(row["candidate_item"]) for row in run[:3]])
    image.save(OUT / "hinh_02_top5_chay_thu.png", optimize=True)


def box(draw, xy, lines):
    draw.rectangle(xy, fill="white", outline="black", width=2)
    x1, y1, x2, y2 = xy
    line_height = 36
    first_y = (y1 + y2 - len(lines) * line_height) // 2
    for index, label in enumerate(lines):
        bounds = draw.textbbox((0, 0), label, font=font(23))
        draw.text(((x1 + x2 - (bounds[2] - bounds[0])) // 2,
                   first_y + index * line_height), label, font=font(23), fill="black")


def arrow(draw, start, end):
    draw.line((*start, *end), fill="black", width=2)
    if end[0] > start[0]:
        draw.polygon([(end[0], end[1]), (end[0] - 13, end[1] - 7),
                      (end[0] - 13, end[1] + 7)], fill="black")
    else:
        draw.polygon([(end[0], end[1]), (end[0] + 13, end[1] - 7),
                      (end[0] + 13, end[1] + 7)], fill="black")


def api_flow():
    image = Image.new("RGB", (1600, 420), "white")
    draw = ImageDraw.Draw(image)
    top = [(25, 50, 215, 145), (255, 50, 555, 145),
           (595, 50, 775, 145), (815, 50, 1125, 145),
           (1165, 50, 1485, 145)]
    labels = [["Trình duyệt"], ["GET /api/", "recommendations"],
              ["Xếp hạng", "catalog"], ["PostgreSQL", "Request + Item"],
              ["Trả Top-K", "và requestId"]]
    for bounds, lines in zip(top, labels):
        box(draw, bounds, lines)
    for left, right in zip(top, top[1:]):
        arrow(draw, (left[2], 97), (right[0], 97))
    box(draw, (25, 265, 300, 355), ["Hiển thị / bấm", "sản phẩm"])
    box(draw, (365, 265, 700, 355), ["POST /api/events"])
    box(draw, (770, 265, 1125, 355), ["PostgreSQL", "RecommendationEvent"])
    arrow(draw, (300, 310), (365, 310))
    arrow(draw, (700, 310), (770, 310))
    image.save(OUT / "hinh_03_luong_api.png", optimize=True)


def main():
    OUT.mkdir(exist_ok=True)
    dataset_pairs()
    top_five()
    api_flow()
    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
