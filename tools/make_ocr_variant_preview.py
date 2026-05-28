from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw

from src.datasets.metadata import read_jsonl
from src.preprocessing.image_io import load_image_as_grayscale


def fit_to_height(img: Image.Image, height: int) -> Image.Image:
    w, h = img.size
    scale = height / h
    return img.resize((max(1, int(w * scale)), height), Image.Resampling.BICUBIC)


def main() -> None:
    base = read_jsonl("data/processed/cyrillic_handwriting/metadata.final.jsonl")[:1000]
    p0 = read_jsonl("data/processed/cyrillic_handwriting/metadata.ocr_p0_pilot.jsonl")
    p1 = read_jsonl("data/processed/cyrillic_handwriting/metadata.ocr_p1_pilot.jsonl")
    p2 = read_jsonl("data/processed/cyrillic_handwriting/metadata.ocr_p2_pilot.jsonl")

    p0_map = {r["sample_id"]: r for r in p0}
    p1_map = {r["sample_id"]: r for r in p1}
    p2_map = {r["sample_id"]: r for r in p2}

    rng = random.Random(42)
    sample = rng.sample(base, 40)

    row_h = 80
    label_w = 420
    rows = []

    for r in sample:
        sid = r["sample_id"]

        original = fit_to_height(load_image_as_grayscale(r["image_path"]), row_h)
        img_p0 = fit_to_height(load_image_as_grayscale(p0_map[sid]["ocr_image_path"]), row_h)
        img_p1 = fit_to_height(load_image_as_grayscale(p1_map[sid]["ocr_image_path"]), row_h)
        img_p2 = fit_to_height(load_image_as_grayscale(p2_map[sid]["ocr_image_path"]), row_h)

        label = f'{sid}: {r["raw_transcription"]}'
        total_w = label_w + original.width + img_p0.width + img_p1.width + img_p2.width + 60
        row = Image.new("RGB", (total_w, row_h + 32), "white")
        draw = ImageDraw.Draw(row)
        draw.text((5, 5), label[:70], fill="black")
        draw.text((label_w, 5), "original | p0 | p1 | p2", fill="black")

        x = label_w
        for img in [original, img_p0, img_p1, img_p2]:
            row.paste(img.convert("RGB"), (x, 32))
            x += img.width + 10

        rows.append(row)

    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)

    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height

    out = Path("data/reports/cyrillic_handwriting/ocr_variant_preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()