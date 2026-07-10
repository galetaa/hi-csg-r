from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from src.datasets.metadata import read_jsonl
from src.preprocessing.image_io import load_image_as_grayscale


def fit_to_height(img: Image.Image, height: int) -> Image.Image:
    w, h = img.size
    scale = height / h
    return img.resize((max(1, int(w * scale)), height), Image.Resampling.BICUBIC)


def make_preview(
    metadata_path: str,
    out_path: str,
    n: int = 30,
    seed: int = 42,
    row_height: int = 96,
) -> None:
    records = read_jsonl(metadata_path)
    records = [r for r in records if r.get("ocr_image_path") and r.get("feature_image_path")]

    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))

    rows = []

    for r in sample:
        original = fit_to_height(load_image_as_grayscale(r["image_path"]), row_height)
        ocr = fit_to_height(load_image_as_grayscale(r["ocr_image_path"]), row_height)
        feature = fit_to_height(load_image_as_grayscale(r["feature_image_path"]), row_height)

        label_text = f'{r["sample_id"]}: {r["raw_transcription"]}'
        label_w = 420

        total_w = label_w + original.width + ocr.width + feature.width + 40
        row = Image.new("RGB", (total_w, row_height + 30), "white")
        draw = ImageDraw.Draw(row)

        draw.text((5, 5), label_text[:70], fill="black")

        x = label_w
        row.paste(original.convert("RGB"), (x, 30))
        x += original.width + 10
        row.paste(ocr.convert("RGB"), (x, 30))
        x += ocr.width + 10
        row.paste(feature.convert("RGB"), (x, 30))

        rows.append(row)

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)

    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"Wrote preview: {out_path}")


if __name__ == "__main__":
    make_preview(
        metadata_path="data/processed/cyrillic_handwriting/metadata.preprocessed_pilot.jsonl",
        out_path="data/reports/cyrillic_handwriting/preprocessing_preview_pilot.png",
        n=40,
    )