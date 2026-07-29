from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageFilter

from src.datasets.metadata import read_jsonl, write_jsonl
from src.preprocessing.image_io import load_image_as_grayscale


def resize_keep_aspect(img: Image.Image, target_height: int) -> Image.Image:
    w, h = img.size
    if h == target_height:
        return img

    scale = target_height / h
    new_w = max(1, int(round(w * scale)))
    return img.resize((new_w, target_height), Image.Resampling.BICUBIC)


def pad_to_min_width(img: Image.Image, min_width: int, value: int = 255) -> Image.Image:
    w, h = img.size
    if w >= min_width:
        return img

    out = Image.new("L", (min_width, h), value)
    out.paste(img, (0, 0))
    return out


def preprocess_ocr_image(
    image_path: str | Path,
    target_height: int = 64,
    min_width: int = 64,
    autocontrast: bool = False,
    median_denoise: bool = False,
    invert: bool = False,
) -> Image.Image:
    img = load_image_as_grayscale(image_path)

    if invert:
        img = ImageOps.invert(img)

    if autocontrast:
        img = ImageOps.autocontrast(img)

    if median_denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))

    img = resize_keep_aspect(img, target_height=target_height)
    img = pad_to_min_width(img, min_width=min_width, value=255)

    return img


def run_ocr_preprocessing(
    metadata_path: str | Path,
    out_dir: str | Path,
    metadata_out: str | Path,
    target_height: int = 64,
    min_width: int = 64,
    limit: int | None = None,
    autocontrast: bool = False,
    median_denoise: bool = False,
    variant_name: str = "ocr_p0",
) -> None:
    records = read_jsonl(metadata_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_records: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        if limit is not None and idx >= limit:
            break

        image_path = record["image_path"]
        sample_id = record["sample_id"]

        dst = out_dir / f"{sample_id}.png"

        try:
            img = preprocess_ocr_image(
                image_path,
                target_height=target_height,
                min_width=min_width,
                autocontrast=autocontrast,
                median_denoise=median_denoise,
            )
            img.save(dst)
            record["ocr_image_path"] = str(dst)
            record.setdefault("metadata", {})["ocr_preprocess_variant"] = variant_name
        except Exception as exc:
            record.setdefault("metadata", {}).setdefault("quality_flags", [])
            record["metadata"]["quality_flags"] = sorted(
                set(record["metadata"]["quality_flags"] + ["ocr_preprocess_failed"])
            )
            record.setdefault("metadata", {})["ocr_preprocess_error"] = repr(exc)

        processed_records.append(record)

        if (idx + 1) % 5000 == 0:
            print(f"OCR preprocessed {idx + 1}/{len(records)}")

    write_jsonl(processed_records, metadata_out)
    print(f"Wrote OCR-preprocessed metadata: {metadata_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--metadata_out", required=True)
    parser.add_argument("--target_height", type=int, default=64)
    parser.add_argument("--min_width", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--median_denoise", action="store_true")
    parser.add_argument("--variant_name", type=str, default="ocr_p0")

    args = parser.parse_args()

    run_ocr_preprocessing(
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        metadata_out=args.metadata_out,
        target_height=args.target_height,
        min_width=args.min_width,
        limit=args.limit,
        autocontrast=args.autocontrast,
        median_denoise=args.median_denoise,
        variant_name=args.variant_name,
    )


if __name__ == "__main__":
    main()