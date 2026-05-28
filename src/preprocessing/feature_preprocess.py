from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import ImageOps, ImageFilter

from src.datasets.metadata import read_jsonl, write_jsonl
from src.preprocessing.image_io import load_image_as_grayscale


def preprocess_feature_image(
    image_path: str | Path,
    autocontrast: bool = False,
    weak_denoise: bool = False,
) -> Any:
    """
    Feature-preserving preprocessing.

    Важно:
    - не меняем размер по умолчанию;
    - не делаем deskew;
    - не делаем жёсткую бинаризацию;
    - сохраняем grayscale;
    - стараемся не уничтожать stroke width.
    """
    img = load_image_as_grayscale(image_path)

    if autocontrast:
        # Для feature branch по умолчанию выключено.
        img = ImageOps.autocontrast(img)

    if weak_denoise:
        # Осторожно: median может съесть тонкие элементы.
        img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


def run_feature_preprocessing(
    metadata_path: str | Path,
    out_dir: str | Path,
    metadata_out: str | Path,
    limit: int | None = None,
    autocontrast: bool = False,
    weak_denoise: bool = False,
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
            img = preprocess_feature_image(
                image_path,
                autocontrast=autocontrast,
                weak_denoise=weak_denoise,
            )
            img.save(dst)
            record["feature_image_path"] = str(dst)
        except Exception as exc:
            record.setdefault("metadata", {}).setdefault("quality_flags", [])
            record["metadata"]["quality_flags"] = sorted(
                set(record["metadata"]["quality_flags"] + ["feature_preprocess_failed"])
            )
            record.setdefault("metadata", {})["feature_preprocess_error"] = repr(exc)

        processed_records.append(record)

        if (idx + 1) % 5000 == 0:
            print(f"Feature preprocessed {idx + 1}/{len(records)}")

    write_jsonl(processed_records, metadata_out)
    print(f"Wrote feature-preprocessed metadata: {metadata_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--metadata_out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--weak_denoise", action="store_true")
    args = parser.parse_args()

    run_feature_preprocessing(
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        metadata_out=args.metadata_out,
        limit=args.limit,
        autocontrast=args.autocontrast,
        weak_denoise=args.weak_denoise,
    )


if __name__ == "__main__":
    main()