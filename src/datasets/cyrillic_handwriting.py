from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Literal

from src.datasets.metadata import SampleMetadata, write_jsonl
from src.datasets.text_normalization import normalize_text_ru


DATASET_NAME = "cyrillic_handwriting"


def _safe_symlink_or_copy(src: Path, dst: Path, mode: Literal["symlink", "copy", "none"]) -> str:
    """
    Возвращает путь, который нужно записать в metadata.image_path.

    mode:
      - symlink: создать symlink в processed/images
      - copy: физически скопировать файл
      - none: ничего не создавать, оставить путь на исходное изображение
    """
    src = src.resolve()

    if mode == "none":
        return str(src)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return str(dst)

    if mode == "symlink":
        dst.symlink_to(src)
        return str(dst)

    if mode == "copy":
        shutil.copy2(src, dst)
        return str(dst)

    raise ValueError(f"Unknown link mode: {mode}")


def _read_tsv(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")

        for line_idx, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) < 2:
                raise ValueError(f"Bad TSV row in {path} at line {line_idx}: {row}")

            filename = row[0].strip()
            transcription = "\t".join(row[1:]).strip()

            if not filename:
                raise ValueError(f"Empty filename in {path} at line {line_idx}")

            rows.append((filename, transcription))

    return rows


def infer_level(transcription: str) -> str:
    """
    В этом датасете фактически встречаются слова и короткие фразы.
    Простая эвристика:
      - есть пробел → phrase
      - иначе → word
    """
    if " " in transcription.strip():
        return "phrase"
    return "word"


def convert_cyrillic_handwriting(
    raw_dir: str | Path,
    out_dir: str | Path,
    link_mode: Literal["symlink", "copy", "none"] = "symlink",
) -> list[dict]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    train_tsv = raw_dir / "train.tsv"
    test_tsv = raw_dir / "test.tsv"
    train_img_dir = raw_dir / "train"
    test_img_dir = raw_dir / "test"

    required = [train_tsv, test_tsv, train_img_dir, test_img_dir]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files/dirs: {missing}")

    processed_images_dir = out_dir / "images"
    records: list[dict] = []

    split_sources = [
        ("train_source", train_tsv, train_img_dir),
        ("test_source", test_tsv, test_img_dir),
    ]

    running_id = 0

    for source_split, tsv_path, image_dir in split_sources:
        rows = _read_tsv(tsv_path)

        for filename, raw_text in rows:
            running_id += 1

            image_src = image_dir / filename
            if not image_src.exists():
                # Не падаем: validator потом отметит missing_image.
                image_path = str(image_src)
            else:
                dst_name = f"cyr_{running_id:06d}_{filename}"
                image_dst = processed_images_dir / dst_name
                image_path = _safe_symlink_or_copy(image_src, image_dst, link_mode)

            norm = normalize_text_ru(raw_text)
            level = infer_level(norm.nfc)

            sample_id = f"cyr_{level}_{running_id:06d}"

            transcription_modes = {
                "raw": norm.raw,
                "nfc": norm.nfc,
                "lower": norm.lower,
                "no_punct": norm.no_punct,
                "ctc_default": norm.ctc_default,
                "ctc_no_punct": norm.ctc_no_punct,
                "ru_yo_to_e": norm.ru_yo_to_e,
            }

            sample = SampleMetadata(
                sample_id=sample_id,
                dataset=DATASET_NAME,
                source_id=filename,
                language="ru",
                script="cyrillic",
                level=level,
                writer_id=None,
                image_path=image_path,
                raw_transcription=norm.raw,
                normalized_transcription=norm.ctc_default,
                transcription_modes=transcription_modes,
                split=None,
                metadata={
                    "source_split": source_split,
                    "scan_type": "unknown",
                    "acquisition_condition": "unknown",
                    "page_id": None,
                    "line_id": None,
                    "word_id": None,
                    "quality_flags": [],
                    "usable_for_htr": True,
                    "usable_for_graph": True,
                    "usable_for_gold_subset": True,
                    "usable_for_robustness": True,
                    "original_filename": filename,
                },
            )

            records.append(sample.to_dict())

    metadata_path = out_dir / "metadata.jsonl"
    write_jsonl(records, metadata_path)

    print(f"Converted {len(records)} samples")
    print(f"Wrote metadata: {metadata_path}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/interim/cyrillic-handwriting-dataset",
        help="Path with train.tsv, test.tsv, train/, test/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/cyrillic_handwriting",
    )
    parser.add_argument(
        "--link_mode",
        type=str,
        choices=["symlink", "copy", "none"],
        default="symlink",
    )

    args = parser.parse_args()

    convert_cyrillic_handwriting(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()