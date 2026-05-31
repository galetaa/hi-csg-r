from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Literal

from src.datasets.metadata import SampleMetadata, write_jsonl
from src.datasets.text_normalization import normalize_text_ru


DATASET_NAME = "hkr_words"


def _safe_symlink_or_copy(src: Path, dst: Path, mode: Literal["symlink", "copy", "none"]) -> str:
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


def load_json(path: Path) -> dict[str, Any]:
    for enc in ["utf-8", "utf-8-sig", "cp1251"]:
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"Cannot read JSON: {path}")


def parse_name_parts(stem: str) -> dict[str, str | None]:
    """
    HKR Words filenames look like:
      0_0_0
      0_10_611
      11_19_50_

    We do not interpret these as reliable writer_id.
    We store them as source structural ids only.
    """
    clean = stem.rstrip("_")
    parts = clean.split("_")

    return {
        "template_id": parts[0] if len(parts) > 0 else None,
        "class_or_line_id": parts[1] if len(parts) > 1 else None,
        "instance_id": parts[2] if len(parts) > 2 else None,
        "had_trailing_underscore": str(stem.endswith("_")),
    }


def infer_level(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "word"

    # HKR Words archive contains both single words and short phrase crops.
    if any(ch.isspace() for ch in stripped):
        return "phrase"

    return "word"


def convert_hkr_words(
    raw_dir: str | Path,
    out_dir: str | Path,
    link_mode: Literal["symlink", "copy", "none"] = "symlink",
) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    ann_dir = raw_dir / "ann"
    img_dir = raw_dir / "img"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Missing ann dir: {ann_dir}")

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing img dir: {img_dir}")

    ann_paths = sorted(ann_dir.glob("*.json"))
    processed_images_dir = out_dir / "images"

    records: list[dict[str, Any]] = []

    missing_images = 0
    empty_text = 0
    name_mismatch = 0

    for idx, ann_path in enumerate(ann_paths, start=1):
        stem = ann_path.stem
        img_path = img_dir / f"{stem}.jpg"

        quality_flags: list[str] = []

        if not img_path.exists():
            missing_images += 1
            quality_flags.append("missing_image")
            image_path = str(img_path)
        else:
            dst = processed_images_dir / f"{stem}.jpg"
            image_path = _safe_symlink_or_copy(img_path, dst, link_mode)

        data = load_json(ann_path)

        raw_text = str(data.get("description", "") or "").strip()
        ann_name = str(data.get("name", "") or "").strip()

        if not raw_text:
            empty_text += 1
            quality_flags.append("empty_raw_transcription")

        if ann_name and ann_name != stem:
            name_mismatch += 1
            quality_flags.append("annotation_name_mismatch")

        norm = normalize_text_ru(raw_text)
        level = infer_level(raw_text)
        name_parts = parse_name_parts(stem)

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
            sample_id=f"hkr_words_{idx:06d}",
            dataset=DATASET_NAME,
            source_id=stem,
            language="ru_kk",
            script="cyrillic",
            level=level,
            writer_id=None,
            image_path=image_path,
            raw_transcription=norm.raw,
            normalized_transcription=norm.ctc_default,
            transcription_modes=transcription_modes,
            split=None,
            metadata={
                "source_split": None,
                "scan_type": "scan",
                "acquisition_condition": "scan",
                "page_id": name_parts["template_id"],
                "line_id": name_parts["class_or_line_id"],
                "word_id": name_parts["instance_id"],
                "quality_flags": quality_flags,
                "usable_for_htr": bool(raw_text and img_path.exists()),
                "usable_for_graph": bool(img_path.exists()),
                "usable_for_gold_subset": bool(raw_text and img_path.exists()),
                "usable_for_robustness": False,
                "template_id": name_parts["template_id"],
                "class_or_line_id": name_parts["class_or_line_id"],
                "instance_id": name_parts["instance_id"],
                "had_trailing_underscore": name_parts["had_trailing_underscore"] == "True",
                "annotation_path": str(ann_path),
                "annotation_name": ann_name,
                "moderation": data.get("moderation"),
                "declared_size": data.get("size"),
                "transcription_scope": "segmented_word_or_short_phrase",
                "original_filename": img_path.name,
            },
        )

        records.append(sample.to_dict())

    metadata_path = out_dir / "metadata.jsonl"
    write_jsonl(records, metadata_path)

    print(f"Converted HKR Words samples: {len(records)}")
    print(f"Missing images: {missing_images}")
    print(f"Empty text: {empty_text}")
    print(f"Annotation name mismatches: {name_mismatch}")
    print(f"Wrote metadata: {metadata_path}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/interim/hkr/words")
    parser.add_argument("--out_dir", default="data/processed/hkr_words")
    parser.add_argument(
        "--link_mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
    )
    args = parser.parse_args()

    convert_hkr_words(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()