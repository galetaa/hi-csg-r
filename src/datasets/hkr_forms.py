from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from src.datasets.metadata import SampleMetadata, write_jsonl
from src.datasets.text_normalization import normalize_text_ru


DATASET_NAME = "hkr_forms"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


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


def read_text_any_encoding(path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1251", "latin1"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def parse_class_indexes(path: Path) -> list[dict[str, Any]]:
    """
    Format examples:
      0\tШёл человек.
      26\tШёл степью,

    Empty lines are ignored.
    """
    text = read_text_any_encoding(path)
    items: list[dict[str, Any]] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip("\ufeff").rstrip()

        if not line.strip():
            continue

        parts = line.split(maxsplit=1)

        if len(parts) < 2:
            continue

        idx_raw, value = parts[0].strip(), parts[1].strip()

        if not idx_raw.isdigit():
            continue

        items.append(
            {
                "class_index": int(idx_raw),
                "text": value,
                "line_no": line_no,
            }
        )

    items.sort(key=lambda x: x["class_index"])
    return items


def load_all_class_indexes(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(raw_dir.glob("class_indexes_*.txt")):
        m = re.search(r"class_indexes_(\d+)\.txt$", path.name)
        if not m:
            continue

        form_id = m.group(1)
        mapping[form_id] = parse_class_indexes(path)

    return mapping


def is_suspicious_filename(path: Path) -> bool:
    # В inspection были файлы вида "... - x.jpg".
    return bool(re.search(r" - x\.(jpg|jpeg|png)$", path.name, flags=re.IGNORECASE))


def convert_hkr_forms(
    raw_dir: str | Path,
    out_dir: str | Path,
    link_mode: Literal["symlink", "copy", "none"] = "symlink",
) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    class_indexes = load_all_class_indexes(raw_dir)
    print(f"Loaded class index files: {len(class_indexes)}")

    processed_images_dir = out_dir / "images"
    records: list[dict[str, Any]] = []

    image_paths = [
        p for p in sorted(raw_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and ".ipynb_checkpoints" not in p.parts
    ]

    print(f"Found images: {len(image_paths)}")

    missing_class_index = 0
    suspicious_filenames = 0

    for idx, src_img in enumerate(image_paths, start=1):
        form_id = src_img.parent.name

        items = class_indexes.get(form_id)
        quality_flags: list[str] = []

        if not items:
            missing_class_index += 1
            quality_flags.append("missing_class_index")
            raw_text = ""
        else:
            # Полный template text формы.
            raw_text = " ".join(item["text"] for item in items).strip()

        if is_suspicious_filename(src_img):
            suspicious_filenames += 1
            quality_flags.append("suspicious_filename_x")

        norm = normalize_text_ru(raw_text)

        sample_id = f"hkr_forms_page_{idx:06d}"
        dst = processed_images_dir / form_id / f"{sample_id}{src_img.suffix.lower()}"
        image_path = _safe_symlink_or_copy(src_img, dst, link_mode)

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
            source_id=str(src_img.relative_to(raw_dir)),
            language="ru",
            script="cyrillic",
            level="form_page",
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
                "page_id": src_img.stem,
                "line_id": None,
                "word_id": None,
                "quality_flags": quality_flags,
                # Не использовать как обычный HTR baseline.
                "usable_for_htr": False,
                "usable_for_graph": True,
                "usable_for_gold_subset": True,
                "usable_for_robustness": False,
                "form_id": form_id,
                "class_index_path": str(raw_dir / f"class_indexes_{form_id}.txt"),
                "num_class_items": len(items or []),
                "class_items_preview": (items or [])[:10],
                "transcription_scope": "full_form_template_text_not_line_level",
                "original_path": str(src_img),
                "original_filename": src_img.name,
            },
        )

        records.append(sample.to_dict())

    metadata_path = out_dir / "metadata.jsonl"
    write_jsonl(records, metadata_path)

    print(f"Converted HKR Forms samples: {len(records)}")
    print(f"Missing class index samples: {missing_class_index}")
    print(f"Suspicious filename samples: {suspicious_filenames}")
    print(f"Wrote metadata: {metadata_path}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/interim/hkr/forms")
    parser.add_argument("--out_dir", default="data/processed/hkr_forms")
    parser.add_argument(
        "--link_mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
    )
    args = parser.parse_args()

    convert_hkr_forms(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()