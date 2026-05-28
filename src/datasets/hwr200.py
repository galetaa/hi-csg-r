from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Literal

from src.datasets.metadata import SampleMetadata, write_jsonl
from src.datasets.text_normalization import normalize_text_ru


DATASET_NAME = "hwr200"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

CONDITION_MAP = {
    "Сканы": "scan",
    "ФотоСветлое": "photo_light",
    "ФотоТемное": "photo_dark",
}


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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def document_group_from_folder(document_folder: str) -> str:
    lower = document_folder.lower()

    if lower.startswith("fpr"):
        return "FPR"

    if lower.startswith("original"):
        return "Originals"

    if lower.startswith("reuse"):
        return "Reuse"

    return "unknown"


def numeric_suffix(text: str) -> str | None:
    m = re.search(r"(\d+)$", text)
    if not m:
        return None
    return m.group(1)


def annotation_path_for(
    annotations_root: Path,
    writer_id: str,
    document_folder: str,
) -> Path | None:
    group = document_group_from_folder(document_folder)
    n = numeric_suffix(document_folder)

    if n is None:
        return None

    if group == "FPR":
        return annotations_root / "FPR" / n / f"fpr{n}.json"

    if group == "Originals":
        return annotations_root / "Originals" / n / f"original{n}.json"

    if group == "Reuse":
        return annotations_root / "Reuse" / writer_id / f"reuse{n}.json"

    return None


def find_image_groups(raw_dir: Path) -> dict[tuple[str, str, str], list[Path]]:
    """
    Возвращает группы:
      (writer_id, document_folder, condition_raw) -> [page images]

    Ожидаемый путь:
      .../hwr200_0_19/hw_dataset/0/fpr0/Сканы/1.JPG
    """
    image_groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)

    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        if ".git" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue

        parts = path.parts

        try:
            hw_idx = parts.index("hw_dataset")
        except ValueError:
            continue

        # Need: hw_dataset / writer_id / document_folder / condition / file
        if len(parts) <= hw_idx + 4:
            continue

        writer_id = parts[hw_idx + 1]
        document_folder = parts[hw_idx + 2]
        condition_raw = parts[hw_idx + 3]

        if condition_raw not in CONDITION_MAP:
            continue

        image_groups[(writer_id, document_folder, condition_raw)].append(path)

    # sort pages naturally by numeric stem if possible
    for key, paths in image_groups.items():
        image_groups[key] = sorted(paths, key=page_sort_key)

    return image_groups


def page_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (10**9, path.name)


def extract_text_from_annotation(data: dict[str, Any]) -> tuple[str, int | None, list[dict[str, Any]]]:
    full_text = str(data.get("full_text", "") or "").strip()
    words_count = data.get("words_count")

    sentences_raw = data.get("sentences", [])
    sentences: list[dict[str, Any]] = []

    if isinstance(sentences_raw, list):
        for s in sentences_raw:
            if isinstance(s, dict):
                sentences.append(s)

    return full_text, words_count, sentences


def convert_hwr200(
    raw_dir: str | Path,
    out_dir: str | Path,
    link_mode: Literal["symlink", "copy", "none"] = "symlink",
) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    annotations_root = raw_dir / "annotations"
    if not annotations_root.exists():
        raise FileNotFoundError(f"Missing annotations root: {annotations_root}")

    image_groups = find_image_groups(raw_dir)
    print(f"Found image groups: {len(image_groups)}")

    processed_images_dir = out_dir / "images"
    records: list[dict[str, Any]] = []

    missing_annotations = 0
    empty_full_text = 0
    linked_pages_total = 0

    for idx, ((writer_id, document_folder, condition_raw), page_paths) in enumerate(
        sorted(image_groups.items()),
        start=1,
    ):
        condition = CONDITION_MAP[condition_raw]
        source_group = document_group_from_folder(document_folder)

        ann_path = annotation_path_for(
            annotations_root=annotations_root,
            writer_id=writer_id,
            document_folder=document_folder,
        )

        annotation_exists = ann_path is not None and ann_path.exists()

        if annotation_exists:
            ann_data = load_json(ann_path)
            full_text, words_count, sentences = extract_text_from_annotation(ann_data)
        else:
            missing_annotations += 1
            ann_data = {}
            full_text = ""
            words_count = None
            sentences = []

        if not full_text:
            empty_full_text += 1

        norm = normalize_text_ru(full_text)

        sample_id = f"hwr200_document_condition_{idx:06d}"

        # Link/copy all page images into processed/images/{sample_id}/
        page_image_paths: list[str] = []
        sample_img_dir = processed_images_dir / sample_id

        for page_idx, src_page in enumerate(page_paths, start=1):
            dst = sample_img_dir / f"page_{page_idx:03d}{src_page.suffix.lower()}"
            page_image_paths.append(_safe_symlink_or_copy(src_page, dst, link_mode))

        linked_pages_total += len(page_image_paths)

        first_image = page_image_paths[0] if page_image_paths else ""

        transcription_modes = {
            "raw": norm.raw,
            "nfc": norm.nfc,
            "lower": norm.lower,
            "no_punct": norm.no_punct,
            "ctc_default": norm.ctc_default,
            "ctc_no_punct": norm.ctc_no_punct,
            "ru_yo_to_e": norm.ru_yo_to_e,
        }

        quality_flags: list[str] = []

        if not annotation_exists:
            quality_flags.append("missing_annotation")

        if not full_text:
            quality_flags.append("empty_full_text")

        if not page_image_paths:
            quality_flags.append("missing_pages")

        sample = SampleMetadata(
            sample_id=sample_id,
            dataset=DATASET_NAME,
            source_id=f"{writer_id}/{document_folder}/{condition}",
            language="ru",
            script="cyrillic",
            level="document_condition",
            writer_id=writer_id,
            image_path=first_image,
            raw_transcription=norm.raw,
            normalized_transcription=norm.ctc_default,
            transcription_modes=transcription_modes,
            split=None,
            metadata={
                "source_split": None,
                "scan_type": condition,
                "acquisition_condition": condition,
                "page_id": document_folder,
                "line_id": None,
                "word_id": None,
                "quality_flags": quality_flags,
                # Важно: не использовать как обычный single-image HTR ground truth.
                "usable_for_htr": False,
                "usable_for_graph": True,
                "usable_for_gold_subset": True,
                "usable_for_robustness": True,
                "writer_id_raw": writer_id,
                "document_id": document_folder,
                "source_group": source_group,
                "condition_raw": condition_raw,
                "annotation_path": str(ann_path) if ann_path is not None else None,
                "annotation_exists": annotation_exists,
                "page_image_paths": page_image_paths,
                "num_pages": len(page_image_paths),
                "words_count": words_count,
                "sentences_count": len(sentences),
                "sentences_preview": sentences[:5],
                "transcription_scope": "document_full_text_not_single_page",
            },
        )

        records.append(sample.to_dict())

    metadata_path = out_dir / "metadata.jsonl"
    write_jsonl(records, metadata_path)

    print(f"Converted HWR200 document-condition samples: {len(records)}")
    print(f"Linked/copied pages total: {linked_pages_total}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Empty full_text: {empty_full_text}")
    print(f"Wrote metadata: {metadata_path}")

    print("Counts by condition:", Counter(r["metadata"]["acquisition_condition"] for r in records))
    print("Counts by source_group:", Counter(r["metadata"]["source_group"] for r in records))

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/interim/hwr200")
    parser.add_argument("--out_dir", default="data/processed/hwr200")
    parser.add_argument(
        "--link_mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
    )
    args = parser.parse_args()

    convert_hwr200(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()