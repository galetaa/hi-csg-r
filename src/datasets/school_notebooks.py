from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.datasets.metadata import SampleMetadata, write_jsonl
from src.datasets.text_normalization import normalize_text_ru

DATASET_NAME = "school_notebooks"

DEFAULT_TEXT_CATEGORIES = {
    "pupil_text",
    "pupil_comment",
    "teacher_comment",
}


SPLIT_FILES = {
    "train": "annotations_train.json",
    "val": "annotations_val.json",
    "test": "annotations_test.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_segmentation(segmentation: Any) -> list[float]:
    """
    COCO polygon segmentation can be:
      [[x1, y1, x2, y2, ...]]
      [x1, y1, x2, y2, ...]

    We use all polygon coordinates for bbox and mask.
    """
    if not segmentation:
        return []

    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        coords: list[float] = []
        for poly in segmentation:
            coords.extend(float(x) for x in poly)
        return coords

    if isinstance(segmentation, list):
        return [float(x) for x in segmentation]

    return []


def coords_to_points(coords: list[float]) -> list[tuple[float, float]]:
    if len(coords) < 4:
        return []

    points = []
    for i in range(0, len(coords) - 1, 2):
        points.append((coords[i], coords[i + 1]))

    return points


def bbox_from_points(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x0 = max(0, int(math.floor(min(xs))) - padding)
    y0 = max(0, int(math.floor(min(ys))) - padding)
    x1 = min(image_width, int(math.ceil(max(xs))) + padding)
    y1 = min(image_height, int(math.ceil(max(ys))) + padding)

    return x0, y0, x1, y1


def polygon_to_crop_points(
    points: list[tuple[float, float]],
    x0: int,
    y0: int,
) -> list[tuple[float, float]]:
    return [(x - x0, y - y0) for x, y in points]


def sanitize_text_for_level(text: str) -> str:
    return text.strip()


def infer_level(text: str) -> str:
    text = sanitize_text_for_level(text)

    if not text:
        return "word"

    if any(ch.isspace() for ch in text):
        return "phrase"

    return "word"


def safe_sample_id(split: str, idx: int) -> str:
    return f"school_notebooks_{split}_{idx:07d}"


def crop_polygon_region(
    *,
    image: Image.Image,
    points: list[tuple[float, float]],
    bbox: tuple[int, int, int, int],
    mask_outside_polygon: bool,
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop = image.crop((x0, y0, x1, y1)).convert("RGB")

    if not mask_outside_polygon:
        return crop

    crop_points = polygon_to_crop_points(points, x0, y0)

    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(crop_points, fill=255)

    white = Image.new("RGB", crop.size, (255, 255, 255))
    white.paste(crop, mask=mask)

    return white


def convert_split(
    *,
    split: str,
    ann_path: Path,
    images_root: Path,
    out_dir: Path,
    include_categories: set[str],
    padding: int,
    mask_outside_polygon: bool,
    max_samples: int | None,
    start_index: int,
    image_cache_size: int,
) -> tuple[list[dict[str, Any]], int]:
    data = load_json(ann_path)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}

    records: list[dict[str, Any]] = []
    sample_idx = start_index

    image_cache: OrderedDict[str, Image.Image] = OrderedDict()

    skipped_no_translation = 0
    skipped_bad_polygon = 0
    skipped_missing_image = 0
    skipped_category = 0

    crop_dir = out_dir / "images"

    for ann_idx, ann in enumerate(data["annotations"]):
        category = categories.get(ann.get("category_id"), str(ann.get("category_id")))

        if category not in include_categories:
            skipped_category += 1
            continue

        attrs = ann.get("attributes") or {}
        raw_text = str(attrs.get("translation", "") or "").strip()

        if not raw_text:
            skipped_no_translation += 1
            continue

        image_info = images.get(ann.get("image_id"))
        if not image_info:
            skipped_missing_image += 1
            continue

        image_filename = image_info["file_name"]
        image_path = images_root / image_filename

        if not image_path.exists():
            skipped_missing_image += 1
            continue

        coords = flatten_segmentation(ann.get("segmentation"))
        points = coords_to_points(coords)

        if len(points) < 3:
            skipped_bad_polygon += 1
            continue

        width = int(image_info["width"])
        height = int(image_info["height"])

        bbox = bbox_from_points(
            points=points,
            image_width=width,
            image_height=height,
            padding=padding,
        )

        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            skipped_bad_polygon += 1
            continue

        if image_cache_size <= 0:
            with Image.open(image_path) as src:
                image = src.convert("RGB")
        else:
            if image_filename in image_cache:
                image = image_cache.pop(image_filename)
                image_cache[image_filename] = image
            else:
                image = Image.open(image_path).convert("RGB")
                image_cache[image_filename] = image

                while len(image_cache) > image_cache_size:
                    _, old_image = image_cache.popitem(last=False)
                    old_image.close()

        sample_idx += 1
        sample_id = safe_sample_id(split, sample_idx)

        crop = crop_polygon_region(
            image=image,
            points=points,
            bbox=bbox,
            mask_outside_polygon=mask_outside_polygon,
        )

        crop_path = crop_dir / split / f"{sample_id}.png"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)

        crop.close()

        if image_cache_size <= 0:
            image.close()

        norm = normalize_text_ru(raw_text)
        level = infer_level(raw_text)

        quality_flags: list[str] = []

        if attrs.get("occluded") is True:
            quality_flags.append("occluded")

        if len(raw_text) <= 1:
            quality_flags.append("single_character_or_mark")

        if mask_outside_polygon:
            quality_flags.append("polygon_masked_crop")

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
            source_id=f"{split}:{image_info['id']}:{ann_idx}",
            language="ru",
            script="cyrillic",
            level=level,
            writer_id=None,
            image_path=str(crop_path),
            raw_transcription=norm.raw,
            normalized_transcription=norm.ctc_default,
            transcription_modes=transcription_modes,
            split=split,
            bbox=[x0, y0, x1, y1],
            polygon=[[float(x), float(y)] for x, y in points],
            metadata={
                "source_split": split,
                "scan_type": "photo_or_scan",
                "acquisition_condition": "notebook_page",
                "page_id": str(image_info["id"]),
                "line_id": str(ann.get("group_id")) if ann.get("group_id") is not None else None,
                "word_id": str(ann_idx),
                "quality_flags": quality_flags,
                "usable_for_htr": True,
                "usable_for_graph": True,
                "usable_for_gold_subset": True,
                "usable_for_robustness": False,
                "usable_for_segmentation": True,
                "category": category,
                "attributes": attrs,
                "source_image_file": image_filename,
                "source_image_path": str(image_path),
                "source_image_width": width,
                "source_image_height": height,
                "crop_bbox_xyxy": [x0, y0, x1, y1],
                "crop_padding": padding,
                "mask_outside_polygon": mask_outside_polygon,
                "transcription_scope": "polygon_crop_translation",
            },
        )

        records.append(sample.to_dict())

        if max_samples is not None and len(records) >= max_samples:
            break

    print(f"[{split}] records: {len(records)}")
    print(f"[{split}] skipped category: {skipped_category}")
    print(f"[{split}] skipped no translation: {skipped_no_translation}")
    print(f"[{split}] skipped bad polygon: {skipped_bad_polygon}")
    print(f"[{split}] skipped missing image: {skipped_missing_image}")

    for cached_image in image_cache.values():
        cached_image.close()
    image_cache.clear()
    return records, sample_idx


def convert_school_notebooks(
    *,
    raw_dir: str | Path,
    out_dir: str | Path,
    include_categories: set[str],
    padding: int = 8,
    mask_outside_polygon: bool = True,
    max_per_split: int | None = None,
image_cache_size: int = 2,
) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    images_root = raw_dir / "images" / "images"

    if not images_root.exists():
        raise FileNotFoundError(f"Missing images root: {images_root}")

    records: list[dict[str, Any]] = []
    counter = 0

    for split, filename in SPLIT_FILES.items():
        ann_path = raw_dir / filename

        if not ann_path.exists():
            raise FileNotFoundError(ann_path)

        split_records, counter = convert_split(
            split=split,
            ann_path=ann_path,
            images_root=images_root,
            out_dir=out_dir,
            include_categories=include_categories,
            padding=padding,
            mask_outside_polygon=mask_outside_polygon,
            max_samples=max_per_split,
            start_index=counter,
            image_cache_size=image_cache_size,
        )

        records.extend(split_records)

    metadata_path = out_dir / "metadata.jsonl"
    write_jsonl(records, metadata_path)

    print(f"Converted School Notebooks samples: {len(records)}")
    print(f"Wrote metadata: {metadata_path}")

    return records


def parse_categories(text: str) -> set[str]:
    return {x.strip() for x in text.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/interim/school_notebooks")
    parser.add_argument("--out_dir", default="data/processed/school_notebooks")
    parser.add_argument(
        "--include_categories",
        default="pupil_text,pupil_comment,teacher_comment",
    )
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--no_mask_outside_polygon", action="store_true")
    parser.add_argument("--max_per_split", type=int, default=None)
    parser.add_argument("--image_cache_size", type=int, default=2)
    args = parser.parse_args()

    convert_school_notebooks(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        include_categories=parse_categories(args.include_categories),
        padding=args.padding,
        mask_outside_polygon=not args.no_mask_outside_polygon,
        max_per_split=args.max_per_split,
        image_cache_size=args.image_cache_size,
    )


if __name__ == "__main__":
    main()
