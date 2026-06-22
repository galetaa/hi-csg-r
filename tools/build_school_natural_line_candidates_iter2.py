from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.preprocessing.school_rectangular_v2 import (
    SchoolCocoSource,
    bbox_from_points,
    coordinates_to_points,
    flatten_segmentation,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in ["metadata", "source_metadata"]:
        value = row.get(key)

        if isinstance(value, dict):
            result.update(value)

    return result


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "normalized_transcription", "raw_transcription"]:
        value = row.get(key)

        if value is not None:
            return str(value)

    return ""


def get_quality_by_sample(
    quality_root: Path,
    split: str,
) -> dict[str, str]:
    out: dict[str, str] = {}

    for bucket in ["clean_core", "hard_real", "invalid_or_review"]:
        path = quality_root / f"{split}.{bucket}.jsonl"

        if not path.exists():
            continue

        for row in read_jsonl(path):
            out[str(row["sample_id"])] = bucket

    return out


def bbox_from_polygon_or_row(row: dict[str, Any]) -> list[int] | None:
    value = row.get("bbox")

    if isinstance(value, list) and len(value) >= 4:
        return [
            int(round(float(x)))
            for x in value[:4]
        ]

    poly = row.get("polygon")

    if isinstance(poly, list) and len(poly) >= 3:
        points = [
            point
            for point in poly
            if isinstance(point, list) and len(point) >= 2
        ]

        if len(points) >= 3:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]

            return [
                int(math.floor(min(xs))),
                int(math.floor(min(ys))),
                int(math.ceil(max(xs))),
                int(math.ceil(max(ys))),
            ]

    return None


def bbox_from_coco(
    row: dict[str, Any],
    coco: SchoolCocoSource,
) -> list[int]:
    annotation, image_info = coco.resolve_annotation(row)
    coordinates = flatten_segmentation(annotation.get("segmentation"))
    points = coordinates_to_points(coordinates)

    return list(
        bbox_from_points(
            points,
            image_width=int(image_info["width"]),
            image_height=int(image_info["height"]),
            padding=0,
        )
    )


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    meta = metadata(row)

    page_id = meta.get("page_id")
    line_id = meta.get("line_id")
    source_image_file = meta.get("source_image_file")

    split = row.get("split")

    if not split or page_id in [None, ""] or line_id in [None, ""]:
        return None

    if source_image_file in [None, ""]:
        source_image_file = str(page_id)

    return (
        str(split),
        str(source_image_file),
        str(page_id),
        str(line_id),
    )


def line_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    bbox = row["_bbox_xyxy"]

    return (
        float(bbox[0]),
        float(bbox[1]),
        str(row["sample_id"]),
    )


def natural_sort_line_id(value: Any) -> tuple[int, str]:
    text = str(value)

    if text.isdigit():
        return int(text), text

    return 10**9, text


def build_candidates_for_split(
    *,
    graph_root: Path,
    quality_root: Path,
    split: str,
    min_words: int,
    coco: SchoolCocoSource,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = [
        row
        for row in read_jsonl(graph_root / f"{split}.jsonl")
        if row.get("dataset") == "school_notebooks_clean"
    ]

    quality_by_id = get_quality_by_sample(
        quality_root,
        split,
    )

    groups: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    skipped = Counter()

    for row in rows:
        key = group_key(row)

        if key is None:
            skipped["missing_group_key"] += 1
            continue

        bbox = bbox_from_polygon_or_row(row)

        if bbox is None:
            try:
                bbox = bbox_from_coco(row, coco)
            except Exception:
                skipped["missing_bbox"] += 1
                continue

        row = dict(row)
        row["_bbox_xyxy"] = bbox
        groups[key].append(row)

    candidates: list[dict[str, Any]] = []

    for (
        split_value,
        source_image_file,
        page_id,
        line_id,
    ), items in groups.items():
        if len(items) < min_words:
            skipped["below_min_words"] += len(items)
            continue

        items = sorted(
            items,
            key=line_sort_key,
        )

        sample_ids = [
            str(row["sample_id"])
            for row in items
        ]

        texts = [
            get_text(row)
            for row in items
        ]

        quality_buckets = [
            quality_by_id.get(
                str(row["sample_id"]),
                "unknown",
            )
            for row in items
        ]

        bboxes = [
            row["_bbox_xyxy"]
            for row in items
        ]

        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)

        bucket_counts = Counter(quality_buckets)

        flags = []

        if bucket_counts.get("hard_real", 0):
            flags.append("has_hard_real")

        if bucket_counts.get("invalid_or_review", 0):
            flags.append("has_invalid_or_review")

        if bucket_counts.get("clean_core", 0) == len(items):
            flags.append("all_clean_core")

        if len(items) <= 2:
            flags.append("short_group")

        if len(items) >= 8:
            flags.append("long_group")

        if any(len(text.strip()) <= 1 for text in texts):
            flags.append("contains_single_char_or_mark")

        line_group_id = (
            f"school_line_{split_value}_"
            f"{page_id}_{line_id}"
        )

        candidates.append({
            "line_group_id": line_group_id,
            "split": split_value,
            "source_image_file": source_image_file,
            "page_id": page_id,
            "line_id": line_id,
            "sample_ids": sample_ids,
            "texts": texts,
            "joined_text": " ".join(
                text.strip()
                for text in texts
                if text.strip()
            ),
            "quality_buckets": quality_buckets,
            "quality_bucket_counts": dict(bucket_counts),
            "n_words": len(items),
            "bbox_xyxy": [x0, y0, x1, y1],
            "word_bboxes_xyxy": bboxes,
            "flags": flags,
        })

    candidates = sorted(
        candidates,
        key=lambda row: (
            row["split"],
            row["source_image_file"],
            natural_sort_line_id(row["line_id"]),
            row["bbox_xyxy"][1],
            row["bbox_xyxy"][0],
        ),
    )

    return candidates, dict(skipped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_root", required=True)
    parser.add_argument("--quality_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--min_words", type=int, default=2)
    parser.add_argument(
        "--school_raw_dir",
        default="data/interim/school_notebooks",
    )
    args = parser.parse_args()

    graph_root = Path(args.graph_root)
    quality_root = Path(args.quality_root)
    out_root = Path(args.out_root)
    coco = SchoolCocoSource(args.school_raw_dir)

    summary: dict[str, Any] = {
        "graph_root": str(graph_root),
        "quality_root": str(quality_root),
        "school_raw_dir": str(Path(args.school_raw_dir)),
        "min_words": args.min_words,
        "splits": {},
    }

    all_candidates = []

    for split in ["train", "val", "test"]:
        candidates, skipped = build_candidates_for_split(
            graph_root=graph_root,
            quality_root=quality_root,
            split=split,
            min_words=args.min_words,
            coco=coco,
        )

        write_jsonl(
            candidates,
            out_root / f"{split}.line_candidates.jsonl",
        )

        all_candidates.extend(candidates)

        summary["splits"][split] = {
            "n_line_groups": len(candidates),
            "n_word_instances": sum(row["n_words"] for row in candidates),
            "n_all_clean_core": sum("all_clean_core" in row["flags"] for row in candidates),
            "n_has_hard_real": sum("has_hard_real" in row["flags"] for row in candidates),
            "n_has_invalid": sum("has_invalid_or_review" in row["flags"] for row in candidates),
            "n_short_groups": sum("short_group" in row["flags"] for row in candidates),
            "n_long_groups": sum("long_group" in row["flags"] for row in candidates),
            "skipped": skipped,
        }

    write_jsonl(
        all_candidates,
        out_root / "all.line_candidates.jsonl",
    )

    summary["total"] = {
        "n_line_groups": len(all_candidates),
        "n_word_instances": sum(row["n_words"] for row in all_candidates),
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
