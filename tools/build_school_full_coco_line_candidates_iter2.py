from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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


def int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    meta = metadata(row)

    split = row.get("split")
    page_id = meta.get("page_id")
    line_id = meta.get("line_id")
    source_image_file = meta.get("source_image_file")

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


def line_sort_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
    bbox = row["_bbox_xyxy"]
    meta = metadata(row)
    word_id = int_or_none(meta.get("word_id"))

    return (
        float(bbox[0]),
        float(bbox[1]),
        word_id if word_id is not None else 10**12,
        str(row["sample_id"]),
    )


def natural_sort_line_id(value: Any) -> tuple[int, str]:
    text = str(value)

    if text.isdigit():
        return int(text), text

    return 10**9, text


def median(values: list[float]) -> float | None:
    if not values:
        return None

    return float(statistics.median(values))


def build_candidate(
    *,
    split: str,
    source_image_file: str,
    page_id: str,
    line_id: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    items = sorted(items, key=line_sort_key)

    sample_ids = [
        str(row["sample_id"])
        for row in items
    ]
    texts = [
        get_text(row)
        for row in items
    ]
    bboxes = [
        row["_bbox_xyxy"]
        for row in items
    ]
    word_ids = [
        int_or_none(metadata(row).get("word_id"))
        for row in items
    ]

    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)

    word_id_gaps: list[int | None] = []

    for left, right in zip(word_ids, word_ids[1:]):
        if left is None or right is None:
            word_id_gaps.append(None)
        else:
            word_id_gaps.append(int(right - left))

    numeric_gaps = [
        gap
        for gap in word_id_gaps
        if gap is not None
    ]

    x_gaps = [
        int(b_right[0] - b_left[2])
        for b_left, b_right in zip(bboxes, bboxes[1:])
    ]

    return {
        "line_group_id": f"school_full_line_{split}_{page_id}_{line_id}",
        "split": split,
        "source_image_file": source_image_file,
        "page_id": page_id,
        "line_id": line_id,
        "sample_ids": sample_ids,
        "word_ids_sorted": word_ids,
        "texts": texts,
        "joined_text_space": " ".join(
            text.strip()
            for text in texts
            if text.strip()
        ),
        "joined_text_raw_concat": "".join(
            text.strip()
            for text in texts
            if text.strip()
        ),
        "n_words": len(items),
        "bbox_xyxy": [x0, y0, x1, y1],
        "word_bboxes_xyxy": bboxes,
        "word_id_gaps": word_id_gaps,
        "has_word_id_gaps": any(
            abs(gap) != 1
            for gap in numeric_gaps
        ) or len(numeric_gaps) != max(len(word_ids) - 1, 0),
        "x_gaps": x_gaps,
        "max_x_gap": max(x_gaps) if x_gaps else None,
        "median_x_gap": median([float(value) for value in x_gaps]),
    }


def build_candidates_for_split(
    *,
    manifest_root: Path,
    split: str,
    min_words: int,
    coco: SchoolCocoSource,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    rows = read_jsonl(manifest_root / f"{split}.jsonl")
    total_samples = len(rows)

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

        try:
            bbox = bbox_from_coco(row, coco)
        except Exception:
            skipped["missing_bbox"] += 1
            continue

        row = dict(row)
        row["_bbox_xyxy"] = bbox
        groups[key].append(row)

    candidates = []

    for (
        split_value,
        source_image_file,
        page_id,
        line_id,
    ), items in groups.items():
        if len(items) < min_words:
            skipped["below_min_words"] += len(items)
            continue

        candidates.append(
            build_candidate(
                split=split_value,
                source_image_file=source_image_file,
                page_id=page_id,
                line_id=line_id,
                items=items,
            )
        )

    candidates.sort(
        key=lambda row: (
            row["split"],
            row["source_image_file"],
            natural_sort_line_id(row["line_id"]),
            row["bbox_xyxy"][1],
            row["bbox_xyxy"][0],
        )
    )

    return candidates, dict(skipped), total_samples


def length_summary(
    candidates: list[dict[str, Any]],
    *,
    total_samples: int,
) -> dict[str, Any]:
    lengths = [
        int(row["n_words"])
        for row in candidates
    ]
    samples_covered = int(sum(lengths))

    if lengths:
        arr = np.asarray(lengths, dtype=np.float64)
        mean_words = float(arr.mean())
        p50 = float(np.quantile(arr, 0.50))
        p90 = float(np.quantile(arr, 0.90))
        p95 = float(np.quantile(arr, 0.95))
        max_words = int(arr.max())
    else:
        mean_words = 0.0
        p50 = 0.0
        p90 = 0.0
        p95 = 0.0
        max_words = 0

    return {
        "n_line_groups": len(candidates),
        "n_word_instances": samples_covered,
        "mean_words_per_group": mean_words,
        "p50_words_per_group": p50,
        "p90_words_per_group": p90,
        "p95_words_per_group": p95,
        "max_words_per_group": max_words,
        "groups_2_words": sum(length == 2 for length in lengths),
        "groups_3_words": sum(length == 3 for length in lengths),
        "groups_4plus_words": sum(length >= 4 for length in lengths),
        "groups_8plus_words": sum(length >= 8 for length in lengths),
        "samples_covered": samples_covered,
        "total_samples": total_samples,
        "coverage_rate": samples_covered / max(total_samples, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_root",
        default="data/experiments/htr_baseline_v1_ctc_ready/school_notebooks_clean",
    )
    parser.add_argument(
        "--school_raw_dir",
        default="data/interim/school_notebooks",
    )
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--min_words", type=int, default=2)
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    out_root = Path(args.out_root)
    coco = SchoolCocoSource(args.school_raw_dir)

    summary: dict[str, Any] = {
        "manifest_root": str(manifest_root),
        "school_raw_dir": str(Path(args.school_raw_dir)),
        "out_root": str(out_root),
        "min_words": args.min_words,
        "splits": {},
    }

    all_candidates = []
    total_samples_all = 0

    for split in ["train", "val", "test"]:
        candidates, skipped, total_samples = build_candidates_for_split(
            manifest_root=manifest_root,
            split=split,
            min_words=args.min_words,
            coco=coco,
        )

        write_jsonl(
            candidates,
            out_root / f"{split}.full_line_candidates.jsonl",
        )

        all_candidates.extend(candidates)
        total_samples_all += total_samples

        summary["splits"][split] = {
            **length_summary(
                candidates,
                total_samples=total_samples,
            ),
            "skipped": skipped,
        }

        print(
            json.dumps(
                {
                    split: summary["splits"][split],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    write_jsonl(
        all_candidates,
        out_root / "all.full_line_candidates.jsonl",
    )

    summary["total"] = length_summary(
        all_candidates,
        total_samples=total_samples_all,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
