from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label
from skimage.morphology import remove_small_objects


IMAGE_KEYS = [
    "image_path",
    "path",
    "crop_path",
    "file_path",
    "img_path",
    "image",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({
        key
        for row in rows
        for key in row.keys()
    })

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def get_text(row: dict[str, Any]) -> str:
    for key in [
        "text",
        "target",
        "normalized_transcription",
        "raw_transcription",
        "transcription",
        "label",
    ]:
        value = row.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def nested_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def first_value(
    row: dict[str, Any],
    metadata: dict[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if row.get(key) not in [None, ""]:
            return row[key]

        if metadata.get(key) not in [None, ""]:
            return metadata[key]

    return None


def get_image_value(row: dict[str, Any]) -> str:
    for key in IMAGE_KEYS:
        value = row.get(key)

        if isinstance(value, str) and value:
            return value

    metadata = nested_metadata(row)

    for key in IMAGE_KEYS:
        value = metadata.get(key)

        if isinstance(value, str) and value:
            return value

    raise KeyError(
        f"No image path in row {row.get('sample_id')}"
    )


def resolve_image_path(
    value: str,
    manifest_path: Path,
) -> Path:
    path = Path(value)

    candidates = [
        path,
        Path.cwd() / path,
        manifest_path.parent / path,
        manifest_path.parent.parent / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(value)


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L"),
        dtype=np.uint8,
    )


def school_dark_auto(arr: np.ndarray) -> tuple[np.ndarray, int]:
    fg145 = remove_small_objects(
        (arr < 145).astype(bool),
        min_size=4,
    )

    if float(fg145.mean()) <= 0.35:
        return fg145.astype(bool), 145

    fg120 = remove_small_objects(
        (arr < 120).astype(bool),
        min_size=4,
    )

    return fg120.astype(bool), 120


def horizontal_line_features(
    fg: np.ndarray,
) -> dict[str, float]:
    h, w = fg.shape

    if h == 0 or w == 0 or not fg.any():
        return {
            "horizontal_line_fraction": 0.0,
            "horizontal_line_width_fraction": 0.0,
            "max_row_occupancy": 0.0,
            "horizontal_line_rows": 0,
        }

    binary = fg.astype(np.uint8) * 255

    kernel_width = max(
        9,
        min(w, int(round(w * 0.35))),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1),
    )

    detected = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    ) > 0

    fg_pixels = max(int(fg.sum()), 1)

    line_fraction = float(
        detected.sum() / fg_pixels
    )

    columns_with_line = (
        detected.any(axis=0).sum()
        if detected.any()
        else 0
    )

    row_occupancy = fg.mean(axis=1)

    return {
        "horizontal_line_fraction": line_fraction,
        "horizontal_line_width_fraction": float(
            columns_with_line / max(w, 1)
        ),
        "max_row_occupancy": float(
            row_occupancy.max(initial=0.0)
        ),
        "horizontal_line_rows": int(
            np.sum(detected.mean(axis=1) >= 0.30)
        ),
    }


def foreground_features(
    fg: np.ndarray,
) -> dict[str, float | int]:
    h, w = fg.shape
    fg_count = int(fg.sum())

    if fg_count == 0:
        return {
            "fg_fraction": 0.0,
            "component_count": 0,
            "border_fg_ratio": 0.0,
            "touches_top": 0,
            "touches_bottom": 0,
            "touches_left": 0,
            "touches_right": 0,
            "touching_sides": 0,
        }

    labels = label(fg, connectivity=2)
    component_count = int(labels.max())

    border_width = max(
        1,
        min(3, h // 4, w // 4),
    )

    border = np.zeros_like(fg, dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    touches = {
        "touches_top": int(fg[0, :].any()),
        "touches_bottom": int(fg[-1, :].any()),
        "touches_left": int(fg[:, 0].any()),
        "touches_right": int(fg[:, -1].any()),
    }

    return {
        "fg_fraction": float(fg.mean()),
        "component_count": component_count,
        "border_fg_ratio": float(
            np.logical_and(fg, border).sum()
            / max(fg_count, 1)
        ),
        **touches,
        "touching_sides": int(sum(touches.values())),
    }


def exact_image_hash(arr: np.ndarray) -> str:
    return hashlib.sha1(
        arr.tobytes()
    ).hexdigest()


def difference_hash(arr: np.ndarray) -> str:
    image = Image.fromarray(
        arr,
        mode="L",
    ).resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )

    small = np.asarray(image, dtype=np.uint8)
    bits = small[:, 1:] > small[:, :-1]

    value = 0

    for bit in bits.flatten():
        value = (value << 1) | int(bit)

    return f"{value:016x}"


def bbox_from_row(
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> list[float] | None:
    value = (
        row.get("bbox")
        or metadata.get("crop_bbox_xyxy")
        or metadata.get("bbox")
    )

    if (
        isinstance(value, list)
        and len(value) >= 4
    ):
        return [float(x) for x in value[:4]]

    return None


def quantiles(
    values: list[float],
) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }

    arr = np.asarray(values, dtype=np.float64)

    return {
        "min": float(arr.min()),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def audit_sample(
    row: dict[str, Any],
    split: str,
    manifest_path: Path,
) -> dict[str, Any]:
    metadata = nested_metadata(row)

    image_value = get_image_value(row)
    image_path = resolve_image_path(
        image_value,
        manifest_path,
    )

    arr = load_gray(image_path)
    h, w = arr.shape

    text = get_text(row)
    text_len = len(text)

    fg, threshold_used = school_dark_auto(arr)

    fg_info = foreground_features(fg)
    line_info = horizontal_line_features(fg)

    p05 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    contrast_range = p95 - p05
    gray_std = float(arr.std())

    width_per_char = (
        float(w / text_len)
        if text_len > 0
        else float(w)
    )

    pixels_per_char = (
        float(w * h / text_len)
        if text_len > 0
        else float(w * h)
    )

    page_id = first_value(
        row,
        metadata,
        "page_id",
    )
    line_id = first_value(
        row,
        metadata,
        "line_id",
        "group_id",
    )
    source_image_file = first_value(
        row,
        metadata,
        "source_image_file",
    )
    source_image_path = first_value(
        row,
        metadata,
        "source_image_path",
    )
    category = first_value(
        row,
        metadata,
        "category",
    )

    bbox = bbox_from_row(row, metadata)

    reasons: list[str] = []
    score = 0

    if not text:
        reasons.append("empty_text")
        score += 5

    if text_len <= 1:
        reasons.append("single_character_or_mark")
        score += 2
    elif text_len <= 2:
        reasons.append("very_short_text")
        score += 1

    if w <= 24:
        reasons.append("tiny_width")
        score += 2

    if h <= 12:
        reasons.append("tiny_height")
        score += 2

    if w * h <= 512:
        reasons.append("tiny_area")
        score += 2

    if width_per_char < 5.0:
        reasons.append("low_width_per_character")
        score += 2

    if float(fg_info["fg_fraction"]) < 0.005:
        reasons.append("almost_empty_foreground")
        score += 4

    if float(fg_info["fg_fraction"]) > 0.35:
        reasons.append("foreground_heavy")
        score += 3

    if contrast_range < 25:
        reasons.append("low_contrast")
        score += 2

    if gray_std < 10:
        reasons.append("low_gray_variance")
        score += 1

    if float(fg_info["border_fg_ratio"]) > 0.20:
        reasons.append("foreground_near_border")
        score += 2

    if int(fg_info["touching_sides"]) >= 3:
        reasons.append("foreground_touches_many_sides")
        score += 2

    horizontal_fraction = float(
        line_info["horizontal_line_fraction"]
    )
    horizontal_width = float(
        line_info["horizontal_line_width_fraction"]
    )
    max_row = float(
        line_info["max_row_occupancy"]
    )

    if (
        horizontal_fraction >= 0.12
        and horizontal_width >= 0.55
    ):
        reasons.append("probable_horizontal_ruling")
        score += 3
    elif max_row >= 0.65:
        reasons.append("long_horizontal_structure")
        score += 2

    component_count = int(
        fg_info["component_count"]
    )

    if component_count > max(
        12,
        5 * max(text_len, 1),
    ):
        reasons.append("excess_components")
        score += 2

    attrs = metadata.get("attributes")

    if isinstance(attrs, dict):
        if attrs.get("occluded") is True:
            reasons.append("source_marked_occluded")
            score += 2

    page_key = (
        str(source_image_file)
        if source_image_file
        else str(source_image_path)
        if source_image_path
        else str(page_id)
        if page_id is not None
        else ""
    )

    return {
        "sample_id": str(row.get("sample_id", "")),
        "split": split,
        "image_path": str(image_path),
        "text": text,
        "text_len": text_len,
        "level": str(row.get("level", "")),
        "category": str(category or ""),
        "page_id": str(page_id or ""),
        "page_key": page_key,
        "line_id": str(line_id or ""),
        "bbox_json": json.dumps(
            bbox,
            ensure_ascii=False,
        ) if bbox else "",
        "width": w,
        "height": h,
        "area": w * h,
        "aspect_ratio": float(
            w / max(h, 1)
        ),
        "width_per_char": width_per_char,
        "pixels_per_char": pixels_per_char,
        "gray_p05": p05,
        "gray_p95": p95,
        "contrast_range": contrast_range,
        "gray_std": gray_std,
        "threshold_used": threshold_used,
        **fg_info,
        **line_info,
        "exact_hash": exact_image_hash(arr),
        "dhash": difference_hash(arr),
        "suspicion_score": score,
        "suspicion_reasons": ";".join(reasons),
        "suspicious": int(score >= 2),
    }


def build_line_groups(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        if not row["page_key"] or not row["line_id"]:
            continue

        key = (
            row["split"],
            row["page_key"],
            row["line_id"],
        )
        grouped[key].append(row)

    output = []

    for (
        split,
        page_key,
        line_id,
    ), items in grouped.items():
        if len(items) < 2:
            continue

        def x0(item: dict[str, Any]) -> float:
            if not item["bbox_json"]:
                return 0.0

            bbox = json.loads(item["bbox_json"])
            return float(bbox[0])

        ordered = sorted(items, key=x0)

        texts = [
            str(item["text"])
            for item in ordered
        ]

        bboxes = [
            json.loads(item["bbox_json"])
            for item in ordered
            if item["bbox_json"]
        ]

        combined_bbox = None

        if bboxes:
            combined_bbox = [
                min(b[0] for b in bboxes),
                min(b[1] for b in bboxes),
                max(b[2] for b in bboxes),
                max(b[3] for b in bboxes),
            ]

        output.append({
            "split": split,
            "page_key": page_key,
            "line_id": line_id,
            "sample_count": len(ordered),
            "sample_ids": [
                item["sample_id"]
                for item in ordered
            ],
            "texts": texts,
            "combined_text": " ".join(texts),
            "contains_tiny_sample": any(
                (
                    int(item["width"]) <= 24
                    or int(item["height"]) <= 12
                    or int(item["text_len"]) <= 2
                )
                for item in ordered
            ),
            "max_suspicion_score": max(
                int(item["suspicion_score"])
                for item in ordered
            ),
            "combined_bbox": combined_bbox,
        })

    output.sort(
        key=lambda row: (
            not row["contains_tiny_sample"],
            -row["sample_count"],
            -row["max_suspicion_score"],
        )
    )

    return output


def build_leakage_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_hash_splits: dict[
        str,
        set[str],
    ] = defaultdict(set)

    page_splits: dict[
        str,
        set[str],
    ] = defaultdict(set)

    dhash_splits: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in rows:
        exact_hash_splits[
            row["exact_hash"]
        ].add(row["split"])

        dhash_splits[
            row["dhash"]
        ].add(row["split"])

        if row["page_key"]:
            page_splits[
                row["page_key"]
            ].add(row["split"])

    cross_split_exact = {
        key: sorted(value)
        for key, value in exact_hash_splits.items()
        if len(value) > 1
    }

    cross_split_dhash = {
        key: sorted(value)
        for key, value in dhash_splits.items()
        if len(value) > 1
    }

    cross_split_pages = {
        key: sorted(value)
        for key, value in page_splits.items()
        if len(value) > 1
    }

    return {
        "cross_split_exact_hash_count": len(
            cross_split_exact
        ),
        "cross_split_dhash_count": len(
            cross_split_dhash
        ),
        "cross_split_page_count": len(
            cross_split_pages
        ),
        "cross_split_exact_hashes": (
            cross_split_exact
        ),
        "cross_split_dhashes": (
            cross_split_dhash
        ),
        "cross_split_pages": (
            cross_split_pages
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest_root",
        default=(
            "data/experiments/"
            "htr_baseline_v1_ctc_ready/"
            "school_notebooks_clean"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=(
            "outputs/iter2_data_audit/"
            "school_notebooks_v1"
        ),
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    failures = []

    for split in ["train", "val", "test"]:
        manifest_path = (
            manifest_root / f"{split}.jsonl"
        )

        rows = read_jsonl(manifest_path)

        print(
            f"{split}: {len(rows)} samples"
        )

        for index, row in enumerate(
            rows,
            start=1,
        ):
            try:
                audited = audit_sample(
                    row,
                    split,
                    manifest_path,
                )
                all_rows.append(audited)
            except Exception as exc:
                failures.append({
                    "split": split,
                    "sample_id": row.get(
                        "sample_id",
                    ),
                    "error": repr(exc),
                })

            if index % 2000 == 0:
                print(
                    f"  {index}/{len(rows)}"
                )

    ranked = sorted(
        all_rows,
        key=lambda row: (
            -int(row["suspicion_score"]),
            -float(
                row["horizontal_line_fraction"]
            ),
            int(row["area"]),
        ),
    )

    suspicious = [
        row for row in ranked
        if int(row["suspicious"]) == 1
    ]

    line_groups = build_line_groups(
        all_rows
    )
    leakage = build_leakage_summary(
        all_rows
    )

    reason_counts = Counter()

    for row in all_rows:
        for reason in str(
            row["suspicion_reasons"]
        ).split(";"):
            if reason:
                reason_counts[reason] += 1

    summary = {
        "sample_count": len(all_rows),
        "failure_count": len(failures),
        "split_counts": dict(Counter(
            row["split"]
            for row in all_rows
        )),
        "suspicious_count": len(suspicious),
        "suspicious_rate": (
            len(suspicious)
            / max(len(all_rows), 1)
        ),
        "reason_counts": dict(
            reason_counts.most_common()
        ),
        "line_group_count": len(
            line_groups
        ),
        "line_groups_with_tiny_count": sum(
            bool(row["contains_tiny_sample"])
            for row in line_groups
        ),
        "threshold_120_rate": float(
            np.mean([
                row["threshold_used"] == 120
                for row in all_rows
            ])
        ) if all_rows else 0.0,
        "distributions": {
            key: quantiles([
                float(row[key])
                for row in all_rows
            ])
            for key in [
                "width",
                "height",
                "area",
                "text_len",
                "width_per_char",
                "fg_fraction",
                "component_count",
                "border_fg_ratio",
                "horizontal_line_fraction",
                "horizontal_line_width_fraction",
                "max_row_occupancy",
                "contrast_range",
                "gray_std",
                "suspicion_score",
            ]
        },
        "leakage": {
            key: value
            for key, value in leakage.items()
            if not isinstance(value, dict)
        },
    }

    write_csv(
        all_rows,
        out_dir / "sample_metrics.csv",
    )

    write_csv(
        suspicious[: args.top_n],
        out_dir / "suspicious_top.csv",
    )

    write_jsonl(
        line_groups,
        out_dir / "natural_line_groups.jsonl",
    )

    write_jsonl(
        failures,
        out_dir / "failures.jsonl",
    )

    (
        out_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        out_dir / "leakage.json"
    ).write_text(
        json.dumps(
            leakage,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()