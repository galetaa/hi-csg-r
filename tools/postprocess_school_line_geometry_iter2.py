from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


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


def compute_x_gaps(
    bboxes: list[list[float]],
) -> list[float]:
    if len(bboxes) < 2:
        return []

    bboxes = sorted(
        bboxes,
        key=lambda box: (
            float(box[0]),
            float(box[1]),
        ),
    )

    gaps = []

    for left, right in zip(bboxes[:-1], bboxes[1:]):
        gap = float(right[0]) - float(left[2])
        gaps.append(max(0.0, gap))

    return gaps


def median_word_width(bboxes: list[list[float]]) -> float:
    widths = [
        max(1.0, float(box[2]) - float(box[0]))
        for box in bboxes
        if len(box) >= 4
    ]

    if not widths:
        return 1.0

    return max(1.0, float(np.median(widths)))


def enrich(row: dict[str, Any], *, large_gap_threshold: float) -> dict[str, Any]:
    row = dict(row)

    bboxes = row.get("word_bboxes_xyxy") or []
    line_bbox = row.get("bbox_xyxy") or [0, 0, 1, 1]

    x0, y0, x1, y1 = [
        float(value)
        for value in line_bbox[:4]
    ]

    line_height = max(1.0, y1 - y0)
    med_word_width = median_word_width(bboxes)

    gaps = compute_x_gaps(bboxes)

    max_gap = max(gaps) if gaps else 0.0
    median_gap = float(np.median(gaps)) if gaps else 0.0

    max_gap_norm_h = max_gap / line_height
    median_gap_norm_h = median_gap / line_height
    max_gap_norm_word_width = max_gap / med_word_width

    flags = list(row.get("flags") or [])

    is_large_gap = (
        int(row.get("n_words", 0)) >= 2
        and max_gap_norm_h > large_gap_threshold
    )

    if is_large_gap and "large_gap_outlier" not in flags:
        flags.append("large_gap_outlier")

    has_invalid = "has_invalid_or_review" in flags
    has_hard = "has_hard_real" in flags
    n_words = int(row.get("n_words", 0))

    if is_large_gap:
        bucket = "large_gap_outlier"
    elif n_words == 2 or has_hard or n_words >= 8:
        bucket = "hard_line"
    else:
        bucket = "clean_line"

    if has_invalid:
        bucket = "invalid_or_review_line"

    row["x_gaps"] = gaps
    row["x_gap_max"] = max_gap
    row["x_gap_median"] = median_gap
    row["x_gap_max_norm_by_height"] = max_gap_norm_h
    row["x_gap_median_norm_by_height"] = median_gap_norm_h
    row["x_gap_max_norm_by_word_height"] = max_gap_norm_h
    row["x_gap_max_norm_by_median_word_width"] = max_gap_norm_word_width
    row["median_word_width"] = med_word_width
    row["is_large_gap_outlier"] = is_large_gap
    row["line_corpus_bucket"] = bucket
    row["flags"] = flags

    return row


def quantile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(
        row["line_corpus_bucket"]
        for row in rows
    )

    n = len(rows)

    return {
        "n": n,
        "bucket_counts": dict(bucket_counts),
        "bucket_rates": {
            key: value / max(n, 1)
            for key, value in bucket_counts.items()
        },
        "n_words": quantile_summary([
            float(row["n_words"])
            for row in rows
        ]),
        "x_gap_max_norm_by_height": quantile_summary([
            float(row["x_gap_max_norm_by_height"])
            for row in rows
        ]),
        "x_gap_max_norm_by_median_word_width": quantile_summary([
            float(row["x_gap_max_norm_by_median_word_width"])
            for row in rows
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--large_gap_threshold", type=float, default=3.5)
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    summary: dict[str, Any] = {
        "in_root": str(in_root),
        "out_root": str(out_root),
        "large_gap_threshold": args.large_gap_threshold,
        "splits": {},
    }

    all_rows = []

    for split in ["train", "val", "test"]:
        rows = read_jsonl(
            in_root / f"{split}.full_line_candidates.jsonl"
        )

        enriched = [
            enrich(row, large_gap_threshold=args.large_gap_threshold)
            for row in rows
        ]

        write_jsonl(
            enriched,
            out_root / f"{split}.line_candidates.geometry.jsonl",
        )

        summary["splits"][split] = summarize(enriched)

        all_rows.extend(enriched)

    write_jsonl(
        all_rows,
        out_root / "all.line_candidates.geometry.jsonl",
    )

    summary["total"] = summarize(all_rows)

    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
