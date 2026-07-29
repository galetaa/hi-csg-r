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


def valid_for_line_corpus(row: dict[str, Any]) -> bool:
    if row.get("line_corpus_bucket") in {
        "large_gap_outlier",
        "invalid_or_review_line",
    }:
        return False

    if row.get("is_large_gap_outlier") is True:
        return False

    if int(row.get("n_words", 0)) < 2:
        return False

    text = str(row.get("joined_text_space", row.get("joined_text", ""))).strip()

    return bool(text)


def to_line_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    text = str(row.get("joined_text_space", row.get("joined_text", ""))).strip()

    return {
        "sample_id": str(row["line_group_id"]),
        "dataset": "school_notebooks_line",
        "source_dataset": "school_notebooks_clean",
        "split": split,
        "language": "ru",
        "script": "cyrillic",
        "level": "line",
        "text": text,
        "raw_transcription": text,
        "normalized_transcription": text,
        "line_group_id": row["line_group_id"],
        "source_image_file": row.get("source_image_file"),
        "page_id": row.get("page_id"),
        "line_id": row.get("line_id"),
        "sample_ids": row.get("sample_ids", []),
        "texts": row.get("texts", []),
        "bbox_xyxy": row.get("bbox_xyxy"),
        "word_bboxes_xyxy": row.get("word_bboxes_xyxy"),
        "n_words": int(row.get("n_words", 0)),
        "line_corpus_bucket": row.get("line_corpus_bucket"),
        "flags": row.get("flags", []),
        "x_gaps": row.get("x_gaps", []),
        "x_gap_max": row.get("x_gap_max"),
        "x_gap_median": row.get("x_gap_median"),
        "x_gap_max_norm_by_height": row.get("x_gap_max_norm_by_height"),
        "x_gap_median_norm_by_height": row.get("x_gap_median_norm_by_height"),
        "x_gap_max_norm_by_word_height": row.get("x_gap_max_norm_by_word_height"),
        "x_gap_max_norm_by_median_word_width": row.get("x_gap_max_norm_by_median_word_width"),
        "median_word_width": row.get("median_word_width"),
        "source_type": "natural_line_group",
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n_words = [
        float(row["n_words"])
        for row in records
    ]

    if n_words:
        arr = np.asarray(n_words, dtype=np.float64)
        n_words_summary = {
            "min": float(arr.min()),
            "mean": float(arr.mean()),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max()),
        }
    else:
        n_words_summary = {
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    return {
        "n": len(records),
        "line_corpus_buckets": dict(Counter(
            row.get("line_corpus_bucket")
            for row in records
        )),
        "n_words": n_words_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_root", required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    geometry_root = Path(args.geometry_root)
    out_root = Path(args.out_root)

    summary: dict[str, Any] = {
        "geometry_root": str(geometry_root),
        "out_root": str(out_root),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        rows = read_jsonl(
            geometry_root / f"{split}.line_candidates.geometry.jsonl"
        )

        records = [
            to_line_record(row, split)
            for row in rows
            if valid_for_line_corpus(row)
        ]

        records.sort(
            key=lambda row: row["sample_id"]
        )

        write_jsonl(
            records,
            out_root / f"{split}.jsonl",
        )

        summary["splits"][split] = {
            **summarize(records),
            "source_candidates": len(rows),
            "excluded": len(rows) - len(records),
        }

    summary["total"] = {
        "n": sum(
            split_summary["n"]
            for split_summary in summary["splits"].values()
        ),
        "excluded": sum(
            split_summary["excluded"]
            for split_summary in summary["splits"].values()
        ),
    }

    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
