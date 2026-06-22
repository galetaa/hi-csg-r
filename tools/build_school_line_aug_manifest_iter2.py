from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_BUCKETS = {
    "2_words": 1000,
    "3_words": 1000,
    "4_7_words": 2250,
    "8plus_words": 750,
}


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


def length_bucket(row: dict[str, Any]) -> str:
    n = int(row.get("n_words", 0))

    if n == 2:
        return "2_words"

    if n == 3:
        return "3_words"

    if 4 <= n <= 7:
        return "4_7_words"

    return "8plus_words"


def valid_for_line_aug(row: dict[str, Any]) -> bool:
    bucket = row.get("line_corpus_bucket")

    if bucket not in {"clean_line", "hard_line"}:
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


def stable_jitter(value: str, seed: int) -> float:
    digest = hashlib.sha256(
        f"{seed}:{value}".encode("utf-8")
    ).hexdigest()
    integer = int(digest[:16], 16)

    return integer / float(16**16)


def sample_bucket(
    rows: list[dict[str, Any]],
    target_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("x_gap_max_norm_by_height", 0.0)),
            stable_jitter(str(row.get("line_group_id", "")), seed),
        ),
    )

    preferred_cut = int(len(rows) * 0.80)
    preferred = rows[:preferred_cut]
    fallback = rows[preferred_cut:]

    if len(preferred) >= target_n:
        return rng.sample(preferred, target_n)

    selected = list(preferred)
    need = target_n - len(selected)

    if need > 0:
        selected.extend(
            rng.sample(
                fallback,
                min(need, len(fallback)),
            )
        )

    return selected


def scaled_targets(target_total: int) -> dict[str, int]:
    if target_total == 5000:
        return dict(TARGET_BUCKETS)

    scale = target_total / 5000.0
    targets = {
        key: int(round(value * scale))
        for key, value in TARGET_BUCKETS.items()
    }

    delta = target_total - sum(targets.values())
    targets["4_7_words"] += delta

    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--target_total", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    geometry_root = Path(args.geometry_root)
    out_root = Path(args.out_root)

    path = geometry_root / f"{args.split}.line_candidates.geometry.jsonl"
    rows = [
        row
        for row in read_jsonl(path)
        if valid_for_line_aug(row)
    ]

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_bucket[length_bucket(row)].append(row)

    target_buckets = scaled_targets(args.target_total)
    selected = []

    for bucket, target_n in target_buckets.items():
        candidates = by_bucket[bucket]

        chosen = sample_bucket(
            candidates,
            target_n,
            seed=args.seed + sum(ord(ch) for ch in bucket),
        )

        selected.extend(chosen)

    selected_records = [
        to_line_record(row, args.split)
        for row in selected
    ]

    selected_records.sort(
        key=lambda row: row["sample_id"],
    )

    write_jsonl(
        selected_records,
        out_root / f"{args.split}.jsonl",
    )

    summary = {
        "geometry_root": str(geometry_root),
        "out_root": str(out_root),
        "split": args.split,
        "target_total": args.target_total,
        "selected_total": len(selected_records),
        "available_total": len(rows),
        "seed": args.seed,
        "target_buckets": target_buckets,
        "available_length_buckets": {
            key: len(value)
            for key, value in sorted(by_bucket.items())
        },
        "selected_length_buckets": dict(Counter(
            length_bucket(row)
            for row in selected
        )),
        "selected_line_corpus_buckets": dict(Counter(
            row.get("line_corpus_bucket")
            for row in selected
        )),
        "selected_n_words": {
            "min": min(row["n_words"] for row in selected_records),
            "max": max(row["n_words"] for row in selected_records),
            "mean": sum(row["n_words"] for row in selected_records) / max(len(selected_records), 1),
        },
        "selected_x_gap_max_norm_by_height": {
            "max": max(
                float(row.get("x_gap_max_norm_by_height") or 0.0)
                for row in selected_records
            ),
            "mean": sum(
                float(row.get("x_gap_max_norm_by_height") or 0.0)
                for row in selected_records
            ) / max(len(selected_records), 1),
        },
    }

    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
