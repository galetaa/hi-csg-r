from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.datasets.metadata import read_jsonl

METADATA = Path("data/processed/iam/metadata.preprocessed.jsonl")
OUT = Path("data/reports/iam/summary_report.md")


def stats(xs):
    xs = list(xs)
    return {
        "count": len(xs),
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
        "mean": sum(xs) / len(xs) if xs else None,
    }


def main() -> None:
    records = read_jsonl(METADATA)

    split_counts = Counter(r["split"] for r in records)
    level_counts = Counter(r["level"] for r in records)
    flags = Counter(f for r in records for f in r["metadata"].get("quality_flags", []))

    writers_total = len({r.get("writer_id") for r in records if r.get("writer_id")})
    writers_by_split = {
        split: len(
            {r.get("writer_id") for r in records if r["split"] == split and r.get("writer_id")}
        )
        for split in ["train", "val", "test"]
    }

    text_lengths = [len(r["normalized_transcription"]) for r in records]
    image_widths = [
        r["metadata"].get("image_info", {}).get("width")
        for r in records
        if r["metadata"].get("image_info", {}).get("width") is not None
    ]
    image_heights = [
        r["metadata"].get("image_info", {}).get("height")
        for r in records
        if r["metadata"].get("image_info", {}).get("height") is not None
    ]

    seg_status = Counter(r["metadata"].get("segmentation_status") for r in records)

    report = f"""# IAM Dataset — audit summary

Status

Dataset is converted, validated, writer-independent split, and preprocessed.

Final metadata:

{METADATA}

Counts
{json.dumps(split_counts, indent=2)}

Writers
Total unique writers: {writers_total}

Unique writers by split
{json.dumps(writers_by_split, indent=2)}

Quality flags
{json.dumps(flags, indent=2)}

Segmentation status
{json.dumps(seg_status, indent=2)}

Text length (characters)
{json.dumps(stats(text_lengths), indent=2)}

Image widths (pixels)
{json.dumps(stats(image_widths), indent=2)}

Image heights (pixels)
{json.dumps(stats(image_heights), indent=2)}
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()