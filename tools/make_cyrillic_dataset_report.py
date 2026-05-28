from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.datasets.metadata import read_jsonl

import json

METADATA = Path("data/processed/cyrillic_handwriting/metadata.preprocessed.jsonl")
OUT = Path("data/reports/cyrillic_handwriting/summary_report.md")


def main() -> None:
    records = read_jsonl(METADATA)

    split_counts = Counter(r["split"] for r in records)
    level_counts = Counter(r["level"] for r in records)
    flags = Counter(f for r in records for f in r["metadata"].get("quality_flags", []))

    text_lengths = [len(r["normalized_transcription"]) for r in records if r["split"] != "excluded"]
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

    def stats(xs):
        xs = list(xs)
        return {
            "count": len(xs),
            "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
            "mean": sum(xs) / len(xs) if xs else None,
        }

    report = f"""# Cyrillic Handwriting Dataset — audit summary

Status

Dataset is converted, validated, deduplicated by split, and preprocessed.

Final metadata:
{METADATA}

Counts
total: {len(records)}

Split counts
{json.dumps(dict(split_counts), ensure_ascii=False, indent=2)}

Level counts
{json.dumps(dict(level_counts), ensure_ascii=False, indent=2)}

Quality flags
{json.dumps(dict(flags), ensure_ascii=False, indent=2)}

Text length stats (for non-excluded records)
{json.dumps(stats(text_lengths), ensure_ascii=False, indent=2)}

Image width stats
{json.dumps(stats(image_widths), ensure_ascii=False, indent=2)}

Image height stats
{json.dumps(stats(image_heights), ensure_ascii=False, indent=2)}
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
