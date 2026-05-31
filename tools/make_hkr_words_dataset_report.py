from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.datasets.metadata import read_jsonl

METADATA = Path("data/processed/hkr_words/metadata.preprocessed.jsonl")
OUT = Path("data/reports/hkr_words/summary_report.md")


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
    text_counts = Counter(
        r["transcription_modes"]["ctc_default"] for r in records if r.get("transcription_modes")
    )

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

    text_lengths = [len(r.get("normalized_transcription", "")) for r in records]

    report = f"""# HKR Words — audit summary

## Status

HKR Words is converted, validated, text-group split, and preprocessed.

Final metadata:

{METADATA}

Counts
total samples: {len(records)}
unique normalized texts: {len(text_counts)}

Split counts
{json.dumps(dict(split_counts), ensure_ascii=False, indent=2)}

Level counts
{json.dumps(dict(level_counts), ensure_ascii=False, indent=2)}

Quality flags
{json.dumps(dict(flags), ensure_ascii=False, indent=2)}

Image width stats
{json.dumps(stats(image_widths), ensure_ascii=False, indent=2)}

Image height stats
{json.dumps(stats(image_heights), ensure_ascii=False, indent=2)}

Text length stats
{json.dumps(stats(text_lengths), ensure_ascii=False, indent=2)}

Top repeated normalized texts
{json.dumps(text_counts.most_common(30), ensure_ascii=False, indent=2)}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
