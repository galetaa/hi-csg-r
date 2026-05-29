from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.datasets.metadata import read_jsonl

METADATA = Path("data/processed/hkr_forms/metadata.preprocessed.jsonl")
OUT = Path("data/reports/hkr_forms/summary_report.md")


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
    form_counts = Counter(r["metadata"].get("form_id") for r in records)
    flags = Counter(f for r in records for f in r["metadata"].get("quality_flags", []))

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

    report = f"""# HKR Forms — audit summary

## Status

HKR Forms is converted, validated, split, and preprocessed as a form/page-level dataset.

Final metadata:

{METADATA}

Counts
total samples: {len(records)}

Split counts
{json.dumps(dict(split_counts), ensure_ascii=False, indent=2)}

Level counts
{json.dumps(dict(level_counts), ensure_ascii=False, indent=2)}

Form counts
{json.dumps(dict(form_counts), ensure_ascii=False, indent=2)}

Quality flags
{json.dumps(dict(flags), ensure_ascii=False, indent=2)}

Image width stats
{json.dumps(stats(image_widths), ensure_ascii=False, indent=2)}

Image height stats
{json.dumps(stats(image_heights), ensure_ascii=False, indent=2)}

Text length stats
{json.dumps(stats(text_lengths), ensure_ascii=False, indent=2)}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"wrote: {OUT}")

if __name__ == "__main__":
    main()