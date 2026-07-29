from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.datasets.metadata import read_jsonl

METADATA = Path("data/processed/school_notebooks/metadata.preprocessed.jsonl")
LEAKAGE = Path("data/reports/school_notebooks/text_leakage_report.json")
OUT = Path("data/reports/school_notebooks/summary_report.md")


def stats(xs):
    xs = [x for x in xs if x is not None]
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
    category_counts = Counter(r["metadata"].get("category") for r in records)
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

    leakage_info = {}
    if LEAKAGE.exists():
        leakage_info = json.loads(LEAKAGE.read_text(encoding="utf-8"))

    report = f"""# School Notebooks — audit summary

## Status

School Notebooks is converted, validated, and preprocessed as a polygon crop-level handwriting dataset.

Final metadata:

{METADATA}

Counts
total samples: {len(records)}

Split counts
{json.dumps(dict(split_counts), ensure_ascii=False, indent=2)}

Level counts
{json.dumps(dict(level_counts), ensure_ascii=False, indent=2)}

Category counts
{json.dumps(dict(category_counts), ensure_ascii=False, indent=2)}

Quality flags
{json.dumps(dict(flags), ensure_ascii=False, indent=2)}

Image width stats
{json.dumps(stats(image_widths), ensure_ascii=False, indent=2)}

Image height stats
{json.dumps(stats(image_heights), ensure_ascii=False, indent=2)}

Text length stats
{json.dumps(stats(text_lengths), ensure_ascii=False, indent=2)}

Text leakage check
{
        json.dumps(
            {
                k: leakage_info.get(k)
                for k in ["num_unique_texts", "num_leakage_texts", "leakage_ratio_by_unique_text"]
            },
            ensure_ascii=False,
            indent=2,
        )
    }
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print("wrote:", OUT)

if __name__ == "__main__":
    main()
