from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.datasets.metadata import read_jsonl

METADATA = Path("data/processed/hwr200/metadata.preprocessed.jsonl")
PAIRED = Path("data/processed/hwr200/hwr200_paired_robustness_index.json")
FAILED = Path("data/reports/hwr200/hwr200_failed_pages_report.csv")
OUT = Path("data/reports/hwr200/summary_report.md")


def main() -> None:
    records = read_jsonl(METADATA)

    split_counts = Counter(r["split"] for r in records)
    condition_counts = Counter(r["metadata"]["acquisition_condition"] for r in records)
    source_counts = Counter(r["metadata"]["source_group"] for r in records)
    flags = Counter(f for r in records for f in r["metadata"].get("quality_flags", []))

    total_pages = sum(r["metadata"].get("num_pages", 0) for r in records)
    total_ocr_pages = sum(r["metadata"].get("num_page_ocr_images", 0) for r in records)
    total_feature_pages = sum(r["metadata"].get("num_page_feature_images", 0) for r in records)

    paired_info = {}
    if PAIRED.exists():
        paired_info = json.loads(PAIRED.read_text(encoding="utf-8"))

    report = f"""# HWR200 — audit summary

## Status

HWR200 is converted as a document-condition robustness dataset.

Final metadata:


{METADATA}

Counts
document-condition samples: {len(records)}
total pages: {total_pages}
ocr pages: {total_ocr_pages}
feature pages: {total_feature_pages}

Split counts
{json.dumps(dict(split_counts), ensure_ascii=False, indent=2)}

Condition counts
{json.dumps(dict(condition_counts), ensure_ascii=False, indent=2)}

Source group counts
{json.dumps(dict(source_counts), ensure_ascii=False, indent=2)}

Quality flags
{json.dumps(dict(flags), ensure_ascii=False, indent=2)}

Paired robustness index
{json.dumps({k: paired_info.get(k) for k in ["num_document_groups", "num_complete_triplets", "num_incomplete_groups"]}, ensure_ascii=False, indent=2)}

Failed pages report
{FAILED if FAILED.exists() else "No failed pages report found."}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
