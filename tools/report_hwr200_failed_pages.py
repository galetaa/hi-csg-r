from __future__ import annotations

import csv
from pathlib import Path

from src.datasets.metadata import read_jsonl


METADATA = Path("data/processed/hwr200/metadata.preprocessed.jsonl")
OUT = Path("data/reports/hwr200/hwr200_failed_pages_report.csv")


def main() -> None:
    records = read_jsonl(METADATA)

    rows = []

    for r in records:
        errors = r["metadata"].get("page_preprocess_errors", [])
        if not errors:
            continue

        for e in errors:
            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "split": r["split"],
                    "writer_id": r.get("writer_id"),
                    "document_id": r["metadata"].get("document_id"),
                    "source_group": r["metadata"].get("source_group"),
                    "condition": r["metadata"].get("acquisition_condition"),
                    "page_idx": e.get("page_idx"),
                    "page_path": e.get("page_path"),
                    "error": e.get("error"),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "split",
            "writer_id",
            "document_id",
            "source_group",
            "condition",
            "page_idx",
            "page_path",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"failed page rows: {len(rows)}")
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()