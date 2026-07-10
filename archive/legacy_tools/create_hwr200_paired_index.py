from __future__ import annotations

import json
from collections import defaultdict, Counter
from pathlib import Path

from src.datasets.metadata import read_jsonl


METADATA = Path("data/processed/hwr200/metadata.preprocessed.jsonl")
OUT = Path("data/processed/hwr200/hwr200_paired_robustness_index.json")


def main() -> None:
    records = read_jsonl(METADATA)

    groups = defaultdict(dict)

    for r in records:
        key = f'{r["writer_id"]}::{r["metadata"]["document_id"]}'
        condition = r["metadata"]["acquisition_condition"]
        groups[key][condition] = r["sample_id"]

    complete = {}
    incomplete = {}

    for key, cond_map in groups.items():
        if {"scan", "photo_light", "photo_dark"} <= set(cond_map):
            complete[key] = cond_map
        else:
            incomplete[key] = cond_map

    report = {
        "num_document_groups": len(groups),
        "num_complete_triplets": len(complete),
        "num_incomplete_groups": len(incomplete),
        "complete_triplets": complete,
        "incomplete_groups": incomplete,
        "condition_counts": dict(Counter(r["metadata"]["acquisition_condition"] for r in records)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("document groups:", len(groups))
    print("complete triplets:", len(complete))
    print("incomplete groups:", len(incomplete))
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()