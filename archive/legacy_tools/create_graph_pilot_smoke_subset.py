from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.datasets.metadata import read_jsonl, write_jsonl


PILOT = Path("data/pilot/graph_pilot_v1.jsonl")
OUT = Path("data/pilot/graph_pilot_smoke_40.jsonl")


def main() -> None:
    records = read_jsonl(PILOT)

    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)

    selected = []
    for dataset in ["iam", "cyrillic_handwriting", "hwr200", "hkr_forms"]:
        selected.extend(by_dataset[dataset][:10])

    write_jsonl(selected, OUT)

    print(f"wrote: {OUT}")
    print("records:", len(selected))
    for dataset in ["iam", "cyrillic_handwriting", "hwr200", "hkr_forms"]:
        print(dataset, sum(1 for r in selected if r["dataset"] == dataset))


if __name__ == "__main__":
    main()