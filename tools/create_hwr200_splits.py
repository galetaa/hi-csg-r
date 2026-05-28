from __future__ import annotations

import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

from src.datasets.metadata import read_jsonl, write_jsonl


METADATA = Path("data/processed/hwr200/metadata.validated.jsonl")
SPLIT_OUT = Path("data/splits/hwr200_splits.condition_aware.json")
METADATA_OUT = Path("data/processed/hwr200/metadata.condition_splits.jsonl")
SEED = 42


def main() -> None:
    records = read_jsonl(METADATA)

    # Group by writer+document so same document in scan/light/dark stays in one split.
    doc_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in records:
        writer_id = r.get("writer_id")
        document_id = r.get("metadata", {}).get("document_id")
        key = f"{writer_id}::{document_id}"
        doc_to_records[key].append(r)

    groups = list(doc_to_records.values())

    rng = random.Random(SEED)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * 0.70)
    n_val = int(n * 0.10)

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]

    def flatten(gs):
        return [r["sample_id"] for g in gs for r in g]

    split = {
        "train": flatten(train_groups),
        "val": flatten(val_groups),
        "test": flatten(test_groups),
    }

    sample_to_split = {}
    for split_name, ids in split.items():
        for sid in ids:
            sample_to_split[sid] = split_name

    # Verify no document condition leakage.
    doc_to_splits = defaultdict(set)
    for r in records:
        writer_id = r.get("writer_id")
        document_id = r.get("metadata", {}).get("document_id")
        key = f"{writer_id}::{document_id}"
        doc_to_splits[key].add(sample_to_split[r["sample_id"]])

    leakage = {k: v for k, v in doc_to_splits.items() if len(v) > 1}
    if leakage:
        raise RuntimeError(f"Document leakage: {len(leakage)} groups")

    output = {
        "dataset": "hwr200",
        "split_version": "condition_aware_v1",
        "split_type": "document_condition_grouped",
        "seed": SEED,
        "policy": {
            "group_key": "writer_id::document_id",
            "reason": "same document across scan/photo_light/photo_dark must not cross splits",
            "usable_for_htr": False,
            "primary_usage": "robustness_graph_quality_and_external_evaluation",
        },
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        "num_train": len(split["train"]),
        "num_val": len(split["val"]),
        "num_test": len(split["test"]),
    }

    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in records:
        r["split"] = sample_to_split[r["sample_id"]]

    write_jsonl(records, METADATA_OUT)

    print(f"wrote split: {SPLIT_OUT}")
    print(f"wrote metadata: {METADATA_OUT}")
    print("split counts:", Counter(r["split"] for r in records))
    print("condition counts:", Counter(r["metadata"]["acquisition_condition"] for r in records))
    print("source group counts:", Counter(r["metadata"]["source_group"] for r in records))
    print("document leakage:", len(leakage))


if __name__ == "__main__":
    main()