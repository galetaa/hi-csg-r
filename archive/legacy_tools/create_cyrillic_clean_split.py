from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

from src.datasets.metadata import read_jsonl, write_jsonl


METADATA_PATH = Path("data/processed/cyrillic_handwriting/metadata.validated.jsonl")
SPLIT_OUT = Path("data/splits/cyrillic_handwriting_splits.clean.json")
METADATA_OUT = Path("data/processed/cyrillic_handwriting/metadata.clean_splits.jsonl")
DUP_REPORT_OUT = Path("data/reports/cyrillic_handwriting/duplicate_groups_report.csv")

SEED = 42
VAL_RATIO_FROM_TRAIN_SOURCE_GROUPS = 0.10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_excluded_from_htr(record: dict[str, Any]) -> bool:
    flags = set(record.get("metadata", {}).get("quality_flags", []))

    if not record.get("raw_transcription"):
        return True

    if not record.get("normalized_transcription"):
        return True

    hard_bad_flags = {
        "missing_image",
        "broken_image",
        "empty_raw_transcription",
        "empty_normalized_transcription",
    }

    return bool(flags & hard_bad_flags)


def main() -> None:
    records = read_jsonl(METADATA_PATH)

    excluded_ids: set[str] = set()
    usable_records: list[dict[str, Any]] = []

    for r in records:
        if is_excluded_from_htr(r):
            excluded_ids.add(r["sample_id"])
        else:
            usable_records.append(r)

    print("total records:", len(records))
    print("usable records:", len(usable_records))
    print("excluded records:", len(excluded_ids))

    hash_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in usable_records:
        p = Path(r["image_path"])
        if not p.exists():
            excluded_ids.add(r["sample_id"])
            continue

        h = sha256_file(p)
        hash_to_records[h].append(r)

    groups = list(hash_to_records.values())
    print("hash groups:", len(groups))

    # Report duplicate groups and annotation conflicts.
    DUP_REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    with DUP_REPORT_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group_id",
            "group_size",
            "splits_source",
            "num_unique_transcriptions",
            "transcriptions",
            "sample_ids",
            "source_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        duplicate_group_count = 0
        conflict_group_count = 0

        for group_id, group in enumerate(groups):
            if len(group) <= 1:
                continue

            duplicate_group_count += 1
            transcriptions = sorted({r.get("normalized_transcription", "") for r in group})
            source_splits = sorted({r.get("metadata", {}).get("source_split", "unknown") for r in group})

            if len(transcriptions) > 1:
                conflict_group_count += 1

            writer.writerow(
                {
                    "group_id": group_id,
                    "group_size": len(group),
                    "splits_source": "|".join(source_splits),
                    "num_unique_transcriptions": len(transcriptions),
                    "transcriptions": " || ".join(transcriptions),
                    "sample_ids": " | ".join(r["sample_id"] for r in group),
                    "source_ids": " | ".join(str(r.get("source_id")) for r in group),
                }
            )

    print("duplicate groups:", duplicate_group_count)
    print("duplicate groups with annotation conflicts:", conflict_group_count)
    print(f"wrote duplicate report: {DUP_REPORT_OUT}")

    # Split strategy:
    # 1. If a hash group contains any original test_source sample, whole group goes to test.
    # 2. Otherwise the group is from train_source and goes group-wise to train/val.
    test_groups = []
    train_source_groups = []

    for group in groups:
        source_splits = {r.get("metadata", {}).get("source_split") for r in group}

        if "test_source" in source_splits:
            test_groups.append(group)
        else:
            train_source_groups.append(group)

    rng = random.Random(SEED)
    rng.shuffle(train_source_groups)

    n_val_groups = max(1, int(len(train_source_groups) * VAL_RATIO_FROM_TRAIN_SOURCE_GROUPS))
    val_groups = train_source_groups[:n_val_groups]
    train_groups = train_source_groups[n_val_groups:]

    def flatten(groups_: list[list[dict[str, Any]]]) -> list[str]:
        return [r["sample_id"] for g in groups_ for r in g]

    train_ids = flatten(train_groups)
    val_ids = flatten(val_groups)
    test_ids = flatten(test_groups)

    # Safety checks.
    sample_to_split = {}
    for split_name, ids in {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }.items():
        for sid in ids:
            if sid in sample_to_split:
                raise RuntimeError(f"Sample assigned twice: {sid}")
            sample_to_split[sid] = split_name

    # Check hash leakage.
    hash_to_splits = defaultdict(set)
    for h, group in hash_to_records.items():
        for r in group:
            split = sample_to_split.get(r["sample_id"])
            if split:
                hash_to_splits[h].add(split)

    leakage = {h: splits for h, splits in hash_to_splits.items() if len(splits) > 1}
    if leakage:
        raise RuntimeError(f"Hash leakage still exists: {len(leakage)} groups")

    output = {
        "dataset": "cyrillic_handwriting",
        "split_version": "clean_v1",
        "split_type": "source_test_preserved_hash_group_aware",
        "seed": SEED,
        "policy": {
            "exclude_unusable_htr": True,
            "duplicate_grouping": "sha256(image)",
            "test_priority": "if any duplicate group member is source test, whole group goes to test",
            "val_policy": "10% of train-source hash groups",
        },
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "excluded": sorted(excluded_ids),
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
        "num_excluded": len(excluded_ids),
    }

    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # Assign split back to metadata.
    for r in records:
        sid = r["sample_id"]
        if sid in excluded_ids:
            r["split"] = "excluded"
        else:
            r["split"] = sample_to_split.get(sid, "excluded")

    write_jsonl(records, METADATA_OUT)

    print(f"wrote split: {SPLIT_OUT}")
    print(f"wrote metadata: {METADATA_OUT}")
    print("split counts:", Counter(r["split"] for r in records))


if __name__ == "__main__":
    main()