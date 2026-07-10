from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.datasets.metadata import read_jsonl, write_jsonl


def split_grouped_by_text(
    records: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    eligible = [
        r for r in records
        if r.get("metadata", {}).get("usable_for_htr") is True
        and r.get("normalized_transcription")
    ]

    excluded = [
        r for r in records
        if r not in eligible
    ]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in eligible:
        key = r.get("transcription_modes", {}).get("ctc_default") or r["normalized_transcription"]
        groups[key].append(r)

    group_items = list(groups.items())

    rng = random.Random(seed)
    rng.shuffle(group_items)

    # Greedy balancing by sample count.
    total = len(eligible)
    target_train = int(total * train_ratio)
    target_val = int(total * val_ratio)
    target_test = total - target_train - target_val

    split_to_ids = {"train": [], "val": [], "test": [], "excluded": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    targets = {
        "train": target_train,
        "val": target_val,
        "test": target_test,
    }

    # Put large groups first after shuffle to reduce imbalance.
    group_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    for text_key, group_records in group_items:
        # choose split with minimal relative fill
        best_split = min(
            ["train", "val", "test"],
            key=lambda s: split_counts[s] / max(targets[s], 1),
        )

        for r in group_records:
            split_to_ids[best_split].append(r["sample_id"])

        split_counts[best_split] += len(group_records)

    for r in excluded:
        split_to_ids["excluded"].append(r["sample_id"])

    sample_to_split = {}
    for split_name, ids in split_to_ids.items():
        for sid in ids:
            sample_to_split[sid] = split_name

    return split_to_ids, sample_to_split


def verify_no_text_leakage(records: list[dict[str, Any]], sample_to_split: dict[str, str]) -> dict[str, Any]:
    text_to_splits = defaultdict(set)
    text_to_count = Counter()

    for r in records:
        split = sample_to_split.get(r["sample_id"])
        if split in {None, "excluded"}:
            continue

        text = r.get("transcription_modes", {}).get("ctc_default") or r.get("normalized_transcription")
        if not text:
            continue

        text_to_splits[text].add(split)
        text_to_count[text] += 1

    leakage = {
        text: sorted(splits)
        for text, splits in text_to_splits.items()
        if len(splits) > 1
    }

    top_repeated = text_to_count.most_common(30)

    return {
        "num_unique_texts": len(text_to_splits),
        "num_text_leakage_groups": len(leakage),
        "leakage_preview": dict(list(leakage.items())[:20]),
        "top_repeated_texts": top_repeated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/processed/hkr_words/metadata.validated.jsonl")
    parser.add_argument("--out", default="data/splits/hkr_words_splits.text_grouped.json")
    parser.add_argument("--assign_metadata_out", default="data/processed/hkr_words/metadata.text_splits.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    records = read_jsonl(metadata_path)

    split_to_ids, sample_to_split = split_grouped_by_text(
        records=records,
        seed=args.seed,
    )

    leakage_report = verify_no_text_leakage(records, sample_to_split)

    if leakage_report["num_text_leakage_groups"] != 0:
        raise RuntimeError(f"Text leakage detected: {leakage_report['num_text_leakage_groups']}")

    for r in records:
        r["split"] = sample_to_split.get(r["sample_id"], "excluded")

    split_info = {
        "dataset": "hkr_words",
        "split_version": "text_grouped_v1",
        "split_type": "normalized_transcription_grouped",
        "seed": args.seed,
        "policy": {
            "reason": "HKR Words contains many repeated target strings; identical normalized transcriptions must not cross splits.",
            "group_key": "transcription_modes.ctc_default",
            "writer_independent": False,
            "writer_id_available": False,
        },
        "num_train": len(split_to_ids["train"]),
        "num_val": len(split_to_ids["val"]),
        "num_test": len(split_to_ids["test"]),
        "num_excluded": len(split_to_ids["excluded"]),
        "train": split_to_ids["train"],
        "val": split_to_ids["val"],
        "test": split_to_ids["test"],
        "excluded": split_to_ids["excluded"],
        "leakage_report": leakage_report,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")

    write_jsonl(records, args.assign_metadata_out)

    print(f"Wrote split: {out}")
    print(f"Wrote metadata: {args.assign_metadata_out}")
    print("split counts:", Counter(r["split"] for r in records))
    print("unique texts:", leakage_report["num_unique_texts"])
    print("text leakage groups:", leakage_report["num_text_leakage_groups"])
    print("top repeated texts:")
    for text, count in leakage_report["top_repeated_texts"][:20]:
        print(repr(text), count)


if __name__ == "__main__":
    main()