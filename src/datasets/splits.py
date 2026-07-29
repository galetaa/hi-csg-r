from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.datasets.metadata import read_jsonl, write_jsonl


def create_random_split(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    sample_ids = [r["sample_id"] for r in records]
    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    n = len(sample_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": sample_ids[:n_train],
        "val": sample_ids[n_train:n_train + n_val],
        "test": sample_ids[n_train + n_val:],
    }


def create_source_split_based_split(records: list[dict[str, Any]], seed: int = 42) -> dict[str, list[str]]:
    """
    Для Cyrillic Handwriting Dataset уже есть source train/test.
    Делаем:
      - original train_source → train/val
      - original test_source → test
    """
    train_source = []
    test_source = []

    for r in records:
        source_split = r.get("metadata", {}).get("source_split")
        if source_split == "test_source":
            test_source.append(r["sample_id"])
        else:
            train_source.append(r["sample_id"])

    rng = random.Random(seed)
    rng.shuffle(train_source)

    n_val = max(1, int(len(train_source) * 0.1))

    return {
        "train": train_source[n_val:],
        "val": train_source[:n_val],
        "test": test_source,
    }


def create_writer_independent_split(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    writer_to_samples: dict[str, list[str]] = defaultdict(list)
    missing_writer = False

    for r in records:
        writer_id = r.get("writer_id")
        if not writer_id:
            missing_writer = True
            break
        writer_to_samples[writer_id].append(r["sample_id"])

    if missing_writer:
        raise ValueError("writer_id is missing for at least one sample; cannot create writer-independent split.")

    writers = list(writer_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(writers)

    n = len(writers)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_writers = writers[:n_train]
    val_writers = writers[n_train:n_train + n_val]
    test_writers = writers[n_train + n_val:]

    return {
        "train": [s for w in train_writers for s in writer_to_samples[w]],
        "val": [s for w in val_writers for s in writer_to_samples[w]],
        "test": [s for w in test_writers for s in writer_to_samples[w]],
    }


def assign_split_to_metadata(
    metadata_path: str | Path,
    splits_path: str | Path,
    out_path: str | Path,
) -> None:
    records = read_jsonl(metadata_path)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))

    sample_to_split = {}
    for split_name in ["train", "val", "test"]:
        for sample_id in splits[split_name]:
            sample_to_split[sample_id] = split_name

    for r in records:
        r["split"] = sample_to_split.get(r["sample_id"])

    write_jsonl(records, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--split_type",
        choices=["random", "source_train_test", "writer_independent"],
        default="random",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--assign_metadata_out", type=str, default=None)

    args = parser.parse_args()

    records = read_jsonl(args.metadata)

    if args.split_type == "random":
        split = create_random_split(records, args.train_ratio, args.val_ratio, args.seed)
    elif args.split_type == "source_train_test":
        split = create_source_split_based_split(records, args.seed)
    elif args.split_type == "writer_independent":
        split = create_writer_independent_split(records, args.train_ratio, args.val_ratio, args.seed)
    else:
        raise ValueError(args.split_type)

    output = {
        "split_version": "v1",
        "split_type": args.split_type,
        "seed": args.seed,
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        "num_train": len(split["train"]),
        "num_val": len(split["val"]),
        "num_test": len(split["test"]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote splits: {out_path}")
    print(f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    if args.assign_metadata_out:
        assign_split_to_metadata(args.metadata, out_path, args.assign_metadata_out)
        print(f"Wrote metadata with split: {args.assign_metadata_out}")


if __name__ == "__main__":
    main()