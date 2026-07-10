from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


DATASETS = [
    "iam",
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks_clean",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_rows(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/experiments/htr_baseline_v1_ctc_ready")
    parser.add_argument("--train_n", type=int, default=1000)
    parser.add_argument("--val_n", type=int, default=500)
    parser.add_argument("--test_n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)

    for dataset in DATASETS:
        ds_dir = root / dataset
        smoke_dir = ds_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)

        selected_all = []

        for split, n in [("train", args.train_n), ("val", args.val_n), ("test", args.test_n)]:
            rows = read_jsonl(ds_dir / f"{split}.jsonl")
            selected = sample_rows(rows, n, rng)
            write_jsonl(selected, smoke_dir / f"{split}.jsonl")
            selected_all.extend(selected)

        write_jsonl(selected_all, smoke_dir / "all.jsonl")
        shutil.copy2(ds_dir / "vocab.json", smoke_dir / "vocab.json")

        summary = {
            "dataset": dataset,
            "source_dir": str(ds_dir),
            "smoke_dir": str(smoke_dir),
            "counts": {
                "train": sum(1 for _ in (smoke_dir / "train.jsonl").open("r", encoding="utf-8")),
                "val": sum(1 for _ in (smoke_dir / "val.jsonl").open("r", encoding="utf-8")),
                "test": sum(1 for _ in (smoke_dir / "test.jsonl").open("r", encoding="utf-8")),
            },
        }

        (smoke_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()