from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


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
    if n <= 0 or len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_lens = [len(r["text"]) for r in rows]
    return {
        "count": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "levels": dict(Counter(r.get("level") for r in rows)),
        "categories": dict(Counter(r.get("category") for r in rows)),
        "text_len": {
            "min": min(text_lens) if text_lens else None,
            "max": max(text_lens) if text_lens else None,
            "mean": sum(text_lens) / len(text_lens) if text_lens else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--subset_name", required=True)
    parser.add_argument("--train_n", type=int, default=10000)
    parser.add_argument("--val_n", type=int, default=2000)
    parser.add_argument("--test_n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = dataset_dir / "subsets" / args.subset_name
    rng = random.Random(args.seed)

    selected_all = []

    for split, n in [
        ("train", args.train_n),
        ("val", args.val_n),
        ("test", args.test_n),
    ]:
        rows = read_jsonl(dataset_dir / f"{split}.jsonl")
        selected = sample_rows(rows, n, rng)
        write_jsonl(selected, out_dir / f"{split}.jsonl")
        selected_all.extend(selected)

    write_jsonl(selected_all, out_dir / "all.jsonl")
    shutil.copy2(dataset_dir / "vocab.json", out_dir / "vocab.json")

    summary = {
        "source": str(dataset_dir),
        "subset_name": args.subset_name,
        "out_dir": str(out_dir),
        "seed": args.seed,
        "requested": {
            "train_n": args.train_n,
            "val_n": args.val_n,
            "test_n": args.test_n,
        },
        "summary": summarize(selected_all),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()