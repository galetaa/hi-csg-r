from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DATASETS = {
    "cyrillic_handwriting": "cyrillic_handwriting",
    "hkr_words": "hkr_words",
    "school_notebooks_clean": "school_notebooks_clean",
}


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


def sample(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/experiments/htr_baseline_v1_ctc_ready")
    parser.add_argument("--out_dir", default="data/experiments/htr_graph_v1/subsets/tri10k")
    parser.add_argument("--train_n", type=int, default=10000)
    parser.add_argument("--val_n", type=int, default=2000)
    parser.add_argument("--test_n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    summary: dict[str, Any] = {
        "name": out_dir.name,
        "seed": args.seed,
        "datasets": {},
    }

    for ds_key, rel in DATASETS.items():
        ds_dir = root / rel
        ds_out = out_dir / ds_key

        for split, n in [
            ("train", args.train_n),
            ("val", args.val_n),
            ("test", args.test_n),
        ]:
            rows = read_jsonl(ds_dir / f"{split}.jsonl")
            rows = [dict(r, dataset=ds_key, source_dataset=ds_key) for r in rows]
            chosen = sample(rows, n, rng)
            write_jsonl(chosen, ds_out / f"{split}.jsonl")

            summary["datasets"].setdefault(ds_key, {})[split] = {
                "source_n": len(rows),
                "selected_n": len(chosen),
                "path": str(ds_out / f"{split}.jsonl"),
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()