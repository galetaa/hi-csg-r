from __future__ import annotations

import argparse
import json
import random
import shutil
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    rows = read_jsonl(dataset_dir / "train.jsonl")

    # Prefer short/medium words for first overfit.
    candidates = [
        r for r in rows
        if 3 <= len(str(r.get("text", ""))) <= 12
    ]

    if len(candidates) < args.n:
        candidates = rows

    selected = rng.sample(candidates, min(args.n, len(candidates)))

    write_jsonl(selected, out_dir / "train.jsonl")
    write_jsonl(selected, out_dir / "val.jsonl")
    write_jsonl(selected, out_dir / "all.jsonl")
    shutil.copy2(dataset_dir / "vocab.json", out_dir / "vocab.json")

    summary = {
        "source": str(dataset_dir),
        "out_dir": str(out_dir),
        "n": len(selected),
        "text_len_min": min(len(r["text"]) for r in selected),
        "text_len_max": max(len(r["text"]) for r in selected),
        "examples": [
            {"sample_id": r["sample_id"], "text": r["text"], "image_path": r["image_path"]}
            for r in selected[:20]
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()