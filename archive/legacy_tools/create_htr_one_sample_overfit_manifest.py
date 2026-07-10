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
    parser.add_argument("--n_repeats", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_len", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=8)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    rows = read_jsonl(dataset_dir / "train.jsonl")

    candidates = [
        r for r in rows
        if args.min_len <= len(str(r.get("text", ""))) <= args.max_len
        and Path(r["image_path"]).exists()
    ]

    if not candidates:
        candidates = [r for r in rows if Path(r["image_path"]).exists()]

    base = rng.choice(candidates)

    repeated = []
    for i in range(args.n_repeats):
        r = dict(base)
        r["sample_id"] = f"{base['sample_id']}_repeat_{i:04d}"
        repeated.append(r)

    write_jsonl(repeated, out_dir / "train.jsonl")
    write_jsonl(repeated, out_dir / "val.jsonl")
    write_jsonl(repeated, out_dir / "all.jsonl")
    shutil.copy2(dataset_dir / "vocab.json", out_dir / "vocab.json")

    summary = {
        "source": str(dataset_dir),
        "out_dir": str(out_dir),
        "n_repeats": len(repeated),
        "base_sample": {
            "sample_id": base["sample_id"],
            "text": base["text"],
            "text_len": len(base["text"]),
            "image_path": base["image_path"],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()