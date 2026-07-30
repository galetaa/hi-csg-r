from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.htr.dataset_adapter_v2 import core_domain
from src.htr.xaligned_hi_csg_r import read_jsonl


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=128)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    rng = random.Random(args.seed)
    candidates = [
        row
        for row in rows
        if 4 <= len(str(row["text"])) <= 8
    ]
    if not candidates:
        raise ValueError("No one-sample smoke candidate")
    one = [rng.choice(candidates)]
    by_domain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[core_domain(str(row["dataset"]))].append(row)
    counts = {
        "cyrillic": args.subset_size // 3,
        "hkr": args.subset_size // 3,
        "school": args.subset_size - 2 * (args.subset_size // 3),
    }
    subset: list[dict[str, Any]] = []
    for domain, count in counts.items():
        subset.extend(rng.sample(by_domain[domain], count))
    rng.shuffle(subset)
    output = Path(args.out_dir)
    write_jsonl(output / "one_sample.jsonl", one)
    write_jsonl(output / "subset_128.jsonl", subset)
    summary = {
        "source_manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "one_sample_id": one[0]["sample_id"],
        "one_sample_text": one[0]["text"],
        "subset_size": len(subset),
        "domain_counts": counts,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

