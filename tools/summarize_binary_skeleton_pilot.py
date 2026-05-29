from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/graph_pilot_smoke_40/binary_skeleton_pilot_summary.json",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    print("records:", len(data))
    print("by dataset:", Counter(r["dataset"] for r in data))

    grouped = defaultdict(list)
    warnings = Counter()

    for r in data:
        dataset = r["dataset"]

        for method, d in r["methods"].items():
            if "foreground_ratio" in d:
                grouped[(dataset, method)].append(d)

            for w in d.get("warnings", []):
                warnings[(dataset, method, w)] += 1

    print("\nStats by dataset + method:")
    for (dataset, method), rows in sorted(grouped.items()):
        fg = [x["foreground_ratio"] for x in rows]
        sk = [x["skeleton_pixels"] for x in rows]
        fp = [x["foreground_pixels"] for x in rows]

        grid_removed = [
            x.get("grid_removed_ratio")
            for x in rows
            if x.get("grid_removed_ratio") is not None
        ]

        print(f"\n{dataset} / {method}")
        print("  n:", len(rows))
        print("  foreground_ratio mean:", mean(fg))
        print("  foreground_pixels mean:", mean(fp))
        print("  skeleton_pixels mean:", mean(sk))
        if grid_removed:
            print("  grid_removed_ratio mean:", mean(grid_removed))

    print("\nWarnings:")
    for (dataset, method, warning), count in warnings.most_common(100):
        print(f"  {dataset:22s} {method:20s} {warning:30s} {count}")


if __name__ == "__main__":
    main()