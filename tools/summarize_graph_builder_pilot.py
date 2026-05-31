from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


REPORT = Path("outputs/graph_pilot/graph_builder_pilot_report.json")


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    runs = data["runs"]

    print("graphs built:", len(runs))
    print("by dataset:", Counter(r["dataset"] for r in runs))
    print("by method:", Counter(r["method"] for r in runs))

    grouped = defaultdict(list)
    warnings = Counter()

    for r in runs:
        grouped[(r["dataset"], r["method"])].append(r)
        for w in r.get("warnings", []):
            warnings[(r["dataset"], r["method"], w)] += 1

    print("\nStats by dataset + method:")
    for (dataset, method), rows in sorted(grouped.items()):
        print(f"\n{dataset} / {method}")
        print("  n:", len(rows))
        print("  node_count mean:", mean([r["node_count"] for r in rows]))
        print("  edge_count mean:", mean([r["edge_count"] for r in rows]))
        print("  component_count mean:", mean([r["component_count"] for r in rows]))
        print("  junction_count mean:", mean([r["junction_count"] for r in rows]))
        print("  endpoint_count mean:", mean([r["endpoint_count"] for r in rows]))
        print("  skeleton_pixels mean:", mean([r["skeleton_pixels"] for r in rows]))

    print("\nWarnings:")
    for (dataset, method, warning), count in warnings.most_common(100):
        print(f"  {dataset:22s} {method:16s} {warning:30s} {count}")

    if data.get("skipped"):
        print("\nSkipped:")
        for s in data["skipped"][:30]:
            print(s)


if __name__ == "__main__":
    main()