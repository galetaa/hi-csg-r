from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct_change(base, new):
    if base in {None, 0} or new is None:
        return None
    return (new - base) / base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="outputs/graph_pilot_component_filter/component_filter_experiment_report.json",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = data["runs"]

    print("filtered runs:", len(rows))
    print("by dataset:", Counter(r["dataset"] for r in rows))
    print("by threshold:", Counter(r["threshold"] for r in rows))

    grouped = defaultdict(list)

    for r in rows:
        grouped[(r["dataset"], r["base_method"], r["threshold"])].append(r)

    print("\nStats by dataset / method / threshold:")
    for (dataset, method, threshold), group in sorted(grouped.items()):
        node_change = [
            pct_change(r["base_node_count"], r["filtered_node_count"])
            for r in group
        ]
        component_change = [
            pct_change(r["base_component_count"], r["filtered_component_count"])
            for r in group
        ]
        endpoint_change = [
            pct_change(r["base_endpoint_count"], r["filtered_endpoint_count"])
            for r in group
        ]
        skeleton_change = [
            pct_change(r["base_skeleton_pixels"], r["filtered_skeleton_pixels"])
            for r in group
        ]

        print(f"\n{dataset} / {method} / min_skel={threshold}")
        print("  n:", len(group))
        print("  removed_pixel_ratio mean:", mean([r["removed_pixel_ratio"] for r in group]))
        print("  node_count change mean:", mean(node_change))
        print("  component_count change mean:", mean(component_change))
        print("  endpoint_count change mean:", mean(endpoint_change))
        print("  skeleton_pixels change mean:", mean(skeleton_change))

    print("\nTop filtered overlays to review:")
    ranked = sorted(
        rows,
        key=lambda r: (
            r["removed_pixel_ratio"],
            r["base_component_count"] or 0,
        ),
        reverse=True,
    )

    for r in ranked[:20]:
        print(
            r["dataset"],
            r["base_method"],
            "thr=", r["threshold"],
            "removed=", round(r["removed_pixel_ratio"], 4),
            "base_comp=", r["base_component_count"],
            "filtered_comp=", r["filtered_component_count"],
            "overlay=", r["filtered_overlay_path"],
        )


if __name__ == "__main__":
    main()