from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


BINARY_SUMMARY = Path("outputs/graph_pilot/binary_skeleton_pilot_summary.json")
GRAPH_REPORT = Path("outputs/graph_pilot/graph_builder_pilot_report.json")
OUT = Path("outputs/graph_pilot/graph_pilot_report.md")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    binary = json.loads(BINARY_SUMMARY.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH_REPORT.read_text(encoding="utf-8"))

    runs = graph["runs"]
    skipped = graph["skipped"]

    binary_by_dataset_method = defaultdict(list)
    for r in binary:
        dataset = r["dataset"]
        for method, d in r["methods"].items():
            if "foreground_ratio" in d:
                binary_by_dataset_method[(dataset, method)].append(d)

    graph_by_dataset_method = defaultdict(list)
    warnings = Counter()

    for r in runs:
        graph_by_dataset_method[(r["dataset"], r["method"])].append(r)
        for w in r.get("warnings", []):
            warnings[(r["dataset"], r["method"], w)] += 1

    lines = []
    lines.append("# Graph pilot report\n")
    lines.append("## Summary\n")
    lines.append("```text")
    lines.append(f"binary/skeleton records: {len(binary)}")
    lines.append(f"graphs built: {graph['num_graphs_built']}")
    lines.append(f"graphs skipped: {graph['num_skipped']}")
    lines.append("```")
    lines.append("")

    lines.append("## Built by dataset\n")
    lines.append("```json")
    lines.append(json.dumps(graph["built_by_dataset"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Built by method\n")
    lines.append("```json")
    lines.append(json.dumps(graph["built_by_method"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Binary/skeleton statistics\n")
    for (dataset, method), rows in sorted(binary_by_dataset_method.items()):
        lines.append(f"### {dataset} / {method}")
        lines.append("```text")
        lines.append(f"n: {len(rows)}")
        lines.append(f"foreground_ratio_mean: {mean([r.get('foreground_ratio') for r in rows])}")
        lines.append(f"foreground_pixels_mean: {mean([r.get('foreground_pixels') for r in rows])}")
        lines.append(f"skeleton_pixels_mean: {mean([r.get('skeleton_pixels') for r in rows])}")
        lines.append(f"grid_removed_ratio_mean: {mean([r.get('grid_removed_ratio') for r in rows])}")
        lines.append("```")
        lines.append("")

    lines.append("## Graph statistics\n")
    for (dataset, method), rows in sorted(graph_by_dataset_method.items()):
        lines.append(f"### {dataset} / {method}")
        lines.append("```text")
        lines.append(f"n: {len(rows)}")
        lines.append(f"node_count_mean: {mean([r.get('node_count') for r in rows])}")
        lines.append(f"edge_count_mean: {mean([r.get('edge_count') for r in rows])}")
        lines.append(f"component_count_mean: {mean([r.get('component_count') for r in rows])}")
        lines.append(f"junction_count_mean: {mean([r.get('junction_count') for r in rows])}")
        lines.append(f"endpoint_count_mean: {mean([r.get('endpoint_count') for r in rows])}")
        lines.append(f"skeleton_pixels_mean: {mean([r.get('skeleton_pixels') for r in rows])}")
        lines.append("```")
        lines.append("")

    lines.append("## Warnings\n")
    lines.append("```text")
    for (dataset, method, warning), count in warnings.most_common(100):
        lines.append(f"{dataset:24s} {method:18s} {warning:32s} {count}")
    lines.append("```")
    lines.append("")

    lines.append("## Skipped graphs\n")
    lines.append("```json")
    lines.append(json.dumps(skipped, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Current decisions\n")
    lines.append("- IAM: `otsu` is the primary graph-builder variant.\n")
    lines.append("- Cyrillic Handwriting: `otsu` is the primary graph-builder variant.\n")
    lines.append("- HWR200: `otsu` and `otsu_gridless` are diagnostic variants; full-page graph is a stress-test.\n")
    lines.append("- HKR Forms: `otsu` and `otsu_gridless` are diagnostic variants; form/grid background remains a major factor.\n")
    lines.append("- `adaptive_gaussian` is not used for graph-builder pilot because it tends to produce more skeleton noise.\n")
    lines.append("- Raw graph counts must not be compared across word/line/page levels without normalization.\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()