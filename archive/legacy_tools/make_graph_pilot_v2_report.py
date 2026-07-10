from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


PILOT_SUMMARY = Path("data/pilot/graph_pilot_v2_summary.json")
BINARY_SUMMARY = Path("outputs/graph_pilot_v2/binary_skeleton_pilot_summary.json")
GRAPH_REPORT = Path("outputs/graph_pilot_v2/graph_builder_pilot_report.json")
OUT = Path("outputs/graph_pilot_v2/graph_pilot_v2_report.md")


PRIMARY_VARIANTS = {
    "iam": "otsu",
    "cyrillic_handwriting": "otsu",
    "hkr_words": "otsu",
    "school_notebooks": "sauvola",
    "hwr200": "otsu_gridless",
    "hkr_forms": "otsu_gridless",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt_float(x, ndigits: int = 6):
    if x is None:
        return "None"
    return str(round(float(x), ndigits))


def main() -> None:
    pilot = json.loads(PILOT_SUMMARY.read_text(encoding="utf-8"))
    binary = json.loads(BINARY_SUMMARY.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH_REPORT.read_text(encoding="utf-8"))

    graph_runs = graph["runs"]
    skipped = graph.get("skipped", [])

    binary_by_dataset_method = defaultdict(list)
    binary_warnings = Counter()

    for r in binary:
        dataset = r["dataset"]
        for method, d in r["methods"].items():
            if "foreground_ratio" in d:
                binary_by_dataset_method[(dataset, method)].append(d)
            for w in d.get("warnings", []):
                binary_warnings[(dataset, method, w)] += 1

    graph_by_dataset_method = defaultdict(list)
    graph_warnings = Counter()

    for r in graph_runs:
        graph_by_dataset_method[(r["dataset"], r["method"])].append(r)
        for w in r.get("warnings", []):
            graph_warnings[(r["dataset"], r["method"], w)] += 1

    lines = []
    lines.append("# Graph pilot v2 report\n")
    lines.append("## Status\n")
    lines.append("```text")
    lines.append("graph_pilot_v2 is the main expanded graph pilot.")
    lines.append("It includes IAM, Cyrillic Handwriting, HKR Words, School Notebooks, HWR200, and HKR Forms.")
    lines.append("```")
    lines.append("")

    lines.append("## Pilot composition\n")
    lines.append("```json")
    lines.append(json.dumps({
        "num_records": pilot.get("num_records"),
        "by_dataset": pilot.get("by_dataset"),
        "by_level": pilot.get("by_level"),
        "by_split": pilot.get("by_split"),
        "school_categories": pilot.get("school_categories"),
        "hwr200_conditions": pilot.get("hwr200_conditions"),
    }, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Graph-builder summary\n")
    lines.append("```json")
    lines.append(json.dumps({
        "num_binary_summaries": graph.get("num_binary_summaries"),
        "num_selected_summaries": graph.get("num_selected_summaries"),
        "num_graphs_built": graph.get("num_graphs_built"),
        "num_skipped": graph.get("num_skipped"),
        "built_by_dataset": graph.get("built_by_dataset"),
        "built_by_method": graph.get("built_by_method"),
    }, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Primary graph variants\n")
    lines.append("```json")
    lines.append(json.dumps(PRIMARY_VARIANTS, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Methodological decisions\n")
    lines.append("- IAM: primary graph variant is `otsu`.\n")
    lines.append("- Cyrillic Handwriting: primary graph variant is `otsu`.\n")
    lines.append("- HKR Words: primary graph variant is `otsu`.\n")
    lines.append("- School Notebooks: primary graph variant is `sauvola`; Otsu often over-selects polygon/background foreground.\n")
    lines.append("- HWR200: `otsu_gridless` is a diagnostic page-level stress-test variant, not a clean canonical graph setting.\n")
    lines.append("- HKR Forms: `otsu_gridless` is a diagnostic page/form stress-test variant, not a clean canonical graph setting.\n")
    lines.append("- `adaptive_gaussian` is not used for graph-builder because it tends to increase skeleton noise.\n")
    lines.append("- Component filtering is not used by default on word/line/polygon crops.\n")
    lines.append("- `min_skel=8` is only a diagnostic page-noise filter for page/form datasets.\n")
    lines.append("- Raw node/edge/component counts must not be compared across word/line/page levels without normalization.\n")
    lines.append("")

    lines.append("## Binary/skeleton statistics\n")
    for (dataset, method), rows in sorted(binary_by_dataset_method.items()):
        lines.append(f"### {dataset} / {method}")
        lines.append("```text")
        lines.append(f"n: {len(rows)}")
        lines.append(f"foreground_ratio_mean: {fmt_float(mean([r.get('foreground_ratio') for r in rows]))}")
        lines.append(f"foreground_pixels_mean: {fmt_float(mean([r.get('foreground_pixels') for r in rows]), 3)}")
        lines.append(f"skeleton_pixels_mean: {fmt_float(mean([r.get('skeleton_pixels') for r in rows]), 3)}")
        lines.append(f"grid_removed_ratio_mean: {fmt_float(mean([r.get('grid_removed_ratio') for r in rows]))}")
        lines.append("```")
        lines.append("")

    lines.append("## Graph statistics\n")
    for (dataset, method), rows in sorted(graph_by_dataset_method.items()):
        lines.append(f"### {dataset} / {method}")
        lines.append("```text")
        lines.append(f"n: {len(rows)}")
        lines.append(f"node_count_mean: {fmt_float(mean([r.get('node_count') for r in rows]), 3)}")
        lines.append(f"edge_count_mean: {fmt_float(mean([r.get('edge_count') for r in rows]), 3)}")
        lines.append(f"component_count_mean: {fmt_float(mean([r.get('component_count') for r in rows]), 3)}")
        lines.append(f"junction_count_mean: {fmt_float(mean([r.get('junction_count') for r in rows]), 3)}")
        lines.append(f"endpoint_count_mean: {fmt_float(mean([r.get('endpoint_count') for r in rows]), 3)}")
        lines.append(f"skeleton_pixels_mean: {fmt_float(mean([r.get('skeleton_pixels') for r in rows]), 3)}")
        lines.append("```")
        lines.append("")

    lines.append("## Binary warnings\n")
    lines.append("```text")
    for (dataset, method, warning), count in binary_warnings.most_common(150):
        lines.append(f"{dataset:24s} {method:18s} {warning:34s} {count}")
    lines.append("```")
    lines.append("")

    lines.append("## Graph warnings\n")
    lines.append("```text")
    for (dataset, method, warning), count in graph_warnings.most_common(150):
        lines.append(f"{dataset:24s} {method:18s} {warning:34s} {count}")
    lines.append("```")
    lines.append("")

    lines.append("## Skipped graph builds\n")
    lines.append("```json")
    lines.append(json.dumps(skipped, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append("- Clean crop/line datasets already produce usable graph overlays.\n")
    lines.append("- Page-level HWR200/HKR Forms remain stress-test datasets because grid/form/background structures dominate graph complexity.\n")
    lines.append("- School Notebooks is a valid polygon crop graph dataset, but its masked crop geometry makes local thresholding preferable.\n")
    lines.append("- The next step is to export normalized graph quality metrics, not only raw graph counts.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print("wrote:", OUT)


if __name__ == "__main__":
    main()