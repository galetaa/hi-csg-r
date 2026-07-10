from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.graph.quality_metrics import compute_normalized_graph_metrics


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


def load_graph(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_primary(row: dict) -> bool:
    return PRIMARY_VARIANTS.get(row["dataset"]) == row["method"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_report",
        default="outputs/graph_pilot_v2/graph_builder_pilot_report.json",
    )
    parser.add_argument(
        "--out_csv",
        default="outputs/graph_pilot_v2/graph_quality_metrics_v1.csv",
    )
    parser.add_argument(
        "--out_summary",
        default="outputs/graph_pilot_v2/graph_quality_metrics_v1_summary.json",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.graph_report).read_text(encoding="utf-8"))
    runs = report["runs"]

    rows = []

    for run in runs:
        graph = load_graph(run["graph_path"])
        row = compute_normalized_graph_metrics(run, graph)
        row["is_primary_variant"] = is_primary(row)
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pilot_id",
        "sample_id",
        "dataset",
        "level",
        "method",
        "is_primary_variant",
        "is_page_stress_dataset",
        "image_width",
        "image_height",
        "image_area",
        "foreground_ratio",
        "skeleton_pixels",
        "skeleton_density",
        "node_count",
        "edge_count",
        "component_count",
        "junction_count",
        "endpoint_count",
        "short_branch_count",
        "loop_candidate_count",
        "nodes_per_1k_skeleton",
        "edges_per_1k_skeleton",
        "components_per_1k_skeleton",
        "junctions_per_1k_skeleton",
        "endpoints_per_1k_skeleton",
        "short_branches_per_1k_skeleton",
        "junction_endpoint_ratio",
        "edge_node_ratio",
        "component_node_ratio",
        "mean_width_proxy",
        "warning_count",
        "warning_risk_score",
        "graph_path",
        "overlay_path",
        "warnings",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = dict(row)
            out["warnings"] = "|".join(out.get("warnings", []))
            writer.writerow({k: out.get(k) for k in fieldnames})

    grouped = defaultdict(list)
    primary_grouped = defaultdict(list)

    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
        if row["is_primary_variant"]:
            primary_grouped[row["dataset"]].append(row)

    metric_names = [
        "foreground_ratio",
        "skeleton_density",
        "nodes_per_1k_skeleton",
        "edges_per_1k_skeleton",
        "components_per_1k_skeleton",
        "junctions_per_1k_skeleton",
        "endpoints_per_1k_skeleton",
        "junction_endpoint_ratio",
        "edge_node_ratio",
        "warning_count",
        "warning_risk_score",
    ]

    by_dataset_method = {}
    for (dataset, method), group in sorted(grouped.items()):
        by_dataset_method[f"{dataset}/{method}"] = {
            "n": len(group),
            **{f"{m}_mean": mean([r.get(m) for r in group]) for m in metric_names},
        }

    primary_by_dataset = {}
    for dataset, group in sorted(primary_grouped.items()):
        primary_by_dataset[dataset] = {
            "primary_method": PRIMARY_VARIANTS.get(dataset),
            "n": len(group),
            **{f"{m}_mean": mean([r.get(m) for r in group]) for m in metric_names},
        }

    summary = {
        "num_rows": len(rows),
        "primary_variants": PRIMARY_VARIANTS,
        "rows_by_dataset": dict(Counter(r["dataset"] for r in rows)),
        "rows_by_method": dict(Counter(r["method"] for r in rows)),
        "rows_primary_by_dataset": dict(Counter(r["dataset"] for r in rows if r["is_primary_variant"])),
        "by_dataset_method": by_dataset_method,
        "primary_by_dataset": primary_by_dataset,
        "notes": [
            "These are diagnostic normalized graph metrics, not ground-truth graph accuracy.",
            "Raw node/edge/component counts should not be compared across levels without normalization.",
            "Page stress datasets are expected to have high component and junction complexity.",
        ],
    }

    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_csv)
    print("wrote:", args.out_summary)
    print(json.dumps({
        "num_rows": len(rows),
        "rows_by_dataset": summary["rows_by_dataset"],
        "rows_by_method": summary["rows_by_method"],
        "rows_primary_by_dataset": summary["rows_primary_by_dataset"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()