from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLEAN_SUBSET = Path("outputs/graph_pilot_v2/clean_graph_subset_v1.jsonl")
STRESS_SUBSET = Path("outputs/graph_pilot_v2/page_stress_graph_subset_v1.jsonl")

OUT_CLEAN_CSV = Path("outputs/graph_pilot_v2/graph_features_clean_v1.csv")
OUT_STRESS_CSV = Path("outputs/graph_pilot_v2/graph_features_page_stress_v1.csv")
OUT_SUMMARY = Path("outputs/graph_pilot_v2/graph_features_v1_summary.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_graph(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_div(a: float | int | None, b: float | int | None) -> float | None:
    if a is None or b in {None, 0}:
        return None
    return float(a) / float(b)


def graph_feature_row(subset_record: dict[str, Any]) -> dict[str, Any]:
    graph = load_graph(subset_record["graph_path"])

    q = subset_record["quality"]
    features = graph.get("graph_features", {})
    image = graph.get("image", {})
    binary = graph.get("binary", {})

    width = image.get("width")
    height = image.get("height")
    area = width * height if width and height else None

    node_count = features.get("node_count")
    edge_count = features.get("edge_count")
    component_count = features.get("component_count")
    junction_count = features.get("junction_count")
    endpoint_count = features.get("endpoint_count")
    short_branch_count = features.get("short_branch_count")
    loop_candidate_count = features.get("loop_candidate_count")
    skeleton_pixels = features.get("skeleton_pixels")

    return {
        "subset": subset_record["subset"],
        "pilot_id": subset_record["pilot_id"],
        "sample_id": subset_record["sample_id"],
        "dataset": subset_record["dataset"],
        "level": subset_record["level"],
        "method": subset_record["method"],

        "graph_path": subset_record["graph_path"],
        "overlay_path": subset_record["overlay_path"],

        "image_width": width,
        "image_height": height,
        "image_area": area,
        "foreground_ratio": binary.get("foreground_ratio"),
        "skeleton_pixels": skeleton_pixels,
        "skeleton_density": safe_div(skeleton_pixels, area),

        "node_count": node_count,
        "edge_count": edge_count,
        "component_count": component_count,
        "junction_count": junction_count,
        "endpoint_count": endpoint_count,
        "short_branch_count": short_branch_count,
        "loop_candidate_count": loop_candidate_count,

        "nodes_per_1k_skeleton": safe_div(node_count * 1000 if node_count is not None else None, skeleton_pixels),
        "edges_per_1k_skeleton": safe_div(edge_count * 1000 if edge_count is not None else None, skeleton_pixels),
        "components_per_1k_skeleton": safe_div(component_count * 1000 if component_count is not None else None, skeleton_pixels),
        "junctions_per_1k_skeleton": safe_div(junction_count * 1000 if junction_count is not None else None, skeleton_pixels),
        "endpoints_per_1k_skeleton": safe_div(endpoint_count * 1000 if endpoint_count is not None else None, skeleton_pixels),
        "short_branches_per_1k_skeleton": safe_div(short_branch_count * 1000 if short_branch_count is not None else None, skeleton_pixels),

        "junction_endpoint_ratio": safe_div(junction_count, endpoint_count),
        "edge_node_ratio": safe_div(edge_count, node_count),
        "component_node_ratio": safe_div(component_count, node_count),
        "edges_per_component": safe_div(edge_count, component_count),
        "nodes_per_component": safe_div(node_count, component_count),

        "mean_width_proxy": features.get("mean_width_proxy"),
        "warning_count": q.get("warning_count"),
        "warning_risk_score": q.get("warning_risk_score"),
        "warnings": "|".join(q.get("warnings", [])),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subset",
        "pilot_id",
        "sample_id",
        "dataset",
        "level",
        "method",
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
        "edges_per_component",
        "nodes_per_component",
        "mean_width_proxy",
        "warning_count",
        "warning_risk_score",
        "graph_path",
        "overlay_path",
        "warnings",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def mean(xs: list[Any]) -> float | None:
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            vals.append(float(x))
        except Exception:
            continue

    if not vals:
        return None

    return sum(vals) / len(vals)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)

    metric_names = [
        "foreground_ratio",
        "skeleton_density",
        "skeleton_pixels",
        "nodes_per_1k_skeleton",
        "edges_per_1k_skeleton",
        "components_per_1k_skeleton",
        "junctions_per_1k_skeleton",
        "endpoints_per_1k_skeleton",
        "short_branches_per_1k_skeleton",
        "junction_endpoint_ratio",
        "edge_node_ratio",
        "component_node_ratio",
        "edges_per_component",
        "nodes_per_component",
        "mean_width_proxy",
        "warning_count",
        "warning_risk_score",
    ]

    out = {
        "count": len(rows),
        "by_dataset": dict(Counter(r["dataset"] for r in rows)),
        "by_method": dict(Counter(r["method"] for r in rows)),
        "metrics_by_dataset": {},
    }

    for dataset, group in sorted(by_dataset.items()):
        out["metrics_by_dataset"][dataset] = {
            "n": len(group),
            **{f"{m}_mean": mean([r.get(m) for r in group]) for m in metric_names},
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_subset", default=str(CLEAN_SUBSET))
    parser.add_argument("--stress_subset", default=str(STRESS_SUBSET))
    parser.add_argument("--out_clean_csv", default=str(OUT_CLEAN_CSV))
    parser.add_argument("--out_stress_csv", default=str(OUT_STRESS_CSV))
    parser.add_argument("--out_summary", default=str(OUT_SUMMARY))
    args = parser.parse_args()

    clean_subset = read_jsonl(Path(args.clean_subset))
    stress_subset = read_jsonl(Path(args.stress_subset))

    clean_rows = [graph_feature_row(r) for r in clean_subset]
    stress_rows = [graph_feature_row(r) for r in stress_subset]

    write_csv(clean_rows, Path(args.out_clean_csv))
    write_csv(stress_rows, Path(args.out_stress_csv))

    summary = {
        "clean_features": {
            "path": args.out_clean_csv,
            **summarize_rows(clean_rows),
        },
        "page_stress_features": {
            "path": args.out_stress_csv,
            **summarize_rows(stress_rows),
        },
        "notes": [
            "graph_features_clean_v1.csv is the first clean graph feature table for crop/line datasets.",
            "graph_features_page_stress_v1.csv is separate and must not be mixed with clean graph features.",
            "Features are diagnostic structural features, not ground-truth graph accuracy.",
            "mean_width_proxy is a stroke-width proxy, not true pen pressure.",
        ],
    }

    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", args.out_clean_csv)
    print("wrote:", args.out_stress_csv)
    print("wrote:", args.out_summary)
    print(json.dumps({
        "clean": {
            "count": summary["clean_features"]["count"],
            "by_dataset": summary["clean_features"]["by_dataset"],
            "by_method": summary["clean_features"]["by_method"],
        },
        "stress": {
            "count": summary["page_stress_features"]["count"],
            "by_dataset": summary["page_stress_features"]["by_dataset"],
            "by_method": summary["page_stress_features"]["by_method"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()