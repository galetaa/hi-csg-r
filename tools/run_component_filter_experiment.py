from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.graph.component_filter import filter_skeleton_components
from src.graph.graph_builder import build_graph_from_binary_skeleton, load_mask_png
from src.graph.graph_io import write_json
from src.graph.skeletonize import skeleton_to_pil
from src.visualization.draw_graph import draw_graph_overlay


PAGE_DATASETS = {"hwr200", "hkr_forms"}


def parse_thresholds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def load_graph_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_for_component_filter_candidate(run: dict) -> float:
    score = 0.0

    score += (run.get("component_count") or 0) / 50.0
    score += (run.get("endpoint_count") or 0) / 100.0
    score += (run.get("junction_count") or 0) / 100.0

    if run["dataset"] in PAGE_DATASETS:
        score += 50.0

    if run["method"].endswith("gridless"):
        score += 20.0

    return score


def select_runs(runs: list[dict], max_per_dataset: int) -> list[dict]:
    by_dataset = defaultdict(list)

    for r in runs:
        by_dataset[r["dataset"]].append(r)

    selected = []

    for dataset, rows in by_dataset.items():
        ranked = sorted(
            rows,
            key=score_for_component_filter_candidate,
            reverse=True,
        )
        selected.extend(ranked[:max_per_dataset])

    return selected


def paths_for_run(run: dict) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(run["graph_path"]).parent
    method = run["method"]

    feature_path = out_dir / "feature_scaled.png"
    binary_path = out_dir / f"binary_{method}.png"
    skeleton_path = out_dir / f"skeleton_{method}.png"

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    if not binary_path.exists():
        raise FileNotFoundError(binary_path)

    if not skeleton_path.exists():
        raise FileNotFoundError(skeleton_path)

    return out_dir, feature_path, binary_path, skeleton_path


def run_one_filter(
    *,
    run: dict,
    threshold: int,
    experiment_root: Path,
) -> dict:
    base_out_dir, feature_path, binary_path, skeleton_path = paths_for_run(run)

    skeleton = load_mask_png(skeleton_path)
    result = filter_skeleton_components(
        skeleton=skeleton,
        min_component_pixels=threshold,
    )

    dataset = run["dataset"]
    pilot_id = run["pilot_id"]
    method = run["method"]
    filtered_method = f"{method}_skelfilter{threshold}"

    out_dir = experiment_root / dataset / pilot_id
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_skeleton_path = out_dir / f"skeleton_{filtered_method}.png"
    graph_path = out_dir / f"graph_{filtered_method}.json"
    overlay_path = out_dir / f"graph_overlay_{filtered_method}.png"

    skeleton_to_pil(result.skeleton).save(filtered_skeleton_path)

    graph = build_graph_from_binary_skeleton(
        sample_id=run["sample_id"],
        dataset=dataset,
        level=run["level"],
        source_image_path=None,
        feature_image_path=str(feature_path),
        binary_path=str(binary_path),
        skeleton_path=str(filtered_skeleton_path),
        method=filtered_method,
        scale=1.0,
        threshold_info={
            "base_method": method,
            "component_filter": "skeleton_connected_components",
            "min_component_pixels": threshold,
            "removed_components": result.removed_components,
            "removed_pixels": result.removed_pixels,
            "removed_pixel_ratio": result.removed_pixel_ratio,
        },
        inherited_warnings=list(run.get("warnings", [])) + ["skeleton_component_filter_experiment"],
    )

    write_json(graph, graph_path)

    draw_graph_overlay(
        feature_image_path=feature_path,
        graph=graph,
        out_path=overlay_path,
        edge_width=1,
        endpoint_radius=2,
        junction_radius=2,
        other_node_radius=2,
    )

    return {
        "pilot_id": pilot_id,
        "sample_id": run["sample_id"],
        "dataset": dataset,
        "level": run["level"],
        "base_method": method,
        "filtered_method": filtered_method,
        "threshold": threshold,
        "base_graph_path": run["graph_path"],
        "filtered_graph_path": str(graph_path),
        "filtered_overlay_path": str(overlay_path),

        "base_node_count": run.get("node_count"),
        "base_edge_count": run.get("edge_count"),
        "base_component_count": run.get("component_count"),
        "base_junction_count": run.get("junction_count"),
        "base_endpoint_count": run.get("endpoint_count"),
        "base_skeleton_pixels": run.get("skeleton_pixels"),

        "filtered_node_count": graph["graph_features"].get("node_count"),
        "filtered_edge_count": graph["graph_features"].get("edge_count"),
        "filtered_component_count": graph["graph_features"].get("component_count"),
        "filtered_junction_count": graph["graph_features"].get("junction_count"),
        "filtered_endpoint_count": graph["graph_features"].get("endpoint_count"),
        "filtered_skeleton_pixels": graph["graph_features"].get("skeleton_pixels"),

        "removed_components": result.removed_components,
        "removed_pixels": result.removed_pixels,
        "removed_pixel_ratio": result.removed_pixel_ratio,
        "warnings": graph.get("warnings", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_report",
        default="outputs/graph_pilot/graph_builder_pilot_report.json",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/graph_pilot_component_filter",
    )
    parser.add_argument("--max_per_dataset", type=int, default=6)
    parser.add_argument("--thresholds", type=str, default="4,8,16,32")
    args = parser.parse_args()

    graph_report = load_graph_report(args.graph_report)
    runs = graph_report["runs"]

    thresholds = parse_thresholds(args.thresholds)
    selected = select_runs(runs, max_per_dataset=args.max_per_dataset)

    experiment_root = Path(args.out_dir)
    experiment_root.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = []

    for run in selected:
        for threshold in thresholds:
            try:
                rows.append(
                    run_one_filter(
                        run=run,
                        threshold=threshold,
                        experiment_root=experiment_root,
                    )
                )
            except Exception as exc:
                failed.append(
                    {
                        "pilot_id": run.get("pilot_id"),
                        "dataset": run.get("dataset"),
                        "method": run.get("method"),
                        "threshold": threshold,
                        "error": repr(exc),
                    }
                )

    report = {
        "source_graph_report": str(args.graph_report),
        "thresholds": thresholds,
        "num_selected_base_runs": len(selected),
        "num_filtered_runs": len(rows),
        "num_failed": len(failed),
        "selected_base_counts": dict(Counter(r["dataset"] for r in selected)),
        "filtered_counts": dict(Counter(r["dataset"] for r in rows)),
        "failed": failed,
        "runs": rows,
    }

    report_path = experiment_root / "component_filter_experiment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "num_selected_base_runs": len(selected),
            "num_filtered_runs": len(rows),
            "num_failed": len(failed),
            "selected_base_counts": report["selected_base_counts"],
            "filtered_counts": report["filtered_counts"],
            "report": str(report_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()