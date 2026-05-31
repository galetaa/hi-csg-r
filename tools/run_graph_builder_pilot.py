from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.graph.graph_builder import build_graph_from_binary_skeleton
from src.graph.graph_io import write_json
from src.visualization.draw_graph import draw_graph_overlay


VARIANTS_BY_DATASET = {
    "iam": ["otsu"],
    "cyrillic_handwriting": ["otsu"],
    "hwr200": ["otsu", "otsu_gridless"],
    "hkr_forms": ["otsu", "otsu_gridless"],
}


def load_summaries(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_limited(summaries: list[dict], max_per_dataset: int) -> list[dict]:
    counts = Counter()
    out = []

    for s in summaries:
        dataset = s["dataset"]
        if counts[dataset] >= max_per_dataset:
            continue

        out.append(s)
        counts[dataset] += 1

    return out


def method_paths(summary: dict, method: str) -> tuple[Path, Path] | None:
    out_dir = Path(summary["output_dir"])
    binary_path = out_dir / f"binary_{method}.png"
    skeleton_path = out_dir / f"skeleton_{method}.png"

    if not binary_path.exists() or not skeleton_path.exists():
        return None

    return binary_path, skeleton_path


def process_one_graph(task: tuple[dict, str, int]) -> tuple[str, dict]:
    summary, method, max_skeleton_pixels = task

    dataset = summary["dataset"]

    paths = method_paths(summary, method)
    if paths is None:
        return "skipped", {
            "pilot_id": summary["pilot_id"],
            "dataset": dataset,
            "method": method,
            "reason": "missing_binary_or_skeleton",
        }

    method_diag = summary["methods"].get(method, {})
    skeleton_pixels = method_diag.get("skeleton_pixels")

    if skeleton_pixels is not None and skeleton_pixels > max_skeleton_pixels:
        return "skipped", {
            "pilot_id": summary["pilot_id"],
            "dataset": dataset,
            "method": method,
            "reason": "too_many_skeleton_pixels",
            "skeleton_pixels": skeleton_pixels,
        }

    binary_path, skeleton_path = paths
    out_dir = Path(summary["output_dir"])

    graph_path = out_dir / f"graph_{method}.json"
    overlay_path = out_dir / f"graph_overlay_{method}.png"

    try:
        graph = build_graph_from_binary_skeleton(
            sample_id=summary["sample_id"],
            dataset=dataset,
            level=summary["level"],
            source_image_path=summary.get("input_image_path"),
            feature_image_path=str(out_dir / "feature_scaled.png"),
            binary_path=str(binary_path),
            skeleton_path=str(skeleton_path),
            method=method,
            scale=float(summary.get("scale", 1.0)),
            threshold_info=method_diag.get("threshold_info", {}),
            inherited_warnings=method_diag.get("warnings", []),
        )

        write_json(graph, graph_path)

        draw_graph_overlay(
            feature_image_path=out_dir / "feature_scaled.png",
            graph=graph,
            out_path=overlay_path,
            edge_width=1,
            endpoint_radius=2,
            junction_radius=2,
            other_node_radius=2,
        )

        return "run", {
            "pilot_id": summary["pilot_id"],
            "sample_id": summary["sample_id"],
            "dataset": dataset,
            "level": summary["level"],
            "method": method,
            "graph_path": str(graph_path),
            "overlay_path": str(overlay_path),
            "node_count": graph["graph_features"].get("node_count"),
            "edge_count": graph["graph_features"].get("edge_count"),
            "component_count": graph["graph_features"].get("component_count"),
            "junction_count": graph["graph_features"].get("junction_count"),
            "endpoint_count": graph["graph_features"].get("endpoint_count"),
            "skeleton_pixels": graph["graph_features"].get("skeleton_pixels"),
            "warnings": graph.get("warnings", []),
        }

    except Exception as exc:
        return "skipped", {
            "pilot_id": summary["pilot_id"],
            "dataset": dataset,
            "method": method,
            "reason": "graph_build_failed",
            "error": repr(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary_summary",
        default="outputs/graph_pilot/binary_skeleton_pilot_summary.json",
    )
    parser.add_argument("--max_per_dataset", type=int, default=3)
    parser.add_argument("--max_skeleton_pixels", type=int, default=180000)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes. Use 1 for sequential execution.",
    )
    args = parser.parse_args()

    summaries = load_summaries(args.binary_summary)
    selected = select_limited(summaries, args.max_per_dataset)

    tasks = []

    for summary in selected:
        dataset = summary["dataset"]
        variants = VARIANTS_BY_DATASET.get(dataset, ["otsu"])

        for method in variants:
            tasks.append((summary, method, args.max_skeleton_pixels))

    if args.workers <= 1:
        results = [process_one_graph(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(process_one_graph, tasks))

    run_rows = []
    skipped = []

    for status, row in results:
        if status == "run":
            run_rows.append(row)
        else:
            skipped.append(row)

    report = {
        "num_binary_summaries": len(summaries),
        "num_selected_summaries": len(selected),
        "num_graphs_built": len(run_rows),
        "num_skipped": len(skipped),
        "built_by_dataset": dict(Counter(r["dataset"] for r in run_rows)),
        "built_by_method": dict(Counter(r["method"] for r in run_rows)),
        "skipped": skipped,
        "runs": run_rows,
    }

    out_report = Path("outputs/graph_pilot/graph_builder_pilot_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "num_graphs_built": len(run_rows),
                "num_skipped": len(skipped),
                "built_by_dataset": report["built_by_dataset"],
                "built_by_method": report["built_by_method"],
                "skipped_reasons": dict(Counter(s["reason"] for s in skipped)),
                "report": str(out_report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()