from __future__ import annotations

from typing import Any


PAGE_STRESS_DATASETS = {"hwr200", "hkr_forms"}


def safe_div(a: float | int | None, b: float | int | None) -> float | None:
    if a is None or b in {None, 0}:
        return None
    return float(a) / float(b)


def warning_risk_score(warnings: list[str]) -> float:
    score = 0.0

    for w in warnings:
        if w in {"skeleton_empty", "empty_foreground", "graph_build_failed"}:
            score += 100
        elif w in {"too_many_components", "too_many_junctions"}:
            score += 25
        elif w in {"too_high_foreground_ratio", "too_low_foreground_ratio"}:
            score += 18
        elif w in {"too_many_short_branches", "no_special_nodes_detected"}:
            score += 15
        elif w in {"large_page_scaled", "hkr_possible_form_grid", "hwr200_page", "hkr_forms_page"}:
            score += 5
        else:
            score += 2

    return score


def compute_normalized_graph_metrics(run: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    features = graph.get("graph_features", {})
    image = graph.get("image", {})
    binary = graph.get("binary", {})
    warnings = graph.get("warnings", [])

    width = image.get("width")
    height = image.get("height")
    area = (width or 0) * (height or 0) if width and height else None

    skeleton_pixels = features.get("skeleton_pixels")
    node_count = features.get("node_count")
    edge_count = features.get("edge_count")
    component_count = features.get("component_count")
    junction_count = features.get("junction_count")
    endpoint_count = features.get("endpoint_count")
    short_branch_count = features.get("short_branch_count")
    loop_candidate_count = features.get("loop_candidate_count")

    risk = warning_risk_score(warnings)

    return {
        "pilot_id": run.get("pilot_id"),
        "sample_id": run.get("sample_id"),
        "dataset": run.get("dataset"),
        "level": run.get("level"),
        "method": run.get("method"),
        "graph_path": run.get("graph_path"),
        "overlay_path": run.get("overlay_path"),

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

        "mean_width_proxy": features.get("mean_width_proxy"),
        "warning_count": len(warnings),
        "warning_risk_score": risk,
        "warnings": warnings,

        "is_page_stress_dataset": run.get("dataset") in PAGE_STRESS_DATASETS,
    }