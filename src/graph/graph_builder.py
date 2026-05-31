from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from src.graph.graph_schema import HICSGRGraph
from src.graph.pixel_graph import build_pixel_graph


def load_mask_png(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr < 128


def build_graph_from_binary_skeleton(
    *,
    sample_id: str,
    dataset: str,
    level: str,
    source_image_path: str | None,
    feature_image_path: str,
    binary_path: str,
    skeleton_path: str,
    method: str,
    scale: float,
    threshold_info: dict[str, Any] | None = None,
    inherited_warnings: list[str] | None = None,
) -> dict[str, Any]:
    binary_mask = load_mask_png(binary_path)
    skeleton = load_mask_png(skeleton_path)

    if binary_mask.shape != skeleton.shape:
        raise ValueError(
            f"binary and skeleton shape mismatch: {binary_mask.shape} vs {skeleton.shape}"
        )

    height, width = binary_mask.shape

    width_map = distance_transform_edt(binary_mask)

    raw_graph = build_pixel_graph(
        skeleton=skeleton,
        binary_mask=binary_mask,
        width_map=width_map,
    )

    warnings = list(inherited_warnings or []) + raw_graph.warnings

    graph = HICSGRGraph(
        sample_id=sample_id,
        dataset=dataset,
        level=level,
        source_image_path=source_image_path,
        feature_image_path=feature_image_path,
        processing={
            "binarization": method,
            "skeletonization": "skimage_skeletonize",
            "pixel_graph": "degree_8_neighborhood_v1",
            "junction_clustering": "connected_degree_ge_3_pixels",
            "edge_tracing": "special_node_path_tracing_v1",
            "pruning": "mark_short_branches_only",
            "scale": float(scale),
        },
        image={
            "width": int(width),
            "height": int(height),
        },
        binary={
            "foreground_ratio": float(binary_mask.mean()),
            "threshold_method": method,
            "threshold_info": threshold_info or {},
        },
        nodes=raw_graph.nodes,
        edges=raw_graph.edges,
        components=raw_graph.components,
        loops=raw_graph.loops,
        graph_features=raw_graph.features,
        warnings=sorted(set(warnings)),
        quality={
            "graph_confidence": None,
            "graph_quality_score": None,
        },
    )

    return graph.to_dict()