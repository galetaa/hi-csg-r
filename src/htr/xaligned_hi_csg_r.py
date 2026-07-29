from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import remove_small_objects, skeletonize
from src.graph.pixel_graph import build_pixel_graph

FEATURE_VERSION = "hi_csg_r_xaligned_v1"
FEATURE_BUILDER_VERSION = "1.0.0"
GRAPH_VERSION = "hi_csg_r_v1"
WIDTH_DOWNSAMPLE = 4

FEATURE_NAMES = (
    "ink_fraction",
    "skeleton_density",
    "edge_length_density",
    "stroke_width_mean",
    "stroke_width_std",
    "curvature_mean",
    "orientation_horizontal",
    "orientation_vertical",
    "orientation_diag_pos",
    "orientation_diag_neg",
    "node_density",
    "endpoint_density",
    "junction_density",
    "loop_edge_fraction",
    "component_count_norm",
    "short_branch_fraction",
    "boundary_crossings_norm",
    "ambiguous_edge_fraction",
    "graph_occupancy",
    "warning_density",
)
QUALITY_FEATURE_NAMES = (
    "ambiguous_edge_fraction",
    "graph_occupancy",
    "warning_density",
)
QUALITY_FEATURE_INDICES = tuple(FEATURE_NAMES.index(name) for name in QUALITY_FEATURE_NAMES)
TOPOLOGY_START_INDEX = 10

DATASET_BINARIZATION = {
    "cyrillic_handwriting": "otsu",
    "hkr_words": "otsu",
    "school_notebooks": "sauvola",
    "school_notebooks_clean": "sauvola",
    "school_notebooks_line": "sauvola",
}

_AMBIGUOUS_MARKERS = ("ambiguous", "uncertain", "warning")
_JUNCTION_TYPES = {"junction", "junction_cluster"}
_LOOP_EDGE_TYPES = {"loop_segment", "isolated_loop"}


def compute_output_steps(width: int, width_downsample: int = WIDTH_DOWNSAMPLE) -> int:
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if width_downsample <= 0:
        raise ValueError(f"width_downsample must be positive, got {width_downsample}")
    return max(int(width) // int(width_downsample), 1)


def file_sha1(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(value: str | Path, reference: str | Path | None = None) -> Path:
    path = Path(value)
    candidates = [path]
    if reference is not None:
        candidates.append(Path(reference).resolve().parent / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(str(value))


def binarize_handwriting(
    gray: np.ndarray,
    dataset: str,
    *,
    method: str | None = None,
    sauvola_window: int = 25,
    min_object_size: int = 4,
) -> tuple[np.ndarray, str]:
    if gray.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape={gray.shape}")
    if gray.size == 0:
        raise ValueError("Cannot binarize an empty image")

    resolved_method = method or DATASET_BINARIZATION.get(dataset, "otsu")
    if resolved_method == "otsu":
        try:
            threshold = threshold_otsu(gray)
        except ValueError:
            threshold = 127
        foreground = gray < threshold
    elif resolved_method == "sauvola":
        min_dim = min(gray.shape)
        if min_dim < 7:
            try:
                threshold = threshold_otsu(gray)
            except ValueError:
                threshold = 127
            foreground = gray < threshold
        else:
            window = min(sauvola_window, min_dim if min_dim % 2 else min_dim - 1)
            window = max(7, window)
            if window % 2 == 0:
                window -= 1
            threshold = threshold_sauvola(gray, window_size=window)
            foreground = gray < threshold
    else:
        raise ValueError(f"Unsupported binarization method: {resolved_method}")

    foreground = foreground.astype(bool)
    if float(foreground.mean()) > 0.5:
        foreground = ~foreground
    if min_object_size > 0 and foreground.any():
        try:
            # scikit-image >= 0.26 removes objects <= max_size.
            foreground = remove_small_objects(
                foreground,
                max_size=max(min_object_size - 1, 0),
            )
        except TypeError:
            # Compatibility with the project lower bound (scikit-image 0.23).
            foreground = remove_small_objects(foreground, min_size=min_object_size)
    return foreground.astype(bool), resolved_method


def graph_from_masks(
    foreground: np.ndarray,
    skeleton: np.ndarray | None = None,
) -> dict[str, Any]:
    foreground = np.asarray(foreground, dtype=bool)
    skeleton_mask = (
        skeletonize(foreground) if skeleton is None else np.asarray(skeleton, dtype=bool)
    )
    if foreground.shape != skeleton_mask.shape:
        raise ValueError("foreground and skeleton shapes differ")

    width_map = ndi.distance_transform_edt(foreground)
    result = build_pixel_graph(
        skeleton=skeleton_mask,
        binary_mask=foreground,
        width_map=width_map,
    )
    height, width = foreground.shape
    return {
        "schema_version": GRAPH_VERSION,
        "image": {"width": int(width), "height": int(height)},
        "nodes": result.nodes,
        "edges": result.edges,
        "components": result.components,
        "loops": result.loops,
        "warnings": result.warnings,
        "graph_features": result.features,
    }


def load_or_extract_graph(
    image_path: str | Path,
    dataset: str,
    *,
    graph_path: str | Path | None = None,
    binarization: str | None = None,
    sauvola_window: int = 25,
    min_object_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    image_file = Path(image_path)
    with Image.open(image_file) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    foreground, method = binarize_handwriting(
        gray,
        dataset,
        method=binarization,
        sauvola_window=sauvola_window,
        min_object_size=min_object_size,
    )
    skeleton_mask = skeletonize(foreground).astype(bool)

    if graph_path is None:
        graph = graph_from_masks(foreground, skeleton_mask)
    else:
        graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        graph_shape = (
            int(graph.get("image", {}).get("height", gray.shape[0])),
            int(graph.get("image", {}).get("width", gray.shape[1])),
        )
        if graph_shape != gray.shape:
            raise ValueError(
                f"Graph/image shape mismatch for {image_file}: {graph_shape} vs {gray.shape}"
            )
    return foreground, skeleton_mask, graph, method


def _node_x_norm(node: dict[str, Any], image_width: int) -> float:
    value = node.get("x_norm")
    if value is None:
        value = float(node.get("x", 0.0)) / max(image_width - 1, 1)
    return float(np.clip(float(value), 0.0, 1.0))


def _bin_index(x_norm: float, time_steps: int) -> int:
    return min(max(int(math.floor(float(x_norm) * time_steps)), 0), time_steps - 1)


def assign_nodes_to_bins(
    graph: dict[str, Any],
    time_steps: int,
    *,
    image_width: int | None = None,
) -> dict[str, Any]:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    width = image_width or int(graph.get("image", {}).get("width", 1))
    all_counts = np.zeros(time_steps, dtype=np.float64)
    endpoint_counts = np.zeros(time_steps, dtype=np.float64)
    junction_counts = np.zeros(time_steps, dtype=np.float64)
    warning_counts = np.zeros(time_steps, dtype=np.float64)
    confidence_sum = np.zeros(time_steps, dtype=np.float64)

    for node in graph.get("nodes", []):
        index = _bin_index(_node_x_norm(node, width), time_steps)
        all_counts[index] += 1.0
        confidence_sum[index] += float(node.get("confidence", 1.0) or 0.0)
        node_type = str(node.get("type", ""))
        if node_type == "endpoint":
            endpoint_counts[index] += 1.0
        if node_type in _JUNCTION_TYPES:
            junction_counts[index] += 1.0
        warning_counts[index] += float(len(node.get("flags") or []))

    return {
        "node_count": all_counts,
        "endpoint_count": endpoint_counts,
        "junction_count": junction_counts,
        "warning_count": warning_counts,
        "confidence_sum": confidence_sum,
    }


def _edge_is_ambiguous(edge: dict[str, Any]) -> bool:
    edge_type = str(edge.get("type", "")).lower()
    flags = " ".join(str(value).lower() for value in (edge.get("flags") or []))
    return edge_type == "uncertain_edge" or any(marker in flags for marker in _AMBIGUOUS_MARKERS)


def _edge_is_short_branch(edge: dict[str, Any]) -> bool:
    return str(edge.get("type", "")) == "short_branch" or "short_branch" in (
        edge.get("flags") or []
    )


def _orientation_index(dx: float, dy: float) -> int:
    angle = math.atan2(dy, dx)
    abs_cos = abs(math.cos(angle))
    abs_sin = abs(math.sin(angle))
    if abs_cos >= math.cos(math.pi / 8):
        return 0
    if abs_sin >= math.cos(math.pi / 8):
        return 1
    return 2 if dx * dy >= 0 else 3


def _point_width(width_map: np.ndarray | None, x: float, y: float, fallback: float) -> float:
    if width_map is None:
        return fallback
    iy = min(max(int(round(y)), 0), width_map.shape[0] - 1)
    ix = min(max(int(round(x)), 0), width_map.shape[1] - 1)
    return float(2.0 * width_map[iy, ix])


def distribute_edges_to_bins(
    graph: dict[str, Any],
    time_steps: int,
    *,
    image_width: int | None = None,
    width_map: np.ndarray | None = None,
) -> dict[str, Any]:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    width = image_width or int(graph.get("image", {}).get("width", 1))
    length = np.zeros(time_steps, dtype=np.float64)
    width_sum = np.zeros(time_steps, dtype=np.float64)
    width_sq_sum = np.zeros(time_steps, dtype=np.float64)
    orientation_length = np.zeros((time_steps, 4), dtype=np.float64)
    loop_length = np.zeros(time_steps, dtype=np.float64)
    short_length = np.zeros(time_steps, dtype=np.float64)
    ambiguous_length = np.zeros(time_steps, dtype=np.float64)
    warning_length = np.zeros(time_steps, dtype=np.float64)
    curvature_sum = np.zeros(time_steps, dtype=np.float64)
    curvature_weight = np.zeros(time_steps, dtype=np.float64)
    boundary_crossings = np.zeros(time_steps, dtype=np.float64)
    components: list[set[str]] = [set() for _ in range(time_steps)]

    loop_edge_ids = {
        str(edge_id)
        for loop in graph.get("loops", [])
        for edge_id in (loop.get("edge_ids") or [])
    }

    for edge in graph.get("edges", []):
        points = edge.get("points") or []
        if len(points) < 2:
            continue
        component_id = str(edge.get("component_id") or "")
        is_loop = str(edge.get("id")) in loop_edge_ids or str(edge.get("type")) in _LOOP_EDGE_TYPES
        is_short = _edge_is_short_branch(edge)
        is_ambiguous = _edge_is_ambiguous(edge)
        flag_count = len(edge.get("flags") or [])
        fallback_width = float(edge.get("width_mean") or 0.0)
        edge_bins: set[int] = set()

        for point_index, (left, right) in enumerate(zip(points, points[1:], strict=False)):
            x1, y1 = float(left[0]), float(left[1])
            x2, y2 = float(right[0]), float(right[1])
            dx, dy = x2 - x1, y2 - y1
            segment_length = math.hypot(dx, dy)
            if segment_length <= 0:
                continue
            midpoint_x = (x1 + x2) / 2.0
            midpoint_y = (y1 + y2) / 2.0
            index = _bin_index(midpoint_x / max(width - 1, 1), time_steps)
            edge_bins.add(index)
            length[index] += segment_length
            segment_width = _point_width(width_map, midpoint_x, midpoint_y, fallback_width)
            width_sum[index] += segment_width * segment_length
            width_sq_sum[index] += segment_width * segment_width * segment_length
            orientation_length[index, _orientation_index(dx, dy)] += segment_length
            if is_loop:
                loop_length[index] += segment_length
            if is_short:
                short_length[index] += segment_length
            if is_ambiguous:
                ambiguous_length[index] += segment_length
            if flag_count:
                warning_length[index] += segment_length * flag_count
            if component_id:
                components[index].add(component_id)

            if 0 < point_index < len(points) - 1:
                previous = points[point_index - 1]
                ax, ay = x1 - float(previous[0]), y1 - float(previous[1])
                a_norm = math.hypot(ax, ay)
                b_norm = segment_length
                if a_norm > 0 and b_norm > 0:
                    cosine = float(np.clip((ax * dx + ay * dy) / (a_norm * b_norm), -1.0, 1.0))
                    curvature = abs(math.acos(cosine))
                    vertex_index = _bin_index(x1 / max(width - 1, 1), time_steps)
                    weight = 0.5 * (a_norm + b_norm)
                    curvature_sum[vertex_index] += curvature * weight
                    curvature_weight[vertex_index] += weight

        if edge_bins:
            first_bin = min(edge_bins)
            last_bin = max(edge_bins)
            for boundary in range(first_bin, last_bin):
                boundary_crossings[boundary] += 1.0
                boundary_crossings[boundary + 1] += 1.0

    return {
        "length": length,
        "width_sum": width_sum,
        "width_sq_sum": width_sq_sum,
        "orientation_length": orientation_length,
        "loop_length": loop_length,
        "short_length": short_length,
        "ambiguous_length": ambiguous_length,
        "warning_length": warning_length,
        "curvature_sum": curvature_sum,
        "curvature_weight": curvature_weight,
        "boundary_crossings": boundary_crossings,
        "component_count": np.asarray([len(value) for value in components], dtype=np.float64),
    }


def _pixel_bin_statistics(mask: np.ndarray, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    counts = np.zeros(time_steps, dtype=np.float64)
    areas = np.zeros(time_steps, dtype=np.float64)
    for index in range(time_steps):
        x0 = int(math.floor(index * width / time_steps))
        x1 = int(math.floor((index + 1) * width / time_steps))
        x1 = max(x1, x0 + 1)
        x1 = min(x1, width)
        counts[index] = float(mask[:, x0:x1].sum())
        areas[index] = float(height * max(x1 - x0, 1))
    return counts, areas


def aggregate_bin_features(
    foreground: np.ndarray,
    skeleton_mask: np.ndarray,
    graph: dict[str, Any],
    time_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    foreground = np.asarray(foreground, dtype=bool)
    skeleton_mask = np.asarray(skeleton_mask, dtype=bool)
    if foreground.shape != skeleton_mask.shape:
        raise ValueError("foreground and skeleton shapes differ")
    height, width = foreground.shape
    if time_steps != compute_output_steps(width):
        raise ValueError(
            f"time_steps={time_steps} does not match output length for width={width}"
        )

    node_stats = assign_nodes_to_bins(graph, time_steps, image_width=width)
    width_map = ndi.distance_transform_edt(foreground)
    edge_stats = distribute_edges_to_bins(
        graph,
        time_steps,
        image_width=width,
        width_map=width_map,
    )
    ink_counts, bin_areas = _pixel_bin_statistics(foreground, time_steps)
    skeleton_counts, _ = _pixel_bin_statistics(skeleton_mask, time_steps)
    bin_widths = bin_areas / max(height, 1)
    edge_length = edge_stats["length"]
    safe_length = np.maximum(edge_length, 1e-12)

    features = np.zeros((time_steps, len(FEATURE_NAMES)), dtype=np.float64)
    features[:, 0] = ink_counts / np.maximum(bin_areas, 1.0)
    features[:, 1] = skeleton_counts / np.maximum(bin_areas, 1.0)
    features[:, 2] = edge_length / np.maximum(bin_areas, 1.0)
    features[:, 3] = edge_stats["width_sum"] / safe_length
    width_variance = edge_stats["width_sq_sum"] / safe_length - features[:, 3] ** 2
    features[:, 4] = np.sqrt(np.maximum(width_variance, 0.0))
    features[:, 5] = edge_stats["curvature_sum"] / np.maximum(
        edge_stats["curvature_weight"], 1e-12
    )
    features[:, 6:10] = edge_stats["orientation_length"] / safe_length[:, None]

    count_scale = 100.0 / np.maximum(bin_widths, 1.0)
    features[:, 10] = node_stats["node_count"] * count_scale
    features[:, 11] = node_stats["endpoint_count"] * count_scale
    features[:, 12] = node_stats["junction_count"] * count_scale
    features[:, 13] = edge_stats["loop_length"] / safe_length
    features[:, 14] = edge_stats["component_count"] * count_scale
    features[:, 15] = edge_stats["short_length"] / safe_length
    features[:, 16] = edge_stats["boundary_crossings"] / 2.0
    features[:, 17] = edge_stats["ambiguous_length"] / safe_length
    features[:, 18] = np.logical_or(edge_length > 0, node_stats["node_count"] > 0)
    features[:, 19] = (
        node_stats["warning_count"] * count_scale
        + edge_stats["warning_length"] / safe_length
    )

    features[edge_length == 0, 3:10] = 0.0
    features[edge_length == 0, 13] = 0.0
    features[edge_length == 0, 15] = 0.0
    features[edge_length == 0, 17] = 0.0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    diagnostics = {
        "image_width": int(width),
        "image_height": int(height),
        "foreground_pixels": int(foreground.sum()),
        "skeleton_pixels": int(skeleton_mask.sum()),
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "graph_edge_length": float(
            sum(float(edge.get("length_px") or 0.0) for edge in graph.get("edges", []))
        ),
        "graph_components": len(graph.get("components", [])),
        "graph_loops": len(graph.get("loops", [])),
        "endpoint_nodes": sum(
            str(node.get("type")) == "endpoint" for node in graph.get("nodes", [])
        ),
        "junction_nodes": sum(
            str(node.get("type")) in _JUNCTION_TYPES for node in graph.get("nodes", [])
        ),
        "local_node_sum": float(node_stats["node_count"].sum()),
        "local_endpoint_sum": float(node_stats["endpoint_count"].sum()),
        "local_junction_sum": float(node_stats["junction_count"].sum()),
        "local_edge_length_sum": float(edge_length.sum()),
    }
    return features.astype(np.float32), diagnostics


def smooth_feature_sequence(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [T, F], got {values.shape}")
    if values.shape[0] <= 1:
        return values.copy()

    output = np.zeros_like(values)
    for index in range(values.shape[0]):
        weighted = 0.5 * values[index]
        weight = 0.5
        if index > 0:
            weighted += 0.25 * values[index - 1]
            weight += 0.25
        if index + 1 < values.shape[0]:
            weighted += 0.25 * values[index + 1]
            weight += 0.25
        output[index] = weighted / weight
    return output


def resample_feature_sequence(
    features: np.ndarray,
    target_steps: int,
    *,
    source_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a donor sequence along normalized x for matched-shuffle inference."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [T, F], got {values.shape}")
    if target_steps < 1:
        raise ValueError("target_steps must be positive")
    source_steps = values.shape[0]
    if source_steps < 1:
        raise ValueError("Cannot resample an empty feature sequence")
    mask = (
        np.ones(source_steps, dtype=bool)
        if source_mask is None
        else np.asarray(source_mask, dtype=bool)
    )
    if mask.shape != (source_steps,):
        raise ValueError("source_mask does not match feature sequence")
    if source_steps == target_steps:
        return values.copy(), mask.copy()

    source_x = (np.arange(source_steps, dtype=np.float32) + 0.5) / source_steps
    target_x = (np.arange(target_steps, dtype=np.float32) + 0.5) / target_steps
    output = np.empty((target_steps, values.shape[1]), dtype=np.float32)
    for feature_index in range(values.shape[1]):
        output[:, feature_index] = np.interp(
            target_x,
            source_x,
            values[:, feature_index],
            left=float(values[0, feature_index]),
            right=float(values[-1, feature_index]),
        )
    resampled_mask = np.interp(
        target_x,
        source_x,
        mask.astype(np.float32),
        left=float(mask[0]),
        right=float(mask[-1]),
    ) >= 0.5
    return output, resampled_mask


def build_feature_record(
    *,
    sample_id: str,
    image_path: str | Path,
    dataset: str,
    graph_path: str | Path | None = None,
    width_downsample: int = WIDTH_DOWNSAMPLE,
    smooth: bool = True,
    feature_version: str = FEATURE_VERSION,
    binarization: str | None = None,
    sauvola_window: int = 25,
    min_object_size: int = 4,
) -> dict[str, Any]:
    resolved_image = Path(image_path).resolve()
    foreground, skeleton_mask, graph, method = load_or_extract_graph(
        resolved_image,
        dataset,
        graph_path=graph_path,
        binarization=binarization,
        sauvola_window=sauvola_window,
        min_object_size=min_object_size,
    )
    width = int(foreground.shape[1])
    time_steps = compute_output_steps(width, width_downsample)
    raw_features, diagnostics = aggregate_bin_features(
        foreground,
        skeleton_mask,
        graph,
        time_steps,
    )
    features = smooth_feature_sequence(raw_features) if smooth else raw_features.copy()
    quality = features[:, QUALITY_FEATURE_INDICES].astype(np.float32)
    valid_mask = np.ones(time_steps, dtype=bool)

    if not np.isfinite(features).all() or not np.isfinite(quality).all():
        raise ValueError(f"Non-finite features for sample {sample_id}")
    return {
        "features": features,
        "raw_features": raw_features,
        "quality": quality,
        "valid_mask": valid_mask,
        "time_steps": time_steps,
        "original_width": width,
        "feature_names": np.asarray(FEATURE_NAMES),
        "quality_feature_names": np.asarray(QUALITY_FEATURE_NAMES),
        "sample_id": sample_id,
        "graph_version": str(graph.get("schema_version", GRAPH_VERSION)),
        "feature_version": feature_version,
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "source_image_sha1": file_sha1(resolved_image),
        "binarization": method,
        "diagnostics": diagnostics,
        "foreground": foreground,
        "skeleton": skeleton_mask,
        "graph": graph,
    }


def save_feature_record(record: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.asarray(record["features"], dtype=np.float32),
        raw_features=np.asarray(record["raw_features"], dtype=np.float32),
        quality=np.asarray(record["quality"], dtype=np.float32),
        valid_mask=np.asarray(record["valid_mask"], dtype=bool),
        time_steps=np.int32(record["time_steps"]),
        original_width=np.int32(record["original_width"]),
        feature_names=np.asarray(record["feature_names"]),
        quality_feature_names=np.asarray(record["quality_feature_names"]),
        sample_id=np.asarray(str(record["sample_id"])),
        graph_version=np.asarray(str(record["graph_version"])),
        feature_version=np.asarray(str(record["feature_version"])),
        feature_builder_version=np.asarray(str(record["feature_builder_version"])),
        source_image_sha1=np.asarray(str(record["source_image_sha1"])),
        binarization=np.asarray(str(record["binarization"])),
        diagnostics_json=np.asarray(
            json.dumps(record.get("diagnostics", {}), ensure_ascii=False)
        ),
    )


def load_feature_record(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        record = {key: archive[key] for key in archive.files}
    for key in (
        "time_steps",
        "original_width",
        "sample_id",
        "graph_version",
        "feature_version",
        "feature_builder_version",
        "source_image_sha1",
        "binarization",
        "diagnostics_json",
    ):
        if key in record and np.asarray(record[key]).ndim == 0:
            record[key] = np.asarray(record[key]).item()
    if "diagnostics_json" in record:
        record["diagnostics"] = json.loads(str(record.pop("diagnostics_json")))
    return record


@dataclass(frozen=True)
class XAlignedFeatureNormalizer:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    clip_value: float
    train_manifest_sha256: str
    graph_version: str
    feature_builder_version: str
    created_at: str
    missing_policy: str = "raw_zero"
    default_policy: str = "empty_real_bin_zero_padding_mask_false"

    def __post_init__(self) -> None:
        dimension = len(self.feature_names)
        if dimension != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {dimension}")
        if len(self.mean) != dimension or len(self.std) != dimension:
            raise ValueError("Normalizer vector dimensions do not match feature_names")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("Normalizer feature order differs from FEATURE_NAMES")

    @classmethod
    def fit(
        cls,
        manifest_path: str | Path,
        *,
        feature_field: str = "xaligned_graph_npz",
        clip_value: float = 5.0,
    ) -> XAlignedFeatureNormalizer:
        manifest = Path(manifest_path)
        count = 0
        total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        total_sq = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        graph_versions: set[str] = set()
        builder_versions: set[str] = set()

        for row in read_jsonl(manifest):
            value = row.get(feature_field)
            if not value:
                raise KeyError(f"{feature_field} missing for {row.get('sample_id')}")
            feature_path = resolve_path(str(value), manifest)
            record = load_feature_record(feature_path)
            names = tuple(str(value) for value in record["feature_names"].tolist())
            if names != FEATURE_NAMES:
                raise ValueError(f"Feature names mismatch in {feature_path}")
            values = np.asarray(record["features"], dtype=np.float64)
            mask = np.asarray(record["valid_mask"], dtype=bool)
            values = values[mask]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {feature_path}")
            count += int(values.shape[0])
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            graph_versions.add(str(record["graph_version"]))
            builder_versions.add(str(record["feature_builder_version"]))

        if count == 0:
            raise ValueError("Cannot fit normalizer on an empty manifest")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0
        return cls(
            feature_names=FEATURE_NAMES,
            mean=tuple(float(value) for value in mean),
            std=tuple(float(value) for value in std),
            clip_value=float(clip_value),
            train_manifest_sha256=file_sha256(manifest),
            graph_version=",".join(sorted(graph_versions)),
            feature_builder_version=",".join(sorted(builder_versions)),
            created_at=datetime.now(UTC).isoformat(),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> XAlignedFeatureNormalizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(data["feature_names"]),
            mean=tuple(float(value) for value in data["mean"]),
            std=tuple(float(value) for value in data["std"]),
            clip_value=float(data["clip_value"]),
            train_manifest_sha256=str(data["train_manifest_sha256"]),
            graph_version=str(data["graph_version"]),
            feature_builder_version=str(data["feature_builder_version"]),
            created_at=str(data["created_at"]),
            missing_policy=str(data.get("missing_policy", "raw_zero")),
            default_policy=str(
                data.get(
                    "default_policy",
                    "empty_real_bin_zero_padding_mask_false",
                )
            ),
        )

    def to_path(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def transform(
        self,
        features: np.ndarray,
        *,
        topology_enabled: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Expected [T, {len(FEATURE_NAMES)}], got {values.shape}")
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        normalized = np.clip((values - mean) / np.maximum(std, 1e-6), -self.clip_value, self.clip_value)
        if not topology_enabled:
            normalized[:, TOPOLOGY_START_INDEX:] = 0.0
        quality = normalized[:, QUALITY_FEATURE_INDICES].copy()
        return normalized.astype(np.float32), quality.astype(np.float32)


def verify_normalizer_for_manifest(
    normalizer: XAlignedFeatureNormalizer,
    train_manifest: str | Path,
) -> None:
    actual = file_sha256(train_manifest)
    if actual != normalizer.train_manifest_sha256:
        raise ValueError(
            "Normalizer was not fitted on this train manifest: "
            f"expected {normalizer.train_manifest_sha256}, got {actual}"
        )


def feature_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(record["sample_id"]),
        "time_steps": int(record["time_steps"]),
        "original_width": int(record["original_width"]),
        "feature_dim": int(np.asarray(record["features"]).shape[1]),
        "feature_version": str(record["feature_version"]),
        "graph_version": str(record["graph_version"]),
        "source_image_sha1": str(record["source_image_sha1"]),
        "binarization": str(record["binarization"]),
        "diagnostics": record.get("diagnostics", {}),
    }


def locate_graph_path(
    row: dict[str, Any],
    *,
    graph_field: str | None,
    graph_root: str | Path | None,
    manifest_path: str | Path,
) -> Path | None:
    if graph_field and row.get(graph_field):
        return resolve_path(str(row[graph_field]), manifest_path)
    if graph_root is None:
        return None

    root = Path(graph_root)
    sample_id = str(row["sample_id"])
    candidates = [
        root / f"{sample_id}.json",
        root / f"{sample_id}.graph.json",
        root / sample_id / "graph.json",
    ]
    sample_dir = root / sample_id
    if sample_dir.exists():
        candidates.extend(sorted(sample_dir.glob("graph_*.json")))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def image_width_from_row(row: dict[str, Any], image_path: str | Path) -> int:
    with Image.open(image_path) as image:
        return int(image.width)


def ensure_feature_names(names: Sequence[str]) -> None:
    if tuple(str(name) for name in names) != FEATURE_NAMES:
        raise ValueError(f"Unexpected feature names: {names}")
