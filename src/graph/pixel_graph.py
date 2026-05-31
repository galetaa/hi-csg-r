from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from src.graph.graph_schema import GraphNode, GraphEdge, GraphComponent, GraphLoop


NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


Pixel = tuple[int, int]  # y, x


@dataclass
class PixelGraphResult:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    components: list[dict[str, Any]]
    loops: list[dict[str, Any]]
    features: dict[str, Any]
    warnings: list[str]


def edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
    return (a, b) if a <= b else (b, a)


def neighbors8(pixel: Pixel, skeleton: np.ndarray) -> list[Pixel]:
    y, x = pixel
    h, w = skeleton.shape
    out = []

    for dy, dx in NEIGHBORS_8:
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and skeleton[yy, xx]:
            out.append((yy, xx))

    return out


def degree_map(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    count = ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
    return count - skeleton.astype(np.uint8)


def component_id_from_label(label: int) -> str:
    return f"cc_{label:06d}"


def node_id(i: int) -> str:
    return f"n_{i:06d}"


def edge_id(i: int) -> str:
    return f"e_{i:06d}"


def loop_id(i: int) -> str:
    return f"loop_{i:06d}"


def bbox_from_pixels(pixels: list[Pixel]) -> list[int]:
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def polyline_length(points_yx: list[Pixel]) -> float:
    if len(points_yx) < 2:
        return 0.0

    total = 0.0
    for (y1, x1), (y2, x2) in zip(points_yx, points_yx[1:]):
        total += hypot(float(y2 - y1), float(x2 - x1))

    return total


def points_to_xy(points_yx: list[Pixel]) -> list[list[int]]:
    return [[int(x), int(y)] for y, x in points_yx]


def rdp(points: list[list[int]], epsilon: float = 1.5) -> list[list[int]]:
    if len(points) < 3:
        return points

    start = np.array(points[0], dtype=float)
    end = np.array(points[-1], dtype=float)
    line = end - start
    line_norm = np.linalg.norm(line)

    if line_norm == 0:
        distances = [np.linalg.norm(np.array(p, dtype=float) - start) for p in points]
    else:
        distances = [
            abs(np.cross(line, start - np.array(p, dtype=float))) / line_norm
            for p in points
        ]

    idx = int(np.argmax(distances))
    dmax = distances[idx]

    if dmax > epsilon:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return left[:-1] + right

    return [points[0], points[-1]]


def edge_width_stats(points_yx: list[Pixel], width_map: np.ndarray | None) -> tuple[float | None, float | None]:
    if width_map is None or not points_yx:
        return None, None

    vals = []
    h, w = width_map.shape

    for y, x in points_yx:
        if 0 <= y < h and 0 <= x < w:
            vals.append(float(2.0 * width_map[y, x]))

    if not vals:
        return None, None

    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


def build_components(
    skeleton: np.ndarray,
    binary_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[int, GraphComponent]]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n = ndi.label(skeleton.astype(bool), structure=structure)

    components: dict[int, GraphComponent] = {}

    for label in range(1, n + 1):
        ys, xs = np.where(labels == label)
        pixels = list(zip(ys.tolist(), xs.tolist()))

        if not pixels:
            continue

        bbox = bbox_from_pixels(pixels)
        foreground_pixels = None

        if binary_mask is not None:
            x0, y0, x1, y1 = bbox
            foreground_pixels = int(binary_mask[y0:y1 + 1, x0:x1 + 1].sum())

        comp = GraphComponent(
            id=component_id_from_label(label),
            node_ids=[],
            edge_ids=[],
            bbox=bbox,
            area_px=int((labels == label).sum()),
            skeleton_length_px=int((labels == label).sum()),
            foreground_pixels=foreground_pixels,
            flags=[],
        )
        components[label] = comp

    return labels, components


def build_nodes(
    skeleton: np.ndarray,
    degrees: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[GraphNode], dict[Pixel, str], dict[str, list[Pixel]], list[str]]:
    h, w = skeleton.shape
    warnings: list[str] = []

    nodes: list[GraphNode] = []
    pixel_to_node: dict[Pixel, str] = {}
    node_to_pixels: dict[str, list[Pixel]] = {}

    endpoint_pixels = list(zip(*np.where(skeleton & (degrees == 1))))
    junction_pixels_mask = skeleton & (degrees >= 3)

    # Endpoints.
    for p in endpoint_pixels:
        y, x = int(p[0]), int(p[1])
        nid = node_id(len(nodes) + 1)
        label = int(labels[y, x])
        comp_id = component_id_from_label(label) if label else None

        node = GraphNode(
            id=nid,
            type="endpoint",
            x=float(x),
            y=float(y),
            x_norm=float(x / max(w - 1, 1)),
            y_norm=float(y / max(h - 1, 1)),
            degree=int(degrees[y, x]),
            component_id=comp_id,
            confidence=1.0,
            source="skeleton_degree",
            flags=[],
        )

        nodes.append(node)
        pixel_to_node[(y, x)] = nid
        node_to_pixels[nid] = [(y, x)]

    # Junction clusters.
    structure = np.ones((3, 3), dtype=np.uint8)
    junction_labels, n_junction = ndi.label(junction_pixels_mask, structure=structure)

    for j_label in range(1, n_junction + 1):
        ys, xs = np.where(junction_labels == j_label)
        pixels = [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]

        if not pixels:
            continue

        cy = float(np.mean([p[0] for p in pixels]))
        cx = float(np.mean([p[1] for p in pixels]))

        rep_y, rep_x = pixels[0]
        label = int(labels[rep_y, rep_x])
        comp_id = component_id_from_label(label) if label else None

        nid = node_id(len(nodes) + 1)

        node = GraphNode(
            id=nid,
            type="junction_cluster",
            x=cx,
            y=cy,
            x_norm=float(cx / max(w - 1, 1)),
            y_norm=float(cy / max(h - 1, 1)),
            degree=int(max(degrees[y, x] for y, x in pixels)),
            component_id=comp_id,
            confidence=0.9,
            source="junction_cluster_degree_ge_3",
            flags=[],
            extra={
                "cluster_size": len(pixels),
                "raw_pixels_preview": points_to_xy(pixels[:20]),
            },
        )

        nodes.append(node)
        node_to_pixels[nid] = pixels

        for p in pixels:
            pixel_to_node[p] = nid

    if len(nodes) == 0 and int(skeleton.sum()) > 0:
        warnings.append("no_special_nodes_detected")

    return nodes, pixel_to_node, node_to_pixels, warnings


def trace_edges(
    skeleton: np.ndarray,
    labels: np.ndarray,
    pixel_to_node: dict[Pixel, str],
    node_to_pixels: dict[str, list[Pixel]],
    width_map: np.ndarray | None,
) -> tuple[list[GraphEdge], list[GraphLoop], list[str]]:
    warnings: list[str] = []
    visited: set[tuple[Pixel, Pixel]] = set()
    edges: list[GraphEdge] = []
    loops: list[GraphLoop] = []

    node_pixels = set(pixel_to_node.keys())
    max_steps = int(skeleton.sum()) + 100

    def same_node(a: Pixel, b: Pixel) -> bool:
        return (
            a in pixel_to_node
            and b in pixel_to_node
            and pixel_to_node[a] == pixel_to_node[b]
        )

    # Trace paths starting from every special node pixel.
    for start_node, start_pixels in node_to_pixels.items():
        for sp in start_pixels:
            for nb in neighbors8(sp, skeleton):
                if same_node(sp, nb):
                    continue

                ek = edge_key(sp, nb)
                if ek in visited:
                    continue

                path = [sp]
                prev = sp
                cur = nb
                end_node = None
                ambiguous_steps = 0

                for _ in range(max_steps):
                    visited.add(edge_key(prev, cur))
                    path.append(cur)

                    if cur in pixel_to_node and pixel_to_node[cur] != start_node:
                        end_node = pixel_to_node[cur]
                        break

                    candidates = [
                        nxt for nxt in neighbors8(cur, skeleton)
                        if nxt != prev
                        and edge_key(cur, nxt) not in visited
                        and not same_node(cur, nxt)
                    ]

                    if not candidates:
                        break

                    if len(candidates) > 1:
                        ambiguous_steps += 1

                    nxt = candidates[0]
                    prev, cur = cur, nxt
                else:
                    warnings.append("edge_trace_max_steps_reached")

                if len(path) < 2:
                    continue

                label = int(labels[path[min(1, len(path) - 1)][0], path[min(1, len(path) - 1)][1]])
                comp_id = component_id_from_label(label) if label else None
                length = polyline_length(path)
                points_xy = points_to_xy(path)
                width_mean, width_std = edge_width_stats(path, width_map)

                flags = []
                edge_type = "stroke_segment"

                if length < 3.0:
                    edge_type = "short_branch"
                    flags.append("short_branch")

                if ambiguous_steps > 0:
                    flags.append("ambiguous_trace_step")

                e = GraphEdge(
                    id=edge_id(len(edges) + 1),
                    from_node=start_node,
                    to_node=end_node,
                    type=edge_type,
                    component_id=comp_id,
                    points=points_xy,
                    length_px=float(length),
                    num_points=len(points_xy),
                    width_mean=width_mean,
                    width_std=width_std,
                    simplified_points=rdp(points_xy, epsilon=1.5),
                    confidence=0.8 if ambiguous_steps else 1.0,
                    flags=flags,
                    extra={
                        "ambiguous_steps": ambiguous_steps,
                    },
                )

                edges.append(e)

    # Handle components without any special node as isolated loops/components.
    structure = np.ones((3, 3), dtype=np.uint8)
    comp_labels, n_comp = ndi.label(skeleton.astype(bool), structure=structure)

    node_components = {
        int(labels[y, x])
        for y, x in pixel_to_node.keys()
        if int(labels[y, x]) > 0
    }

    for label in range(1, n_comp + 1):
        if label in node_components:
            continue

        ys, xs = np.where(comp_labels == label)
        pixels = [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]

        if not pixels:
            continue

        points_xy = points_to_xy(pixels)
        length = float(len(pixels))
        comp_id = component_id_from_label(label)

        e = GraphEdge(
            id=edge_id(len(edges) + 1),
            from_node=None,
            to_node=None,
            type="isolated_loop" if len(pixels) > 2 else "uncertain_edge",
            component_id=comp_id,
            points=points_xy,
            length_px=length,
            num_points=len(points_xy),
            width_mean=None,
            width_std=None,
            simplified_points=rdp(points_xy, epsilon=2.0),
            confidence=0.5,
            flags=["isolated_component_no_special_nodes"],
        )
        edges.append(e)

        if len(pixels) > 2:
            loops.append(
                GraphLoop(
                    id=loop_id(len(loops) + 1),
                    component_id=comp_id,
                    edge_ids=[e.id],
                    bbox=bbox_from_pixels(pixels),
                    area_estimate=float(len(pixels)),
                    confidence=0.3,
                    flags=["candidate", "isolated_loop_preliminary"],
                )
            )

    return edges, loops, warnings


def build_pixel_graph(
    skeleton: np.ndarray,
    binary_mask: np.ndarray | None = None,
    width_map: np.ndarray | None = None,
) -> PixelGraphResult:
    skeleton = skeleton.astype(bool)

    warnings: list[str] = []

    if skeleton.sum() == 0:
        warnings.append("skeleton_empty")
        return PixelGraphResult([], [], [], [], {}, warnings)

    degrees = degree_map(skeleton)
    labels, components_by_label = build_components(skeleton, binary_mask)

    nodes, pixel_to_node, node_to_pixels, node_warnings = build_nodes(
        skeleton=skeleton,
        degrees=degrees,
        labels=labels,
    )
    warnings.extend(node_warnings)

    edges, loops, edge_warnings = trace_edges(
        skeleton=skeleton,
        labels=labels,
        pixel_to_node=pixel_to_node,
        node_to_pixels=node_to_pixels,
        width_map=width_map,
    )
    warnings.extend(edge_warnings)

    # Attach node/edge ids to components.
    components = list(components_by_label.values())
    comp_by_id = {c.id: c for c in components}

    for node in nodes:
        if node.component_id in comp_by_id:
            comp_by_id[node.component_id].node_ids.append(node.id)

    for edge in edges:
        if edge.component_id in comp_by_id:
            comp_by_id[edge.component_id].edge_ids.append(edge.id)

    nodes_dict = [n.to_dict() for n in nodes]
    edges_dict = [e.to_dict() for e in edges]
    components_dict = [c.to_dict() for c in components]
    loops_dict = [l.to_dict() for l in loops]

    short_branches = sum(1 for e in edges if e.type == "short_branch")
    widths = [e.width_mean for e in edges if e.width_mean is not None]

    features = {
        "component_count": len(components),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "endpoint_count": sum(1 for n in nodes if n.type == "endpoint"),
        "junction_count": sum(1 for n in nodes if n.type == "junction_cluster"),
        "loop_candidate_count": len(loops),
        "short_branch_count": short_branches,
        "short_branch_ratio": short_branches / max(len(edges), 1),
        "skeleton_pixels": int(skeleton.sum()),
        "mean_width_proxy": float(np.mean(widths)) if widths else None,
    }

    if features["junction_count"] > 500:
        warnings.append("too_many_junctions")

    if features["component_count"] > 500:
        warnings.append("too_many_components")

    if features["short_branch_ratio"] > 0.35:
        warnings.append("too_many_short_branches")

    return PixelGraphResult(
        nodes=nodes_dict,
        edges=edges_dict,
        components=components_dict,
        loops=loops_dict,
        features=features,
        warnings=sorted(set(warnings)),
    )