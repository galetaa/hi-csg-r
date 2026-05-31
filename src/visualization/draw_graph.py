from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def draw_graph_overlay(
    *,
    feature_image_path: str | Path,
    graph: dict[str, Any],
    out_path: str | Path,
    max_edge_points: int = 2000,
    edge_width: int = 1,
    endpoint_radius: int = 2,
    junction_radius: int = 2,
    other_node_radius: int = 2,
    draw_nodes: bool = True,
    draw_edges: bool = True,
) -> None:
    img = Image.open(feature_image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    edges = graph.get("edges", [])
    nodes = graph.get("nodes", [])

    if draw_edges:
        for edge in edges:
            points = edge.get("simplified_points") or edge.get("points") or []

            if len(points) > max_edge_points:
                step = max(1, len(points) // max_edge_points)
                points = points[::step]

            if len(points) < 2:
                continue

            edge_type = edge.get("type")

            if edge_type == "short_branch":
                color = (255, 140, 0, 180)
            elif edge_type == "isolated_loop":
                color = (0, 120, 255, 180)
            else:
                color = (255, 220, 0, 150)

            xy = [(int(x), int(y)) for x, y in points]
            draw.line(xy, fill=color, width=edge_width)

    if draw_nodes:
        for node in nodes:
            x = int(round(node["x"]))
            y = int(round(node["y"]))
            node_type = node.get("type")

            if node_type == "endpoint":
                color = (0, 220, 0, 210)
                r = endpoint_radius
            elif node_type == "junction_cluster":
                color = (255, 0, 0, 210)
                r = junction_radius
            else:
                color = (0, 120, 255, 210)
                r = other_node_radius

            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)