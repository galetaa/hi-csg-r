from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Literal


SCHEMA_VERSION = "hi_csg_r_v1"


NodeType = Literal[
    "endpoint",
    "junction",
    "junction_cluster",
    "loop_virtual_node",
    "isolated_component",
    "uncertain_node",
]

EdgeType = Literal[
    "stroke_segment",
    "short_branch",
    "loop_segment",
    "isolated_loop",
    "uncertain_edge",
]


@dataclass
class GraphNode:
    id: str
    type: str
    x: float
    y: float
    x_norm: float
    y_norm: float
    degree: int
    component_id: str | None
    confidence: float = 1.0
    source: str = "unknown"
    flags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    id: str
    from_node: str | None
    to_node: str | None
    type: str
    component_id: str | None
    points: list[list[int]]
    length_px: float
    num_points: int
    width_mean: float | None = None
    width_std: float | None = None
    curvature_mean: float | None = None
    simplified_points: list[list[int]] = field(default_factory=list)
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["from"] = d.pop("from_node")
        d["to"] = d.pop("to_node")
        return d


@dataclass
class GraphComponent:
    id: str
    node_ids: list[str]
    edge_ids: list[str]
    bbox: list[int]
    area_px: int | None = None
    skeleton_length_px: int | None = None
    foreground_pixels: int | None = None
    flags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphLoop:
    id: str
    component_id: str | None
    edge_ids: list[str]
    bbox: list[int] | None = None
    area_estimate: float | None = None
    confidence: float = 0.0
    flags: list[str] = field(default_factory=lambda: ["candidate"])
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HICSGRGraph:
    sample_id: str
    dataset: str
    level: str
    source_image_path: str | None
    feature_image_path: str
    processing: dict[str, Any]
    image: dict[str, Any]
    binary: dict[str, Any]

    schema_version: str = SCHEMA_VERSION
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    components: list[dict[str, Any]] = field(default_factory=list)
    loops: list[dict[str, Any]] = field(default_factory=list)
    graph_features: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(
        default_factory=lambda: {
            "graph_confidence": None,
            "graph_quality_score": None,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)