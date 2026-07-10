import numpy as np
from src.graph.pixel_graph import build_pixel_graph
from src.graph.skeletonize import skeletonize_mask


def test_empty_mask_produces_empty_skeleton_and_graph_warning() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    skeleton = skeletonize_mask(mask)
    graph = build_pixel_graph(skeleton.skeleton, binary_mask=mask)
    assert skeleton.skeleton_pixels == 0
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.warnings == ["skeleton_empty"]


def test_straight_stroke_has_two_endpoints_and_one_component() -> None:
    skeleton = np.zeros((9, 12), dtype=bool)
    skeleton[4, 2:10] = True
    graph = build_pixel_graph(skeleton, binary_mask=skeleton)
    assert graph.features["component_count"] == 1
    assert graph.features["endpoint_count"] == 2
    assert graph.features["junction_count"] == 0
    assert graph.features["edge_count"] >= 1
