from __future__ import annotations

import numpy as np
import torch
from src.htr.model import CRNNCTC
from src.htr.xaligned_hi_csg_r import (
    FEATURE_NAMES,
    aggregate_bin_features,
    assign_nodes_to_bins,
    compute_output_steps,
    distribute_edges_to_bins,
    resample_feature_sequence,
    smooth_feature_sequence,
)


def test_bin_assignment_known_coordinates() -> None:
    graph = {
        "image": {"width": 100},
        "nodes": [
            {"x_norm": 0.0, "type": "endpoint"},
            {"x_norm": 0.249, "type": "junction_cluster"},
            {"x_norm": 0.50, "type": "endpoint"},
            {"x_norm": 1.0, "type": "endpoint"},
        ],
    }
    result = assign_nodes_to_bins(graph, 4)
    assert result["node_count"].tolist() == [2.0, 0.0, 1.0, 1.0]
    assert result["endpoint_count"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert result["junction_count"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_long_edge_is_distributed_across_bins() -> None:
    graph = {
        "image": {"width": 101},
        "edges": [
            {
                "id": "edge",
                "type": "stroke_segment",
                "component_id": "component",
                "points": [[0, 5], [25, 5], [50, 5], [75, 5], [100, 5]],
            }
        ],
    }
    result = distribute_edges_to_bins(graph, 4)
    assert np.count_nonzero(result["length"]) == 4
    assert np.allclose(result["length"], 25.0)
    assert np.all(result["orientation_length"][:, 0] > 0)


def test_output_length_and_exact_feature_dimension() -> None:
    assert compute_output_steps(1) == 1
    assert compute_output_steps(63) == 15
    assert compute_output_steps(64) == 16
    foreground = np.zeros((8, 64), dtype=bool)
    foreground[3:5, 4:60] = True
    skeleton = np.zeros_like(foreground)
    skeleton[4, 4:60] = True
    graph = {
        "image": {"width": 64, "height": 8},
        "nodes": [],
        "edges": [],
        "components": [],
        "loops": [],
    }
    features, _ = aggregate_bin_features(foreground, skeleton, graph, 16)
    assert features.shape == (16, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    model = CRNNCTC(
        num_classes=4,
        hidden_size=8,
        lstm_layers=1,
        dropout=0.0,
        feature_size=16,
        height_bins=2,
    )
    image = torch.zeros(1, 1, 16, 64)
    with torch.no_grad():
        assert model(image).shape[0] == compute_output_steps(64)
    assert model.output_lengths(torch.tensor([64])).item() == compute_output_steps(64)


def test_fixed_smoothing_and_resampling() -> None:
    values = np.asarray([[0.0], [4.0], [0.0]], dtype=np.float32)
    smoothed = smooth_feature_sequence(values)
    assert np.allclose(smoothed[:, 0], [4.0 / 3.0, 2.0, 4.0 / 3.0])
    resampled, mask = resample_feature_sequence(values, 5)
    assert resampled.shape == (5, 1)
    assert mask.all()
    assert np.isclose(resampled[2, 0], 4.0)
