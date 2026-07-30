from __future__ import annotations

import numpy as np
import torch
from src.htr.dataset_adapter_v2 import collate_late_correction_batch
from src.htr.xaligned_hi_csg_r import FEATURE_NAMES, XAlignedFeatureNormalizer


def test_raw_zero_standardizes_to_nonzero_regression() -> None:
    normalizer = XAlignedFeatureNormalizer(
        feature_names=FEATURE_NAMES,
        mean=tuple([1.0] * 20),
        std=tuple([2.0] * 20),
        clip_value=5.0,
        train_manifest_sha256="x",
        graph_version="g",
        feature_builder_version="f",
        created_at="now",
    )
    normalized, _ = normalizer.transform(np.zeros((1, 20), dtype=np.float32))
    assert np.count_nonzero(normalized) == 20
    masked = normalized * np.zeros((1, 1), dtype=np.float32)
    assert np.count_nonzero(masked) == 0


def test_collate_distinguishes_empty_real_bin_and_padding() -> None:
    def item(sample: str, steps: int) -> dict:
        nonempty = torch.tensor([True] + [False] * (steps - 1))
        features = torch.zeros(steps, 20)
        features[0] = 1.0
        return {
            "image": torch.zeros(1, 16, steps * 4),
            "width": steps * 4,
            "output_steps": steps,
            "target": torch.tensor([1]),
            "text": "a",
            "sample_id": sample,
            "graph_sample_id": sample,
            "dataset": "hkr_words",
            "core_domain": "hkr",
            "level": "word",
            "category": None,
            "raw_graph_features": features.clone(),
            "normalized_graph_features": features,
            "structural_risk_raw": torch.zeros(steps, 3),
            "time_mask": torch.ones(steps, dtype=torch.bool),
            "nonempty_graph_mask": nonempty,
        }

    batch = collate_late_correction_batch([item("a", 4), item("b", 2)])
    assert batch["time_mask"][1, 1]
    assert not batch["nonempty_graph_mask"][1, 1]
    assert not batch["padding_mask"][1, 1]
    assert batch["padding_mask"][1, 2]
    assert torch.count_nonzero(
        batch["normalized_graph_features"][
            batch["time_mask"] & ~batch["nonempty_graph_mask"]
        ]
    ).item() == 0

