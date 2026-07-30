from __future__ import annotations

import torch
from src.htr.model import CRNNCTC
from src.htr.model_hi_csg_r_late_correction_v2 import (
    HI_CSG_R_LateCorrectionCRNNCTC,
)


def small_model(variant: str = "v2_1") -> HI_CSG_R_LateCorrectionCRNNCTC:
    baseline = CRNNCTC(
        num_classes=12,
        hidden_size=8,
        lstm_layers=1,
        dropout=0.0,
        blank_index=0,
        height_bins=2,
        feature_size=16,
    )
    return HI_CSG_R_LateCorrectionCRNNCTC(
        baseline,
        variant=variant,
        alpha_max=0.25,
        risk_q05=[0.0, 0.0, 0.0],
        risk_q95=[10.0, 1.0, 1.0],
        graph_dropout=0.0,
    )


def small_batch() -> dict[str, torch.Tensor]:
    batch_size = 2
    time_steps = 8
    time_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    nonempty = torch.tensor(
        [
            [1, 0, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    features = torch.randn(batch_size, time_steps, 20)
    features = features * nonempty.unsqueeze(-1)
    return {
        "images": torch.randn(batch_size, 1, 16, 32),
        "widths": torch.tensor([32, 24]),
        "features": features,
        "risk": torch.rand(batch_size, time_steps, 3),
        "time_mask": time_mask,
        "nonempty": nonempty,
    }

