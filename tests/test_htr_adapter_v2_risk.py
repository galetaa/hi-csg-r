from __future__ import annotations

import torch
from src.htr.model_hi_csg_r_late_correction_v2 import StructuralRiskAttenuation


def test_risk_attenuation_is_bounded_and_monotonic() -> None:
    module = StructuralRiskAttenuation([0, 0, 0], [10, 1, 5])
    raw = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 1.0, 5.0]]])
    risk, reliability = module(raw)
    assert torch.all((risk >= 0) & (risk <= 1))
    assert torch.all((reliability > 0) & (reliability <= 1))
    assert reliability[0, 0, 0] > reliability[0, 1, 0]
    assert torch.allclose(reliability[0, 0, 0], torch.tensor(1.0))


def test_risk_formula_matches_frozen_weights() -> None:
    module = StructuralRiskAttenuation([0, 0, 0], [1, 1, 1])
    risk, _ = module(torch.tensor([[[1.0, 0.5, 0.0]]]))
    assert torch.allclose(risk, torch.tensor([[[0.5]]]))

