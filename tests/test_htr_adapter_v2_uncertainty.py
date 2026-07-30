from __future__ import annotations

import torch
from src.htr.uncertainty import VisualUncertaintyEstimator


def test_uncertainty_is_bounded_and_padding_zero() -> None:
    logits = torch.randn(2, 4, 10)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    value = VisualUncertaintyEstimator()(logits, mask)
    assert torch.all((value >= 0) & (value <= 1))
    assert torch.count_nonzero(value[1, 2:]).item() == 0


def test_uniform_logits_are_more_uncertain_than_peaked_logits() -> None:
    logits = torch.zeros(1, 2, 5)
    logits[0, 1, 0] = 20
    mask = torch.ones(1, 2, dtype=torch.bool)
    uncertainty = VisualUncertaintyEstimator()(logits, mask)
    assert uncertainty[0, 0, 0] > uncertainty[0, 1, 0]

