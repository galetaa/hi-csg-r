from __future__ import annotations

import torch
from src.htr.losses_adapter_v2 import (
    auxiliary_ctc_weight,
    baseline_preservation_kl,
)


def test_preservation_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 4, 10)
    uncertainty = torch.rand(2, 4, 1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss = baseline_preservation_kl(logits, logits, uncertainty, mask)
    assert abs(loss.item()) < 1e-6


def test_padded_logits_do_not_affect_preservation_loss() -> None:
    base = torch.randn(1, 3, 5)
    final = base.clone()
    final[:, 2] += 100
    uncertainty = torch.zeros(1, 3, 1)
    mask = torch.tensor([[True, True, False]])
    assert abs(
        baseline_preservation_kl(base, final, uncertainty, mask).item()
    ) < 1e-6


def test_auxiliary_schedule_is_zero_after_epoch_six() -> None:
    assert auxiliary_ctc_weight(1) == 0.15
    assert auxiliary_ctc_weight(4) == 0.05
    assert auxiliary_ctc_weight(7) == 0.0
    assert auxiliary_ctc_weight(100) == 0.0

