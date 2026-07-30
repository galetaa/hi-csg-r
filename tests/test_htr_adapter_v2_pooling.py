from __future__ import annotations

import torch
from src.htr.masked_pooling import (
    MaskedMultiscaleGraphPooling,
    masked_average_pool1d,
)


def test_masked_pooling_excludes_padding_values() -> None:
    features = torch.tensor([[[1.0], [3.0], [1000.0], [1000.0]]])
    mask = torch.tensor([[True, True, False, False]])
    pooled = masked_average_pool1d(features, mask, 5)
    assert torch.allclose(pooled[0, :2, 0], torch.tensor([2.0, 2.0]))


def test_multiscale_output_is_zero_at_empty_target_bin() -> None:
    features = torch.ones(1, 5, 20)
    time = torch.ones(1, 5, dtype=torch.bool)
    nonempty = torch.tensor([[True, True, False, True, True]])
    output = MaskedMultiscaleGraphPooling()(features, time, nonempty)
    assert output.shape == (1, 5, 60)
    assert torch.count_nonzero(output[0, 2]).item() == 0

