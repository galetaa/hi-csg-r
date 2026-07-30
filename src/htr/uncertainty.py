from __future__ import annotations

import math

import torch
from torch import nn


class VisualUncertaintyEstimator(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        base_logits: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.ndim != 3:
            raise ValueError(f"Expected [B,T,C] logits, got {base_logits.shape}")
        if time_mask.shape != base_logits.shape[:2]:
            raise ValueError("time_mask does not match logits")
        class_count = base_logits.shape[-1]
        if class_count < 2:
            raise ValueError("At least two CTC classes are required")
        probabilities = base_logits.detach().softmax(dim=-1)
        entropy = -(
            probabilities * (probabilities + self.eps).log()
        ).sum(dim=-1) / math.log(class_count)
        top2 = probabilities.topk(2, dim=-1).values
        margin_uncertainty = 1.0 - (top2[..., 0] - top2[..., 1])
        uncertainty = (0.5 * entropy + 0.5 * margin_uncertainty).clamp(0.0, 1.0)
        return uncertainty.unsqueeze(-1) * time_mask.unsqueeze(-1).to(uncertainty.dtype)

