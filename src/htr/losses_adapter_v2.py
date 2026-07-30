from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"Value/mask shape mismatch: {values.shape} vs {mask.shape}")
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def baseline_preservation_kl(
    base_logits: torch.Tensor,
    final_logits: torch.Tensor,
    visual_uncertainty: torch.Tensor,
    time_mask: torch.Tensor,
    *,
    temperature: float = 1.5,
) -> torch.Tensor:
    if base_logits.shape != final_logits.shape:
        raise ValueError("Base and final logits must have identical shapes")
    if visual_uncertainty.shape != (*base_logits.shape[:2], 1):
        raise ValueError("visual_uncertainty must have shape [B,T,1]")
    if time_mask.shape != base_logits.shape[:2]:
        raise ValueError("time_mask does not match logits")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    base_distribution = (base_logits.detach() / temperature).softmax(dim=-1)
    final_log_distribution = (final_logits / temperature).log_softmax(dim=-1)
    kl_frame = F.kl_div(
        final_log_distribution,
        base_distribution,
        reduction="none",
    ).sum(dim=-1) * (temperature * temperature)
    confidence = 1.0 - visual_uncertainty[..., 0].detach()
    return masked_mean(kl_frame * confidence, time_mask)


def auxiliary_ctc_weight(epoch: int) -> float:
    if epoch < 1:
        raise ValueError("epoch is one-based")
    if epoch <= 3:
        return 0.15
    if epoch <= 6:
        return 0.05
    return 0.0

