from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def masked_average_pool1d(
    features: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Centered temporal average that excludes invalid and padded timesteps."""
    if features.ndim != 3:
        raise ValueError(f"Expected [B,T,F] features, got {features.shape}")
    if mask.shape != features.shape[:2]:
        raise ValueError("Mask shape does not match temporal features")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    values = features.transpose(1, 2)
    weights = mask.to(features.dtype).unsqueeze(1)
    channels = features.shape[-1]
    value_kernel = torch.ones(
        channels,
        1,
        kernel_size,
        dtype=features.dtype,
        device=features.device,
    )
    mask_kernel = torch.ones(
        1,
        1,
        kernel_size,
        dtype=features.dtype,
        device=features.device,
    )
    padding = kernel_size // 2
    numerator = F.conv1d(
        values * weights,
        value_kernel,
        padding=padding,
        groups=channels,
    )
    denominator = F.conv1d(weights, mask_kernel, padding=padding).clamp_min(eps)
    return (numerator / denominator).transpose(1, 2)


class MaskedMultiscaleGraphPooling(nn.Module):
    def __init__(self, kernels: tuple[int, ...] = (1, 5, 9)) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("At least one pooling kernel is required")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("All pooling kernels must be positive and odd")
        self.kernels = tuple(int(kernel) for kernel in kernels)

    def forward(
        self,
        features: torch.Tensor,
        time_mask: torch.Tensor,
        nonempty_graph_mask: torch.Tensor,
    ) -> torch.Tensor:
        if time_mask.shape != features.shape[:2]:
            raise ValueError("time_mask does not match graph features")
        if nonempty_graph_mask.shape != features.shape[:2]:
            raise ValueError("nonempty_graph_mask does not match graph features")
        valid = time_mask & nonempty_graph_mask
        masked = features * valid.unsqueeze(-1).to(features.dtype)
        pooled = [
            masked_average_pool1d(masked, time_mask, kernel)
            for kernel in self.kernels
        ]
        output = torch.cat(pooled, dim=-1)
        # Context may summarize neighbors, but an empty target bin never receives
        # a graph representation or downstream correction.
        return output * valid.unsqueeze(-1).to(output.dtype)

