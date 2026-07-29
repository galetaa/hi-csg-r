from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class GridSuppressionResult:
    cleaned_mask: np.ndarray
    grid_mask: np.ndarray
    removed_pixels: int
    removed_ratio: float
    horizontal_length: int
    vertical_length: int
    line_width: int


def suppress_long_grid_lines(
    mask: np.ndarray,
    horizontal_length: int | None = 35,
    vertical_length: int | None = 35,
    line_width: int = 1,
) -> GridSuppressionResult:
    """
    Experimental grid/form-line suppression.

    Input:
      mask=True means foreground.

    Method:
      morphological opening with horizontal/vertical kernels,
      then subtract detected long straight structures.

    Important:
      This is diagnostic, not final.
      It reduces grid/form lines but does not guarantee full background removal.
      It can remove parts of handwriting if parameters are too aggressive.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")

    mask = mask.astype(bool)
    h, w = mask.shape

    if horizontal_length is None:
        horizontal_length = max(35, min(120, w // 12))

    if vertical_length is None:
        vertical_length = max(35, min(120, h // 12))

    horizontal_length = max(3, int(horizontal_length))
    vertical_length = max(3, int(vertical_length))
    line_width = max(1, int(line_width))

    horizontal_structure = np.ones((line_width, horizontal_length), dtype=bool)
    vertical_structure = np.ones((vertical_length, line_width), dtype=bool)

    horizontal = ndi.binary_opening(mask, structure=horizontal_structure)
    vertical = ndi.binary_opening(mask, structure=vertical_structure)

    grid_mask = horizontal | vertical
    cleaned = mask & ~grid_mask

    removed_pixels = int((mask & grid_mask).sum())
    total_fg = int(mask.sum())
    removed_ratio = removed_pixels / max(total_fg, 1)

    return GridSuppressionResult(
        cleaned_mask=cleaned,
        grid_mask=grid_mask,
        removed_pixels=removed_pixels,
        removed_ratio=float(removed_ratio),
        horizontal_length=horizontal_length,
        vertical_length=vertical_length,
        line_width=line_width,
    )