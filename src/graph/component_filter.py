from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class SkeletonComponentFilterResult:
    skeleton: np.ndarray
    min_component_pixels: int
    kept_components: int
    removed_components: int
    kept_pixels: int
    removed_pixels: int
    removed_pixel_ratio: float


def filter_skeleton_components(
    skeleton: np.ndarray,
    min_component_pixels: int = 8,
) -> SkeletonComponentFilterResult:
    """
    Diagnostic skeleton component filtering.

    Input:
      skeleton=True means skeleton pixel.

    Removes connected skeleton components smaller than min_component_pixels.

    Important:
      This is diagnostic, not default.
      It can remove punctuation, dots, diacritics, and short handwriting fragments.
    """
    if skeleton.ndim != 2:
        raise ValueError(f"Expected 2D skeleton, got shape={skeleton.shape}")

    skeleton = skeleton.astype(bool)

    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n = ndi.label(skeleton, structure=structure)

    if n == 0:
        return SkeletonComponentFilterResult(
            skeleton=np.zeros_like(skeleton, dtype=bool),
            min_component_pixels=min_component_pixels,
            kept_components=0,
            removed_components=0,
            kept_pixels=0,
            removed_pixels=0,
            removed_pixel_ratio=0.0,
        )

    counts = np.bincount(labels.ravel())
    keep_labels = {
        label
        for label in range(1, n + 1)
        if counts[label] >= min_component_pixels
    }

    filtered = np.isin(labels, list(keep_labels))

    original_pixels = int(skeleton.sum())
    kept_pixels = int(filtered.sum())
    removed_pixels = original_pixels - kept_pixels

    return SkeletonComponentFilterResult(
        skeleton=filtered.astype(bool),
        min_component_pixels=int(min_component_pixels),
        kept_components=len(keep_labels),
        removed_components=n - len(keep_labels),
        kept_pixels=kept_pixels,
        removed_pixels=removed_pixels,
        removed_pixel_ratio=removed_pixels / max(original_pixels, 1),
    )