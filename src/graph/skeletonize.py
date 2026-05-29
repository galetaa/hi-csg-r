from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize as sk_skeletonize


@dataclass
class SkeletonResult:
    skeleton: np.ndarray
    skeleton_pixels: int
    foreground_pixels: int


def skeletonize_mask(mask: np.ndarray) -> SkeletonResult:
    """
    mask: bool array, True = foreground/stroke.
    skeleton: bool array, True = skeleton pixel.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        skeleton = np.zeros_like(mask_bool, dtype=bool)
    else:
        skeleton = sk_skeletonize(mask_bool)

    return SkeletonResult(
        skeleton=skeleton.astype(bool),
        skeleton_pixels=int(skeleton.sum()),
        foreground_pixels=int(mask_bool.sum()),
    )


def skeleton_to_pil(skeleton: np.ndarray) -> Image.Image:
    out = np.where(skeleton, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")