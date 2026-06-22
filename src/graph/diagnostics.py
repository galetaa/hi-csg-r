from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


def connected_components_count(mask: np.ndarray) -> int:
    structure = np.ones((3, 3), dtype=np.uint8)
    _, n = ndi.label(mask.astype(bool), structure=structure)
    return int(n)


def make_binary_skeleton_diagnostics(
    *,
    sample_id: str,
    dataset: str,
    level: str,
    image_shape: tuple[int, int],
    foreground_ratio: float,
    skeleton_pixels: int,
    foreground_pixels: int,
    method: str,
    threshold_info: dict[str, Any],
    scale: float = 1.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    height, width = image_shape

    warnings: list[str] = []

    if foreground_pixels == 0:
        warnings.append("empty_foreground")

    if foreground_ratio < 0.002:
        warnings.append("too_low_foreground_ratio")

    if foreground_ratio > 0.45:
        warnings.append("too_high_foreground_ratio")

    if skeleton_pixels == 0:
        warnings.append("skeleton_empty")

    if scale < 0.999:
        warnings.append("large_page_scaled")

    if dataset == "hwr200":
        warnings.append("hwr200_page")

    if dataset == "hkr_forms":
        warnings.append("hkr_forms_page")
        warnings.append("hkr_possible_form_grid")

    out = {
        "sample_id": sample_id,
        "dataset": dataset,
        "level": level,
        "image_width": int(width),
        "image_height": int(height),
        "binarization": method,
        "threshold_info": threshold_info,
        "foreground_ratio": float(foreground_ratio),
        "foreground_pixels": int(foreground_pixels),
        "skeleton_pixels": int(skeleton_pixels),
        "scale": float(scale),
        "warnings": warnings,
    }

    if extra:
        out.update(extra)

    return out