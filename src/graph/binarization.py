from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image
from skimage import filters
from skimage.util import img_as_ubyte

from src.preprocessing.image_io import load_image_as_grayscale


BinarizationMethod = Literal["otsu", "sauvola", "adaptive_gaussian"]


@dataclass
class BinaryResult:
    mask: np.ndarray
    method: str
    foreground_ratio: float
    threshold_info: dict


def pil_to_uint8_array(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("L"))

    if arr.dtype != np.uint8:
        arr = img_as_ubyte(arr)

    return arr


def load_grayscale_array(path: str) -> np.ndarray:
    img = load_image_as_grayscale(path)
    return pil_to_uint8_array(img)


def maybe_invert_foreground(mask: np.ndarray) -> np.ndarray:
    """
    Внутренний convention:
      foreground = True
      background = False

    Для рукописей обычно foreground занимает меньшую площадь.
    Если foreground > 0.5, вероятно, threshold дал инверсию.
    """
    ratio = float(mask.mean())

    if ratio > 0.5:
        return ~mask

    return mask


def binarize_array(
    arr: np.ndarray,
    method: BinarizationMethod = "sauvola",
    sauvola_window_size: int = 25,
    sauvola_k: float = 0.2,
    adaptive_block_size: int = 35,
    adaptive_offset: float = 10.0,
) -> BinaryResult:
    """
    Возвращает binary mask, где True = stroke/foreground.

    arr: grayscale uint8, 0=black, 255=white.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale 2D array, got shape={arr.shape}")

    arr_float = arr.astype(np.float32)

    threshold_info = {}

    if method == "otsu":
        thr = filters.threshold_otsu(arr)
        # dark strokes: foreground is arr < threshold
        mask = arr < thr
        threshold_info = {"threshold": float(thr)}

    elif method == "sauvola":
        # window must be odd and not larger than image dimensions.
        h, w = arr.shape
        win = min(sauvola_window_size, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
        win = max(3, win)
        if win % 2 == 0:
            win -= 1

        thr = filters.threshold_sauvola(arr, window_size=win, k=sauvola_k)
        mask = arr_float < thr
        threshold_info = {
            "window_size": int(win),
            "k": float(sauvola_k),
        }

    elif method == "adaptive_gaussian":
        h, w = arr.shape
        block = min(adaptive_block_size, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
        block = max(3, block)
        if block % 2 == 0:
            block -= 1

        thr = filters.threshold_local(
            arr,
            block_size=block,
            method="gaussian",
            offset=adaptive_offset,
        )
        mask = arr_float < thr
        threshold_info = {
            "block_size": int(block),
            "offset": float(adaptive_offset),
        }

    else:
        raise ValueError(f"Unknown binarization method: {method}")

    mask = maybe_invert_foreground(mask.astype(bool))
    foreground_ratio = float(mask.mean())

    return BinaryResult(
        mask=mask,
        method=method,
        foreground_ratio=foreground_ratio,
        threshold_info=threshold_info,
    )


def binarize_image(
    image_path: str,
    method: BinarizationMethod = "sauvola",
) -> BinaryResult:
    arr = load_grayscale_array(image_path)
    return binarize_array(arr, method=method)


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """
    Для сохранения:
      foreground True → black
      background False → white
    """
    out = np.where(mask, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")