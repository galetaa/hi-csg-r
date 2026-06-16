from __future__ import annotations

import os

import numpy as np
from skimage.morphology import remove_small_objects

from tools import extract_htr_graph_features as base


ORIGINAL_BINARIZE = base.binarize

SCHOOL_METHOD = os.environ.get("SCHOOL_FOREGROUND_V3_METHOD", "global_dark_145")
SCHOOL_MAX_FG_FRACTION = float(os.environ.get("SCHOOL_FOREGROUND_V3_MAX_FG", "0.35"))
SCHOOL_MIN_OBJECT_SIZE = int(os.environ.get("SCHOOL_FOREGROUND_V3_MIN_OBJECT", "4"))


def _cleanup(fg: np.ndarray) -> np.ndarray:
    fg = fg.astype(bool)
    if SCHOOL_MIN_OBJECT_SIZE > 0 and fg.any():
        fg = remove_small_objects(fg, min_size=SCHOOL_MIN_OBJECT_SIZE)
    return fg.astype(bool)


def _global_dark(arr: np.ndarray, threshold: int) -> np.ndarray:
    return _cleanup(arr < int(threshold))


def _school_dark_auto(arr: np.ndarray) -> np.ndarray:
    """
    Deterministic school-notebooks foreground rule.

    Start with the empirically best threshold 145.
    If it still marks too much crop/background as foreground,
    fall back to stricter threshold 120.
    """
    fg145 = _global_dark(arr, 145)

    if float(fg145.mean()) <= SCHOOL_MAX_FG_FRACTION:
        return fg145

    fg120 = _global_dark(arr, 120)
    return fg120


def binarize(arr: np.ndarray, method: str, sauvola_window: int) -> np.ndarray:
    if method == "global_dark_120":
        return _global_dark(arr, 120)

    if method == "global_dark_145":
        return _global_dark(arr, 145)

    if method == "school_dark_auto":
        return _school_dark_auto(arr)

    return ORIGINAL_BINARIZE(arr, method=method, sauvola_window=sauvola_window)


def main() -> None:
    base.DATASET_BINARIZATION = dict(base.DATASET_BINARIZATION)
    base.DATASET_BINARIZATION["school_notebooks_clean"] = SCHOOL_METHOD
    base.binarize = binarize

    print(f"[school foreground v3] method={SCHOOL_METHOD}")
    print(f"[school foreground v3] max_fg={SCHOOL_MAX_FG_FRACTION}")
    print(f"[school foreground v3] min_object_size={SCHOOL_MIN_OBJECT_SIZE}")

    base.main()


if __name__ == "__main__":
    main()