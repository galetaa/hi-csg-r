from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt

def cleanup(
    mask: np.ndarray,
    min_size: int = 4,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)

    if mask.any():
        mask = remove_small_objects(
            mask,
            min_size=min_size,
        )

    return mask.astype(bool)


def row_metadata(
    row: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in ["metadata", "source_metadata"]:
        value = row.get(key)

        if isinstance(value, dict):
            result.update(value)

    return result


def resolve_path(value: str) -> Path:
    path = Path(value)

    candidates = [
        path,
        Path.cwd() / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(value)


def polygon_support_mask(
    raw_row: dict[str, Any],
    raw_size: tuple[int, int],
) -> np.ndarray:
    raw_w, raw_h = raw_size

    polygon = raw_row.get("polygon")
    bbox = raw_row.get("bbox")

    metadata = row_metadata(raw_row)

    if not bbox:
        bbox = metadata.get("crop_bbox_xyxy")

    if (
        not isinstance(polygon, list)
        or len(polygon) < 3
        or not isinstance(bbox, list)
        or len(bbox) < 4
    ):
        return np.ones(
            (raw_h, raw_w),
            dtype=bool,
        )

    x0, y0, x1, y1 = [
        float(value)
        for value in bbox[:4]
    ]

    bbox_w = max(x1 - x0, 1.0)
    bbox_h = max(y1 - y0, 1.0)

    sx = raw_w / bbox_w
    sy = raw_h / bbox_h

    points = []

    for point in polygon:
        if (
            not isinstance(point, list)
            or len(point) < 2
        ):
            continue

        px = (
            float(point[0]) - x0
        ) * sx

        py = (
            float(point[1]) - y0
        ) * sy

        points.append((px, py))

    if len(points) < 3:
        return np.ones(
            (raw_h, raw_w),
            dtype=bool,
        )

    mask_image = Image.new(
        "L",
        (raw_w, raw_h),
        0,
    )

    draw = ImageDraw.Draw(mask_image)
    draw.polygon(points, fill=255)

    mask = np.asarray(
        mask_image,
        dtype=np.uint8,
    ) > 0

    return mask


def load_rgb_and_support(
    raw_row: dict[str, Any],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    target_h, target_w = target_shape

    image_path = resolve_path(
        str(raw_row["image_path"])
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    raw_w, raw_h = image.size

    support = polygon_support_mask(
        raw_row,
        raw_size=(raw_w, raw_h),
    )

    image = image.resize(
        (target_w, target_h),
        Image.Resampling.LANCZOS,
    )

    support_image = Image.fromarray(
        support.astype(np.uint8) * 255,
        mode="L",
    ).resize(
        (target_w, target_h),
        Image.Resampling.NEAREST,
    )

    rgb = np.asarray(
        image,
        dtype=np.uint8,
    )

    support = np.asarray(
        support_image,
        dtype=np.uint8,
    ) > 0

    if not support.any():
        support = np.ones(
            (target_h, target_w),
            dtype=bool,
        )

    return rgb, support


def white_balance_inside_support(
    rgb: np.ndarray,
    support: np.ndarray,
    percentile: float = 88.0,
) -> np.ndarray:
    pixels = rgb[support]

    if len(pixels) == 0:
        return rgb.copy()

    reference = np.percentile(
        pixels.astype(np.float32),
        percentile,
        axis=0,
    )

    reference = np.maximum(
        reference,
        1.0,
    )

    scale = 245.0 / reference

    scale = np.clip(
        scale,
        0.80,
        2.20,
    )

    balanced = (
        rgb.astype(np.float32)
        * scale[None, None, :]
    )

    balanced = np.clip(
        balanced,
        0,
        255,
    ).astype(np.uint8)

    balanced[~support] = 255

    return balanced


def grayscale_from_rgb(
    rgb: np.ndarray,
) -> np.ndarray:
    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )


def estimate_background(
    gray: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    values = gray[support]

    if len(values) == 0:
        return np.full_like(
            gray,
            255,
        )

    fill_value = int(
        np.percentile(values, 88)
    )

    work = gray.copy()
    work[~support] = fill_value

    h, w = gray.shape

    kernel_size = int(
        round(min(h, w) * 0.27)
    )

    kernel_size = max(
        11,
        min(kernel_size, 31),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    background = cv2.morphologyEx(
        work,
        cv2.MORPH_CLOSE,
        kernel,
    )

    sigma = max(
        1.5,
        min(h, w) / 24.0,
    )

    background = cv2.GaussianBlur(
        background,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return background


def flattened_gray(
    rgb: np.ndarray,
    support: np.ndarray,
    *,
    white_balance: bool,
    gain: float = 2.2,
) -> np.ndarray:
    work_rgb = (
        white_balance_inside_support(
            rgb,
            support,
        )
        if white_balance
        else rgb.copy()
    )

    gray = grayscale_from_rgb(
        work_rgb
    )

    background = estimate_background(
        gray,
        support,
    )

    darkness = (
        background.astype(np.float32)
        - gray.astype(np.float32)
    )

    darkness = np.clip(
        darkness,
        0,
        255,
    )

    normalized = (
        255.0
        - darkness * float(gain)
    )

    normalized = np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)

    normalized[~support] = 255

    return normalized


def keep_weak_components_with_strong_seed(
    weak: np.ndarray,
    strong: np.ndarray,
    min_size: int = 4,
) -> np.ndarray:
    weak = np.asarray(
        weak,
        dtype=bool,
    )

    strong = np.asarray(
        strong,
        dtype=bool,
    )

    count, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8,
    )

    output = np.zeros_like(
        weak,
        dtype=bool,
    )

    for component_id in range(
        1,
        count,
    ):
        component = (
            labels == component_id
        )

        if int(component.sum()) < min_size:
            continue

        if np.logical_and(
            component,
            strong,
        ).any():
            output[component] = True

    return cleanup(
        output,
        min_size=min_size,
    )


def school_dark_auto_v2(
    gray: np.ndarray,
) -> np.ndarray:
    fg145 = cleanup(
        gray < 145
    )

    if float(fg145.mean()) <= 0.25:
        return fg145

    return cleanup(
        gray < 120
    )


def global_dark_120(
    gray: np.ndarray,
) -> np.ndarray:
    return cleanup(
        gray < 120
    )


def adaptive_threshold(
    gray: np.ndarray,
    support: np.ndarray,
) -> float:
    values = gray[support]

    if len(values) == 0:
        return 132.0

    background_level = float(
        np.percentile(
            values,
            88,
        )
    )

    threshold = (
        background_level
        - 45.0
    )

    return float(
        np.clip(
            threshold,
            120.0,
            145.0,
        )
    )


def adaptive_raw_hysteresis(
    gray: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    threshold = adaptive_threshold(
        gray,
        support,
    )

    strong_threshold = max(
        100.0,
        threshold - 12.0,
    )

    weak_threshold = min(
        155.0,
        threshold + 10.0,
    )

    strong = np.logical_and(
        gray < strong_threshold,
        support,
    )

    weak = np.logical_and(
        gray < weak_threshold,
        support,
    )

    result = keep_weak_components_with_strong_seed(
        weak,
        strong,
    )

    if not result.any():
        result = cleanup(
            np.logical_and(
                gray < threshold,
                support,
            )
        )

    return result


def flattened_hysteresis(
    rgb: np.ndarray,
    support: np.ndarray,
    *,
    white_balance: bool,
) -> np.ndarray:
    normalized = flattened_gray(
        rgb,
        support,
        white_balance=white_balance,
    )

    strong = np.logical_and(
        normalized < 120,
        support,
    )

    weak = np.logical_and(
        normalized < 150,
        support,
    )

    result = keep_weak_components_with_strong_seed(
        weak,
        strong,
    )

    if not result.any():
        values = normalized[support]

        if len(values):
            otsu = float(
                threshold_otsu(values)
            )

            otsu = float(
                np.clip(
                    otsu,
                    120,
                    180,
                )
            )

            result = cleanup(
                np.logical_and(
                    normalized < otsu,
                    support,
                )
            )

    return result


def erase_support_border(
    mask: np.ndarray,
    support: np.ndarray,
    border_px: int = 4,
) -> np.ndarray:
    support_u8 = support.astype(np.uint8)

    dist = cv2.distanceTransform(
        support_u8,
        cv2.DIST_L2,
        3,
    )

    inner = dist > float(border_px)

    return np.logical_and(mask, inner)


def raw_seeded_hysteresis(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    strong_threshold: int = 120,
    weak_threshold: int = 145,
    min_size: int = 4,
    border_px: int = 2,
) -> np.ndarray:
    active_support = support.astype(bool)

    if border_px > 0:
        support_u8 = active_support.astype(np.uint8)

        dist = cv2.distanceTransform(
            support_u8,
            cv2.DIST_L2,
            3,
        )

        active_support = dist > float(border_px)

    strong = np.logical_and(
        gray < strong_threshold,
        active_support,
    )

    weak = np.logical_and(
        gray < weak_threshold,
        active_support,
    )

    result = keep_weak_components_with_strong_seed(
        weak,
        strong,
        min_size=min_size,
    )

    return result


def flat_raw_seeded_rescue(
    rgb: np.ndarray,
    gray: np.ndarray,
    support: np.ndarray,
    *,
    white_balance: bool = False,
    strong_raw_threshold: int = 120,
    weak_flat_threshold: int = 150,
    border_px: int = 5,
    min_size: int = 4,
) -> np.ndarray:
    support = support.astype(bool)

    support_u8 = support.astype(np.uint8)

    dist = cv2.distanceTransform(
        support_u8,
        cv2.DIST_L2,
        3,
    )

    inner_support = dist > float(border_px)

    normalized = flattened_gray(
        rgb,
        inner_support,
        white_balance=white_balance,
        gain=2.2,
    )

    raw_seed = np.logical_and(
        gray < strong_raw_threshold,
        inner_support,
    )

    flat_seed = np.logical_and(
        normalized < 115,
        inner_support,
    )

    strong = np.logical_or(
        raw_seed,
        flat_seed,
    )

    weak = np.logical_and(
        normalized < weak_flat_threshold,
        inner_support,
    )

    result = keep_weak_components_with_strong_seed(
        weak,
        strong,
        min_size=min_size,
    )

    result = erase_support_border(
        result,
        support,
        border_px=border_px,
    )

    return cleanup(
        result,
        min_size=min_size,
    )


def choose_school_foreground_v5(
    rgb: np.ndarray,
    gray: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, str]:
    base = raw_seeded_hysteresis(
        gray,
        support,
        strong_threshold=120,
        weak_threshold=145,
        border_px=2,
    )

    base_fraction = float(base.mean())

    if 0.006 <= base_fraction <= 0.30:
        return base, "raw_seeded_145_120"

    conservative = raw_seeded_hysteresis(
        gray,
        support,
        strong_threshold=120,
        weak_threshold=140,
        border_px=2,
    )

    conservative_fraction = float(
        conservative.mean()
    )

    if 0.006 <= conservative_fraction <= 0.30:
        return conservative, "raw_seeded_140_120"

    rescue = flat_raw_seeded_rescue(
        rgb,
        gray,
        support,
        white_balance=False,
        strong_raw_threshold=120,
        weak_flat_threshold=150,
        border_px=5,
    )

    rescue_fraction = float(rescue.mean())

    if 0.006 <= rescue_fraction <= 0.30:
        return rescue, "flat_raw_seeded_rescue"

    fallback = global_dark_120(gray)

    return fallback, "global_dark_120_fallback"

def dilate_mask(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)

    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)


def local_detail_recovery(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    strong_threshold: int = 120,
    weak_threshold: int = 145,
    radius: int = 1,
    min_size: int = 4,
) -> np.ndarray:
    support = support.astype(bool)

    strong = np.logical_and(
        gray < strong_threshold,
        support,
    )

    weak = np.logical_and(
        gray < weak_threshold,
        support,
    )

    near_strong = dilate_mask(
        strong,
        radius=radius,
    )

    recovered = np.logical_or(
        strong,
        np.logical_and(
            weak,
            near_strong,
        ),
    )

    return cleanup(
        recovered,
        min_size=min_size,
    )


def local_detail_recovery_adaptive(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    min_size: int = 4,
) -> np.ndarray:
    support = support.astype(bool)

    fg145 = cleanup(
        np.logical_and(
            gray < 145,
            support,
        ),
        min_size=min_size,
    )

    # Если мягкий порог не перегружает фон, можно использовать v2-поведение.
    if float(fg145.mean()) <= 0.25:
        return fg145

    # Иначе берём 120 как ядро и возвращаем только ближайшие слабые края.
    return local_detail_recovery(
        gray,
        support,
        strong_threshold=120,
        weak_threshold=145,
        radius=1,
        min_size=min_size,
    )


def local_background_response(
    gray: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    support = support.astype(bool)

    values = gray[support]
    if len(values) == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    fill = int(np.percentile(values, 88))
    work = gray.copy()
    work[~support] = fill

    h, w = gray.shape
    kernel_size = max(15, int(round(min(h, w) * 0.35)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    background = cv2.morphologyEx(
        work,
        cv2.MORPH_CLOSE,
        kernel,
    )

    background = cv2.GaussianBlur(
        background,
        (0, 0),
        sigmaX=max(2.0, min(h, w) / 18.0),
        sigmaY=max(2.0, min(h, w) / 18.0),
    )

    response = (
        background.astype(np.float32)
        - gray.astype(np.float32)
    )

    response = np.clip(response, 0, 255)

    response[~support] = 0

    return response.astype(np.uint8)


def dark_image_rescue_limited(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    response_threshold: int = 28,
    raw_seed_threshold: int = 130,
    detail_radius: int = 2,
    min_size: int = 4,
) -> np.ndarray:
    support = support.astype(bool)

    response = local_background_response(
        gray,
        support,
    )

    raw_seed = np.logical_and(
        gray < raw_seed_threshold,
        support,
    )

    response_mask = np.logical_and(
        response > response_threshold,
        support,
    )

    near_seed = dilate_mask(
        raw_seed,
        radius=detail_radius,
    )

    recovered = np.logical_or(
        raw_seed,
        np.logical_and(
            response_mask,
            near_seed,
        ),
    )

    return cleanup(
        recovered,
        min_size=min_size,
    )


def support_inner_mask(
    support: np.ndarray,
    border_px: int = 4,
) -> np.ndarray:
    support = support.astype(bool)

    if not support.any():
        return support

    dist = cv2.distanceTransform(
        support.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )

    inner = dist > float(border_px)

    if not inner.any():
        return support

    return inner.astype(bool)


def masked_gaussian_background(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    border_px: int = 4,
    sigma_scale: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    inner = support_inner_mask(
        support,
        border_px=border_px,
    )

    values = gray[inner]

    if len(values) == 0:
        fill_value = int(np.percentile(gray, 90))
    else:
        fill_value = int(np.percentile(values, 90))

    work = gray.copy()
    work[~inner] = fill_value

    h, w = gray.shape
    sigma = max(5.0, min(h, w) * sigma_scale)

    background = cv2.GaussianBlur(
        work,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    background = np.maximum(background, 20)

    return background.astype(np.uint8), inner


def paper_ratio_normalize_gray(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    border_px: int = 4,
    target_paper: float = 245.0,
    sigma_scale: float = 0.20,
) -> np.ndarray:
    background, inner = masked_gaussian_background(
        gray,
        support,
        border_px=border_px,
        sigma_scale=sigma_scale,
    )

    corrected = (
        gray.astype(np.float32)
        / np.maximum(background.astype(np.float32), 1.0)
        * float(target_paper)
    )

    corrected = np.clip(
        corrected,
        0,
        255,
    ).astype(np.uint8)

    corrected[~inner] = 255

    return corrected


def paper_ratio_normalize_rgb(
    rgb: np.ndarray,
    support: np.ndarray,
    *,
    border_px: int = 4,
    target_paper: float = 245.0,
    sigma_scale: float = 0.20,
) -> np.ndarray:
    support = support.astype(bool)
    inner = support_inner_mask(
        support,
        border_px=border_px,
    )

    values = rgb[inner]

    if len(values) > 0:
        paper_rgb = np.percentile(
            values.astype(np.float32),
            90,
            axis=0,
        )
        paper_rgb = np.maximum(paper_rgb, 1.0)
        scale = target_paper / paper_rgb
        scale = np.clip(scale, 0.75, 2.50)

        balanced = (
            rgb.astype(np.float32)
            * scale[None, None, :]
        )
        balanced = np.clip(
            balanced,
            0,
            255,
        ).astype(np.uint8)
    else:
        balanced = rgb.copy()

    gray = grayscale_from_rgb(balanced)

    return paper_ratio_normalize_gray(
        gray,
        inner,
        border_px=border_px,
        target_paper=target_paper,
        sigma_scale=sigma_scale,
    )


def paper_ratio_dark_auto(
    gray: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    normalized = paper_ratio_normalize_gray(
        gray,
        support,
        border_px=4,
        target_paper=245.0,
        sigma_scale=0.20,
    )

    return school_dark_auto_v2(normalized)


def paper_ratio_threshold(
    gray: np.ndarray,
    support: np.ndarray,
    threshold: int,
) -> np.ndarray:
    normalized = paper_ratio_normalize_gray(
        gray,
        support,
        border_px=4,
        target_paper=245.0,
        sigma_scale=0.20,
    )

    return cleanup(
        np.logical_and(
            normalized < int(threshold),
            support_inner_mask(support, border_px=4),
        )
    )


def paper_ratio_rgb_dark_auto(
    rgb: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    normalized = paper_ratio_normalize_rgb(
        rgb,
        support,
        border_px=4,
        target_paper=245.0,
        sigma_scale=0.20,
    )

    return school_dark_auto_v2(normalized)


def choose_school_foreground_v6(
    rgb: np.ndarray,
    gray: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, str]:
    # 1. Сначала пробуем обычный v2: он уже хорош на нормальных crop.
    v2 = school_dark_auto_v2(gray)
    v2_fraction = float(v2.mean())

    if 0.006 <= v2_fraction <= 0.24:
        return v2, "school_dark_auto_v2"

    # 2. Если v2 слишком жирный, сначала пробуем paper whitening.
    ratio_auto = paper_ratio_dark_auto(gray, support)
    ratio_auto_fraction = float(ratio_auto.mean())

    if 0.006 <= ratio_auto_fraction <= 0.24:
        return ratio_auto, "paper_ratio_dark_auto"

    # 3. Более мягкий вариант после whitening.
    ratio_135 = paper_ratio_threshold(
        gray,
        support,
        threshold=135,
    )
    ratio_135_fraction = float(ratio_135.mean())

    if 0.006 <= ratio_135_fraction <= 0.24:
        return ratio_135, "paper_ratio_135"

    # 4. Консервативный fallback.
    conservative = global_dark_120(gray)
    return conservative, "global_dark_120"

def keep_components_near_seed(
    weak: np.ndarray,
    strong: np.ndarray,
    *,
    proximity_px: int = 2,
    min_size: int = 4,
) -> np.ndarray:
    weak = np.asarray(weak, dtype=bool)
    strong = np.asarray(strong, dtype=bool)

    kernel_size = 2 * proximity_px + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    seed_zone = cv2.dilate(
        strong.astype(np.uint8),
        kernel,
        iterations=1,
    ) > 0

    component_count, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8,
    )

    result = np.zeros_like(
        weak,
        dtype=bool,
    )

    for component_id in range(1, component_count):
        component = labels == component_id
        component_size = int(component.sum())

        if component_size < min_size:
            continue

        if np.logical_and(
            component,
            seed_zone,
        ).any():
            result[component] = True

    return cleanup(
        result,
        min_size=min_size,
    )


def support_border_fraction(
    mask: np.ndarray,
    support: np.ndarray,
    *,
    ring_px: int = 3,
) -> float:
    mask = np.asarray(mask, dtype=bool)
    support = np.asarray(support, dtype=bool)

    foreground_count = int(mask.sum())

    if foreground_count == 0:
        return 0.0

    kernel_size = 2 * ring_px + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    eroded = cv2.erode(
        support.astype(np.uint8),
        kernel,
        iterations=1,
    ) > 0

    support_ring = np.logical_and(
        support,
        np.logical_not(eroded),
    )

    return float(
        np.logical_and(
            mask,
            support_ring,
        ).sum()
        / foreground_count
    )


def paper_ratio_capped_hysteresis(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    strong_threshold: int = 120,
    maximum_weak_threshold: int = 135,
    maximum_total_fraction: float = 0.24,
    maximum_expansion: float = 1.50,
    minimum_extra_fraction: float = 0.015,
    maximum_border_fraction: float = 0.12,
    proximity_px: int = 2,
    min_size: int = 4,
) -> tuple[np.ndarray, int]:
    normalized = paper_ratio_normalize_gray(
        gray,
        support,
        border_px=4,
        target_paper=245.0,
        sigma_scale=0.20,
    )

    inner = support_inner_mask(
        support,
        border_px=4,
    )

    support_count = max(
        int(inner.sum()),
        1,
    )

    strong = cleanup(
        np.logical_and(
            normalized < strong_threshold,
            inner,
        ),
        min_size=min_size,
    )

    strong_count = int(strong.sum())
    strong_fraction = (
        strong_count / support_count
    )

    # Если даже уверенного ядра почти нет,
    # conservative threshold уже не может служить seed.
    if strong_fraction < 0.003:
        fallback = cleanup(
            np.logical_and(
                normalized < maximum_weak_threshold,
                inner,
            ),
            min_size=min_size,
        )

        return fallback, maximum_weak_threshold

    allowed_fraction = min(
        maximum_total_fraction,
        max(
            strong_fraction * maximum_expansion,
            strong_fraction + minimum_extra_fraction,
        ),
    )

    # Начинаем с самого мягкого порога:
    # выбираем максимальную детализацию,
    # удовлетворяющую ограничениям.
    for threshold in range(
        maximum_weak_threshold,
        strong_threshold,
        -1,
    ):
        weak = np.logical_and(
            normalized < threshold,
            inner,
        )

        candidate = keep_components_near_seed(
            weak,
            strong,
            proximity_px=proximity_px,
            min_size=min_size,
        )

        candidate_count = int(
            candidate.sum()
        )

        candidate_fraction = (
            candidate_count
            / support_count
        )

        expansion = (
            candidate_count
            / max(strong_count, 1)
        )

        border_fraction = (
            support_border_fraction(
                candidate,
                inner,
                ring_px=3,
            )
        )

        if candidate_fraction > allowed_fraction:
            continue

        if expansion > maximum_expansion:
            continue

        if border_fraction > maximum_border_fraction:
            continue

        return candidate, threshold

    return strong, strong_threshold

def nearest_fill_outside_support(
    gray: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    gray = np.asarray(
        gray,
        dtype=np.uint8,
    )

    support = np.asarray(
        support,
        dtype=bool,
    )

    if support.all():
        return gray.copy()

    if not support.any():
        return gray.copy()

    _, nearest_indices = distance_transform_edt(
        np.logical_not(support),
        return_indices=True,
    )

    filled = gray.copy()

    outside = np.logical_not(support)

    filled[outside] = gray[
        nearest_indices[0][outside],
        nearest_indices[1][outside],
    ]

    return filled


def global_affine_whiten(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    paper_quantile: float = 0.85,
    ink_quantile: float = 0.05,
    target_paper: float = 245.0,
    target_ink: float = 40.0,
) -> np.ndarray:
    gray = np.asarray(
        gray,
        dtype=np.uint8,
    )

    support = np.asarray(
        support,
        dtype=bool,
    )

    values = gray[support].astype(
        np.float32
    )

    if len(values) == 0:
        return gray.copy()

    paper = float(
        np.quantile(
            values,
            paper_quantile,
        )
    )

    ink = float(
        np.quantile(
            values,
            ink_quantile,
        )
    )

    span = max(
        paper - ink,
        20.0,
    )

    gain = (
        target_paper - target_ink
    ) / span

    gain = float(
        np.clip(
            gain,
            0.8,
            3.0,
        )
    )

    normalized = (
        target_paper
        - gain
        * (
            paper
            - gray.astype(np.float32)
        )
    )

    normalized = np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)

    normalized[
        np.logical_not(support)
    ] = 255

    return normalized


def estimate_masked_local_background(
    gray: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    gray = np.asarray(
        gray,
        dtype=np.uint8,
    )

    support = np.asarray(
        support,
        dtype=bool,
    )

    filled = nearest_fill_outside_support(
        gray,
        support,
    )

    h, w = filled.shape

    kernel_size = int(
        round(min(h, w) * 0.27)
    )

    kernel_size = max(
        11,
        min(kernel_size, 31),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    # Grayscale closing подавляет тёмные
    # штрихи небольшой толщины при оценке бумаги.
    background = cv2.morphologyEx(
        filled,
        cv2.MORPH_CLOSE,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )

    sigma = max(
        1.0,
        min(h, w) / 32.0,
    )

    background = cv2.GaussianBlur(
        background,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )

    return background.astype(
        np.uint8
    )


def masked_local_whiten(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    target_paper: float = 245.0,
    target_ink: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    gray = np.asarray(
        gray,
        dtype=np.uint8,
    )

    support = np.asarray(
        support,
        dtype=bool,
    )

    background = (
        estimate_masked_local_background(
            gray,
            support,
        )
    )

    darkness = np.maximum(
        background.astype(np.float32)
        - gray.astype(np.float32),
        0.0,
    )

    supported_darkness = darkness[
        support
    ]

    if len(supported_darkness) == 0:
        normalized = gray.copy()
        normalized[~support] = 255
        return normalized, darkness

    reference = float(
        np.quantile(
            supported_darkness,
            0.95,
        )
    )

    reference = max(
        reference,
        8.0,
    )

    normalized = (
        target_paper
        - (
            target_paper
            - target_ink
        )
        * np.clip(
            darkness / reference,
            0.0,
            1.0,
        )
    )

    normalized = np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)

    normalized[~support] = 255

    return normalized, darkness


def adaptive_foreground_from_whitened(
    normalized: np.ndarray,
    support: np.ndarray,
    *,
    min_size: int = 3,
) -> tuple[np.ndarray, float]:
    normalized = np.asarray(
        normalized,
        dtype=np.uint8,
    )

    support = np.asarray(
        support,
        dtype=bool,
    )

    values = normalized[
        support
    ].astype(np.float32)

    if len(values) == 0:
        return (
            np.zeros_like(
                support,
                dtype=bool,
            ),
            0.0,
        )

    paper_level = float(
        np.quantile(
            values,
            0.85,
        )
    )

    darkness = np.maximum(
        paper_level
        - normalized.astype(np.float32),
        0.0,
    )

    supported_darkness = darkness[
        support
    ]

    paper_cutoff = float(
        np.quantile(
            values,
            0.60,
        )
    )

    probable_paper = np.logical_and(
        support,
        normalized >= paper_cutoff,
    )

    noise_values = darkness[
        probable_paper
    ]

    if len(noise_values) == 0:
        noise_median = 0.0
        noise_sigma = 0.0
    else:
        noise_median = float(
            np.median(noise_values)
        )

        mad = float(
            np.median(
                np.abs(
                    noise_values
                    - noise_median
                )
            )
        )

        noise_sigma = (
            1.4826 * mad
        )

    signal_reference = float(
        np.quantile(
            supported_darkness,
            0.95,
        )
    )

    threshold = max(
        noise_median
        + 3.0 * noise_sigma,
        0.12 * signal_reference,
        4.0,
    )

    if signal_reference > 0:
        threshold = min(
            threshold,
            0.40 * signal_reference,
        )

    foreground = np.logical_and(
        darkness > threshold,
        support,
    )

    foreground = cleanup(
        foreground,
        min_size=min_size,
    )

    return foreground, float(threshold)


def global_affine_whiten_foreground(
    gray: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    corrected = derim_gray(
        gray,
        support,
        ring_px=3,
    )

    normalized = global_affine_whiten(
        corrected,
        support,
    )

    foreground, threshold = adaptive_foreground_from_whitened(
        normalized,
        support,
    )

    foreground = remove_ring_only_components(
        foreground,
        support,
        ring_px=3,
        min_inner_overlap=1,
    )

    foreground = cleanup(
        foreground,
        min_size=3,
    )

    return foreground, normalized, threshold


def masked_local_whiten_foreground(
    gray: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    corrected = derim_gray(
        gray,
        support,
        ring_px=3,
    )

    normalized, _darkness = masked_local_whiten(
        corrected,
        support,
    )

    foreground, threshold = adaptive_foreground_from_whitened(
        normalized,
        support,
    )

    foreground = remove_ring_only_components(
        foreground,
        support,
        ring_px=3,
        min_inner_overlap=1,
    )

    foreground = cleanup(
        foreground,
        min_size=3,
    )

    return foreground, normalized, threshold

def support_distance_masks(
    support: np.ndarray,
    *,
    ring_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support, dtype=bool)

    if not support.any():
        empty = np.zeros_like(support, dtype=bool)
        return empty, empty

    dist = cv2.distanceTransform(
        support.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )

    inner = dist > float(ring_px)
    ring = np.logical_and(support, np.logical_not(inner))

    return inner.astype(bool), ring.astype(bool)


def fill_ring_from_inner(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    ring_px: int = 3,
) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    support = np.asarray(support, dtype=bool)

    inner, ring = support_distance_masks(
        support,
        ring_px=ring_px,
    )

    if not ring.any() or not inner.any():
        result = gray.copy()
        result[np.logical_not(support)] = 255
        return result

    # nearest inner pixel for every ring pixel
    _, nearest_indices = distance_transform_edt(
        np.logical_not(inner),
        return_indices=True,
    )

    result = gray.copy()

    result[ring] = gray[
        nearest_indices[0][ring],
        nearest_indices[1][ring],
    ]

    result[np.logical_not(support)] = 255
    return result


def remove_ring_only_components(
    foreground: np.ndarray,
    support: np.ndarray,
    *,
    ring_px: int = 3,
    min_inner_overlap: int = 1,
) -> np.ndarray:
    foreground = np.asarray(foreground, dtype=bool)
    support = np.asarray(support, dtype=bool)

    inner, ring = support_distance_masks(
        support,
        ring_px=ring_px,
    )

    n, labels = cv2.connectedComponents(
        foreground.astype(np.uint8),
        connectivity=8,
    )

    result = np.zeros_like(foreground, dtype=bool)

    for component_id in range(1, n):
        component = labels == component_id

        inner_overlap = int(
            np.logical_and(component, inner).sum()
        )

        if inner_overlap >= min_inner_overlap:
            result[component] = True
            continue

        # компонент живёт только в кольце — удаляем
        ring_overlap = int(
            np.logical_and(component, ring).sum()
        )

        if ring_overlap == 0:
            result[component] = True

    return result.astype(bool)


def derim_gray(
    gray: np.ndarray,
    support: np.ndarray,
    *,
    ring_px: int = 3,
) -> np.ndarray:
    corrected = fill_ring_from_inner(
        gray,
        support,
        ring_px=ring_px,
    )
    corrected[np.logical_not(support)] = 255
    return corrected