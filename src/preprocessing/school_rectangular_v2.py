from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from PIL import Image, ImageDraw

SPLIT_FILES = {
    "train": "annotations_train.json",
    "val": "annotations_val.json",
    "test": "annotations_test.json",
}


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in [
        "metadata",
        "source_metadata",
    ]:
        value = row.get(key)

        if isinstance(value, dict):
            result.update(value)

    return result


def flatten_segmentation(
    segmentation: Any,
) -> list[float]:
    if not segmentation:
        return []

    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        coordinates: list[float] = []

        for polygon in segmentation:
            coordinates.extend(float(value) for value in polygon)

        return coordinates

    if isinstance(segmentation, list):
        return [float(value) for value in segmentation]

    return []


def coordinates_to_points(
    coordinates: list[float],
) -> list[tuple[float, float]]:
    points = []

    for index in range(
        0,
        len(coordinates) - 1,
        2,
    ):
        points.append(
            (
                coordinates[index],
                coordinates[index + 1],
            )
        )

    return points


def bbox_from_points(
    points: list[tuple[float, float]],
    *,
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    if len(points) < 3:
        raise ValueError("Polygon contains fewer than three points")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x0 = max(
        0,
        int(math.floor(min(xs))) - padding,
    )
    y0 = max(
        0,
        int(math.floor(min(ys))) - padding,
    )
    x1 = min(
        image_width,
        int(math.ceil(max(xs))) + padding,
    )
    y1 = min(
        image_height,
        int(math.ceil(max(ys))) + padding,
    )

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bounding box: {x0, y0, x1, y1}")

    return x0, y0, x1, y1


class SchoolCocoSource:
    def __init__(
        self,
        raw_dir: str | Path,
    ) -> None:
        self.raw_dir = Path(raw_dir).resolve()
        self.images_root = self.raw_dir / "images" / "images"

        if not self.images_root.exists():
            raise FileNotFoundError(self.images_root)

        self.annotations: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        self.images: dict[
            str,
            dict[int, dict[str, Any]],
        ] = {}

        for split, filename in SPLIT_FILES.items():
            path = self.raw_dir / filename

            if not path.exists():
                raise FileNotFoundError(path)

            data = json.loads(path.read_text(encoding="utf-8"))

            self.annotations[split] = list(data["annotations"])

            self.images[split] = {int(item["id"]): item for item in data["images"]}

    def resolve_annotation(
        self,
        row: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        split = str(row.get("split", ""))
        meta = metadata(row)

        page_id_value = meta.get("page_id")
        word_id_value = meta.get("word_id")

        if split not in self.annotations:
            raise KeyError(f"Unknown split {split!r}")

        if page_id_value in [None, ""]:
            raise KeyError(f"No page_id for {row.get('sample_id')}")

        if word_id_value in [None, ""]:
            raise KeyError(f"No word_id for {row.get('sample_id')}")

        image_id = int(page_id_value)
        annotation_index = int(word_id_value)

        annotations = self.annotations[split]

        if not (0 <= annotation_index < len(annotations)):
            raise IndexError(f"annotation_index={annotation_index}, n={len(annotations)}")

        annotation = annotations[annotation_index]

        annotation_image_id = int(annotation["image_id"])

        if annotation_image_id != image_id:
            raise RuntimeError(
                f"Image ID mismatch for "
                f"{row.get('sample_id')}: "
                f"metadata={image_id}, "
                f"annotation={annotation_image_id}"
            )

        image_info = self.images[split].get(image_id)

        if image_info is None:
            raise KeyError(f"No image metadata for split={split}, image_id={image_id}")

        expected_filename = str(
            meta.get(
                "source_image_file",
                "",
            )
        )

        actual_filename = str(image_info["file_name"])

        if expected_filename and expected_filename != actual_filename:
            raise RuntimeError(
                f"Source filename mismatch for "
                f"{row.get('sample_id')}: "
                f"metadata={expected_filename!r}, "
                f"COCO={actual_filename!r}"
            )

        return annotation, image_info

    def load_rectangular_crop(
        self,
        row: dict[str, Any],
        *,
        target_height: int = 64,
        extra_padding: int = 0,
    ) -> np.ndarray:
        annotation, image_info = self.resolve_annotation(row)

        coordinates = flatten_segmentation(annotation.get("segmentation"))

        points = coordinates_to_points(coordinates)

        meta = metadata(row)

        original_padding = int(
            meta.get(
                "crop_padding",
                8,
            )
        )

        total_padding = original_padding + int(extra_padding)

        image_width = int(image_info["width"])
        image_height = int(image_info["height"])

        bbox = bbox_from_points(
            points,
            image_width=image_width,
            image_height=image_height,
            padding=total_padding,
        )

        source_path = self.images_root / str(image_info["file_name"])

        if not source_path.exists():
            raise FileNotFoundError(source_path)

        with Image.open(source_path) as source:
            source = source.convert("RGB")

            crop = source.crop(bbox)

            scale = target_height / max(crop.height, 1)

            target_width = max(
                1,
                int(round(crop.width * scale)),
            )

            crop = crop.resize(
                (
                    target_width,
                    target_height,
                ),
                Image.Resampling.LANCZOS,
            )

            gray = np.asarray(
                crop.convert("L"),
                dtype=np.uint8,
            )

        return gray

    def load_rectangular_crop_with_polygon_mask(
        self,
        row: dict[str, Any],
        *,
        target_height: int = 64,
        extra_padding: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        annotation, image_info = self.resolve_annotation(row)

        coordinates = flatten_segmentation(annotation.get("segmentation"))

        points = coordinates_to_points(coordinates)

        meta = metadata(row)

        original_padding = int(
            meta.get(
                "crop_padding",
                8,
            )
        )

        total_padding = original_padding + int(extra_padding)

        image_width = int(image_info["width"])
        image_height = int(image_info["height"])

        x0, y0, x1, y1 = bbox_from_points(
            points,
            image_width=image_width,
            image_height=image_height,
            padding=total_padding,
        )

        source_path = self.images_root / str(image_info["file_name"])

        if not source_path.exists():
            raise FileNotFoundError(source_path)

        crop_width = x1 - x0
        crop_height = y1 - y0

        polygon_points = [
            (
                float(x) - x0,
                float(y) - y0,
            )
            for x, y in points
        ]

        polygon_image = Image.new(
            "L",
            (crop_width, crop_height),
            0,
        )

        draw = ImageDraw.Draw(polygon_image)

        draw.polygon(
            polygon_points,
            fill=255,
        )

        with Image.open(source_path) as source:
            source = source.convert("RGB")

            crop = source.crop((x0, y0, x1, y1))

            scale = target_height / max(crop.height, 1)

            target_width = max(
                1,
                int(round(crop.width * scale)),
            )

            crop = crop.resize(
                (
                    target_width,
                    target_height,
                ),
                Image.Resampling.LANCZOS,
            )

            polygon_image = polygon_image.resize(
                (
                    target_width,
                    target_height,
                ),
                Image.Resampling.NEAREST,
            )

            gray = np.asarray(
                crop.convert("L"),
                dtype=np.uint8,
            )

            polygon_mask = (
                np.asarray(
                    polygon_image,
                    dtype=np.uint8,
                )
                > 0
            )

        return gray, polygon_mask

    def load_rectangular_rgb_with_polygon_mask(
        self,
        row: dict[str, Any],
        *,
        target_height: int = 64,
        extra_padding: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        annotation, image_info = self.resolve_annotation(row)

        coordinates = flatten_segmentation(annotation.get("segmentation"))
        points = coordinates_to_points(coordinates)

        meta = metadata(row)

        original_padding = int(meta.get("crop_padding", 8))

        total_padding = original_padding + int(extra_padding)

        image_width = int(image_info["width"])
        image_height = int(image_info["height"])

        x0, y0, x1, y1 = bbox_from_points(
            points,
            image_width=image_width,
            image_height=image_height,
            padding=total_padding,
        )

        source_path = self.images_root / str(image_info["file_name"])

        if not source_path.exists():
            raise FileNotFoundError(source_path)

        crop_width = x1 - x0
        crop_height = y1 - y0

        local_polygon = [
            (
                float(x) - x0,
                float(y) - y0,
            )
            for x, y in points
        ]

        polygon_image = Image.new(
            "L",
            (crop_width, crop_height),
            0,
        )

        draw = ImageDraw.Draw(polygon_image)
        draw.polygon(local_polygon, fill=255)

        with Image.open(source_path) as source:
            source = source.convert("RGB")
            crop = source.crop((x0, y0, x1, y1))

            scale = target_height / max(crop.height, 1)

            target_width = max(
                1,
                int(round(crop.width * scale)),
            )

            crop = crop.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            )

            polygon_image = polygon_image.resize(
                (target_width, target_height),
                Image.Resampling.NEAREST,
            )

            rgb = np.asarray(
                crop,
                dtype=np.uint8,
            )

            polygon_mask = (
                np.asarray(
                    polygon_image,
                    dtype=np.uint8,
                )
                > 0
            )

        return rgb, polygon_mask


def post_binarization_polygon_filter(
    foreground: np.ndarray,
    polygon_mask: np.ndarray,
    *,
    dilation_px: int = 3,
    minimum_polygon_pixels: int = 3,
    minimum_overlap_ratio: float = 0.10,
    minimum_component_size: int = 3,
) -> np.ndarray:
    foreground = np.asarray(
        foreground,
        dtype=bool,
    )

    polygon_mask = np.asarray(
        polygon_mask,
        dtype=bool,
    )

    if not foreground.any():
        return foreground

    if not polygon_mask.any():
        return foreground

    if dilation_px > 0:
        kernel_size = (
            2 * dilation_px + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        relaxed_polygon = cv2.dilate(
            polygon_mask.astype(
                np.uint8
            ),
            kernel,
            iterations=1,
        ) > 0
    else:
        relaxed_polygon = (
            polygon_mask.copy()
        )

    component_count, labels = (
        cv2.connectedComponents(
            foreground.astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    result = np.zeros_like(
        foreground,
        dtype=bool,
    )

    for component_id in range(
        1,
        component_count,
    ):
        component = (
            labels == component_id
        )

        component_size = int(
            component.sum()
        )

        if (
            component_size
            < minimum_component_size
        ):
            continue

        strict_overlap = int(
            np.logical_and(
                component,
                polygon_mask,
            ).sum()
        )

        overlap_ratio = (
            strict_overlap
            / max(component_size, 1)
        )

        # Компонент считается связанным
        # с целевым словом, если он имеет
        # достаточное абсолютное или
        # относительное пересечение.
        related_to_target = (
            strict_overlap
            >= minimum_polygon_pixels
            or overlap_ratio
            >= minimum_overlap_ratio
        )

        if not related_to_target:
            continue

        # Сохраняем целевой компонент,
        # но обрезаем только далёкую часть.
        result |= np.logical_and(
            component,
            relaxed_polygon,
        )

    if result.any():
        result = remove_small_objects(
            result,
            min_size=minimum_component_size,
        )

    return result.astype(bool)

def adaptive_whitened_foreground(
    normalized: np.ndarray,
    polygon_mask: np.ndarray,
    *,
    minimum_threshold: float = 150.0,
    maximum_threshold: float = 220.0,
    maximum_foreground_fraction: float = 0.30,
    minimum_component_size: int = 3,
) -> tuple[np.ndarray, float]:
    """
    Estimate a foreground threshold from whitened grayscale values
    inside the target polygon.

    The polygon is used only for threshold estimation. It is not
    applied to grayscale, so no polygon rim can be created.
    """
    normalized = np.asarray(
        normalized,
        dtype=np.uint8,
    )

    polygon_mask = np.asarray(
        polygon_mask,
        dtype=bool,
    )

    values = normalized[
        polygon_mask
    ].astype(np.float32)

    if values.size < 32:
        threshold = minimum_threshold

        foreground = (
            normalized < threshold
        )

        return (
            remove_small_objects(
                foreground,
                min_size=minimum_component_size,
            ),
            float(threshold),
        )

    unique_values = np.unique(values)

    if unique_values.size < 2:
        threshold = minimum_threshold
    else:
        otsu_threshold = float(
            threshold_otsu(values)
        )

        # Estimate the paper/noise distribution from the brighter
        # portion of the target polygon.
        paper_cutoff = float(
            np.quantile(
                values,
                0.60,
            )
        )

        paper_values = values[
            values >= paper_cutoff
        ]

        if paper_values.size:
            paper_median = float(
                np.median(
                    paper_values
                )
            )

            paper_mad = float(
                np.median(
                    np.abs(
                        paper_values
                        - paper_median
                    )
                )
            )

            paper_sigma = (
                1.4826 * paper_mad
            )

            # Do not move the threshold too close to the paper mode.
            noise_safe_threshold = (
                paper_median
                - max(
                    3.5 * paper_sigma,
                    16.0,
                )
            )
        else:
            noise_safe_threshold = (
                maximum_threshold
            )

        threshold = min(
            max(
                otsu_threshold,
                minimum_threshold,
            ),
            noise_safe_threshold,
            maximum_threshold,
        )

    threshold = float(
        np.clip(
            threshold,
            minimum_threshold,
            maximum_threshold,
        )
    )

    foreground = (
        normalized < threshold
    )

    # Protect against an abnormally high threshold that starts
    # classifying paper texture as ink.
    polygon_foreground_fraction = float(
        foreground[
            polygon_mask
        ].mean()
    )

    if (
        polygon_foreground_fraction
        > maximum_foreground_fraction
    ):
        cap_threshold = float(
            np.quantile(
                values,
                maximum_foreground_fraction,
            )
        )

        threshold = min(
            threshold,
            cap_threshold,
        )

        foreground = (
            normalized < threshold
        )

    foreground = remove_small_objects(
        foreground,
        min_size=minimum_component_size,
    )

    return (
        foreground.astype(bool),
        float(threshold),
    )

def robust_paper_white_balance(
    rgb: np.ndarray,
    polygon_mask: np.ndarray,
    *,
    target_paper: float = 245.0,
) -> np.ndarray:
    """
    Estimate paper colour from bright, low-gradient pixels inside
    the target polygon and map it to neutral light gray.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    polygon_mask = np.asarray(
        polygon_mask,
        dtype=bool,
    )

    luminance = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)

    gx = cv2.Sobel(
        luminance,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        luminance,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = cv2.magnitude(gx, gy)

    masked_luminance = luminance[
        polygon_mask
    ]

    if masked_luminance.size == 0:
        return rgb.copy()

    bright_cutoff = float(
        np.quantile(
            masked_luminance,
            0.65,
        )
    )

    masked_gradient = gradient[
        polygon_mask
    ]

    gradient_cutoff = float(
        np.quantile(
            masked_gradient,
            0.60,
        )
    )

    paper_pixels_mask = (
        polygon_mask
        & (luminance >= bright_cutoff)
        & (gradient <= gradient_cutoff)
    )

    paper_pixels = rgb[
        paper_pixels_mask
    ].astype(np.float32)

    if len(paper_pixels) < 32:
        fallback_cutoff = float(
            np.quantile(
                masked_luminance,
                0.75,
            )
        )

        paper_pixels = rgb[
            polygon_mask
            & (luminance >= fallback_cutoff)
        ].astype(np.float32)

    if len(paper_pixels) == 0:
        return rgb.copy()

    paper_reference = np.percentile(
        paper_pixels,
        80,
        axis=0,
    )

    paper_reference = np.maximum(
        paper_reference,
        1.0,
    )

    gains = (
        float(target_paper)
        / paper_reference
    )

    gains = np.clip(
        gains,
        0.80,
        2.00,
    )

    balanced = (
        rgb.astype(np.float32)
        * gains[None, None, :]
    )

    return np.clip(
        balanced,
        0,
        255,
    ).astype(np.uint8)

def estimate_rectangular_background(
    gray: np.ndarray,
) -> np.ndarray:
    gray = np.asarray(
        gray,
        dtype=np.uint8,
    )

    h, w = gray.shape

    kernel_size = int(
        round(min(h, w) * 0.33)
    )

    kernel_size = max(
        15,
        min(kernel_size, 35),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    background = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )

    background = cv2.GaussianBlur(
        background,
        (0, 0),
        sigmaX=max(1.0, min(h, w) / 32.0),
        sigmaY=max(1.0, min(h, w) / 32.0),
        borderType=cv2.BORDER_REPLICATE,
    )

    return background.astype(np.uint8)

def estimate_horizontal_ruling_response(
    darkness: np.ndarray,
) -> np.ndarray:
    """
    Extract long horizontal darkness shared across a large part
    of the rectangular crop.

    At handwriting intersections, the extra darkness of ink remains
    after subtracting this horizontal baseline.
    """
    darkness = np.asarray(
        darkness,
        dtype=np.float32,
    )

    work = np.clip(
        darkness,
        0,
        255,
    ).astype(np.uint8)

    h, w = work.shape

    long_width = max(
        17,
        int(round(w * 0.30)),
    )

    gap_width = max(
        3,
        int(round(w * 0.025)),
    )

    # Bridge small gaps caused by handwriting intersections.
    closed = cv2.morphologyEx(
        work,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (gap_width, 1),
        ),
    )

    response_1px = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (long_width, 1),
        ),
    )

    response_3px = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (long_width, 3),
        ),
    )

    return np.maximum(
        response_1px,
        response_3px,
    ).astype(np.float32)


def contrast_image_from_darkness(
    darkness: np.ndarray,
    polygon_mask: np.ndarray,
    *,
    target_paper: float = 245.0,
    target_ink: float = 35.0,
    gamma: float = 0.78,
) -> np.ndarray:
    darkness = np.asarray(
        darkness,
        dtype=np.float32,
    )

    polygon_mask = np.asarray(
        polygon_mask,
        dtype=bool,
    )

    values = darkness[
        polygon_mask
    ]

    positive = values[
        values > 0
    ]

    if positive.size == 0:
        return np.full(
            darkness.shape,
            int(target_paper),
            dtype=np.uint8,
        )

    reference = float(
        np.quantile(
            positive,
            0.97,
        )
    )

    reference = max(reference, 6.0)

    score = np.clip(
        darkness / reference,
        0.0,
        1.0,
    )

    # gamma < 1 boosts faint ink more than already-dark ink.
    score = np.power(score, gamma)

    normalized = (
        target_paper
        - (
            target_paper
            - target_ink
        )
        * score
    )

    return np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)


def whitebalance_contrast_foreground(
    rgb: np.ndarray,
    polygon_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    balanced = robust_paper_white_balance(
        rgb,
        polygon_mask,
    )

    gray = cv2.cvtColor(
        balanced,
        cv2.COLOR_RGB2GRAY,
    )

    background = (
        estimate_rectangular_background(
            gray
        )
    )

    darkness = np.maximum(
        background.astype(np.float32)
        - gray.astype(np.float32),
        0.0,
    )

    normalized = (
        contrast_image_from_darkness(
            darkness,
            polygon_mask,
            gamma=0.78,
        )
    )

    foreground, threshold = (
        adaptive_whitened_foreground(
            normalized,
            polygon_mask,
            minimum_threshold=150,
            maximum_threshold=215,
            maximum_foreground_fraction=0.30,
            minimum_component_size=3,
        )
    )

    return foreground, normalized, threshold


def whitebalance_lineaware_foreground(
    rgb: np.ndarray,
    polygon_mask: np.ndarray,
    *,
    ruling_strength: float = 0.85,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    balanced = robust_paper_white_balance(
        rgb,
        polygon_mask,
    )

    gray = cv2.cvtColor(
        balanced,
        cv2.COLOR_RGB2GRAY,
    )

    background = (
        estimate_rectangular_background(
            gray
        )
    )

    darkness = np.maximum(
        background.astype(np.float32)
        - gray.astype(np.float32),
        0.0,
    )

    ruling = (
        estimate_horizontal_ruling_response(
            darkness
        )
    )

    cleaned_darkness = np.maximum(
        darkness
        - float(ruling_strength)
        * ruling,
        0.0,
    )

    normalized = (
        contrast_image_from_darkness(
            cleaned_darkness,
            polygon_mask,
            gamma=0.78,
        )
    )

    foreground, threshold = (
        adaptive_whitened_foreground(
            normalized,
            polygon_mask,
            minimum_threshold=150,
            maximum_threshold=215,
            maximum_foreground_fraction=0.30,
            minimum_component_size=3,
        )
    )

    return foreground, normalized, threshold