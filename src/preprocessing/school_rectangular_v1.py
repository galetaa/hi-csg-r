from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def metadata(
    row: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in [
        "metadata",
        "source_metadata",
    ]:
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


def get_source_page_path(
    row: dict[str, Any],
    *,
    source_images_root: Path | None = None,
) -> Path:
    meta = metadata(row)

    source_image_path = meta.get(
        "source_image_path"
    )

    if source_image_path:
        try:
            return resolve_path(
                str(source_image_path)
            )
        except FileNotFoundError:
            pass

    source_image_file = meta.get(
        "source_image_file"
    )

    if (
        source_image_file
        and source_images_root is not None
    ):
        candidate = (
            source_images_root
            / str(source_image_file)
        )

        if candidate.exists():
            return candidate.resolve()

        # На случай подкаталогов внутри raw images.
        matches = list(
            source_images_root.rglob(
                str(source_image_file)
            )
        )

        if len(matches) == 1:
            return matches[0].resolve()

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple source pages found for "
                f"{source_image_file}: "
                f"{matches[:5]}"
            )

    raise KeyError(
        f"No resolvable source page for "
        f"{row.get('sample_id')}; "
        f"source_image_file={source_image_file!r}, "
        f"source_images_root={source_images_root}"
    )


def get_bbox(
    row: dict[str, Any],
    *,
    fallback_padding: int = 8,
) -> tuple[int, int, int, int]:
    meta = metadata(row)

    value = (
        row.get("bbox")
        or meta.get("crop_bbox_xyxy")
    )

    if (
        isinstance(value, list)
        and len(value) >= 4
    ):
        x0, y0, x1, y1 = [
            int(round(float(x)))
            for x in value[:4]
        ]

        return x0, y0, x1, y1

    polygon = row.get("polygon")

    if (
        isinstance(polygon, list)
        and len(polygon) >= 3
    ):
        points = [
            point
            for point in polygon
            if (
                isinstance(point, list)
                and len(point) >= 2
            )
        ]

        if len(points) >= 3:
            padding = int(
                meta.get(
                    "crop_padding",
                    fallback_padding,
                )
            )

            xs = [
                float(point[0])
                for point in points
            ]
            ys = [
                float(point[1])
                for point in points
            ]

            x0 = int(np.floor(min(xs))) - padding
            y0 = int(np.floor(min(ys))) - padding
            x1 = int(np.ceil(max(xs))) + padding
            y1 = int(np.ceil(max(ys))) + padding

            return x0, y0, x1, y1

    raise KeyError(
        f"No bbox or polygon for "
        f"{row.get('sample_id')}"
    )


def load_rectangular_crop(
    raw_row: dict[str, Any],
    *,
    source_images_root: Path,
    target_height: int = 64,
    extra_padding: int = 0,
) -> np.ndarray:
    source_path = get_source_page_path(
        raw_row,
        source_images_root=source_images_root,
    )

    x0, y0, x1, y1 = get_bbox(
        raw_row
    )

    with Image.open(source_path) as source:
        source = source.convert("RGB")

        width, height = source.size

        x0 = max(
            0,
            x0 - extra_padding,
        )
        y0 = max(
            0,
            y0 - extra_padding,
        )
        x1 = min(
            width,
            x1 + extra_padding,
        )
        y1 = min(
            height,
            y1 + extra_padding,
        )

        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"Invalid bbox for "
                f"{raw_row.get('sample_id')}: "
                f"{x0, y0, x1, y1}"
            )

        crop = source.crop(
            (x0, y0, x1, y1)
        )

        scale = (
            target_height
            / max(crop.height, 1)
        )

        target_width = max(
            1,
            int(round(
                crop.width * scale
            )),
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


def robust_global_whiten(
    gray: np.ndarray,
    *,
    paper_quantile: float = 0.90,
    ink_quantile: float = 0.05,
    target_paper: float = 245.0,
    target_ink: float = 40.0,
    minimum_span: float = 35.0,
) -> np.ndarray:
    gray = np.asarray(
        gray,
        dtype=np.uint8,
    )

    values = gray.astype(
        np.float32
    ).reshape(-1)

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
        minimum_span,
    )

    gain = (
        target_paper
        - target_ink
    ) / span

    gain = float(
        np.clip(
            gain,
            0.75,
            2.50,
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

    return normalized