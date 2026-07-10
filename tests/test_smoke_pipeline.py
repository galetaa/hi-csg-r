from __future__ import annotations

import numpy as np
from src.graph.pixel_graph import build_pixel_graph
from src.graph.skeletonize import skeletonize_mask
from src.pipeline.manifest import validate_manifest
from src.pipeline.registry import REPO_ROOT
from src.preprocessing.image_io import load_image_as_grayscale


def test_synthetic_smoke_manifest_and_graph() -> None:
    manifest = REPO_ROOT / "reproducibility/smoke/manifest.jsonl"
    assert validate_manifest(manifest, check_images=True) == []

    image = load_image_as_grayscale(REPO_ROOT / "reproducibility/smoke/stroke.pgm")
    mask = np.asarray(image) < 128
    skeleton = skeletonize_mask(mask)
    graph = build_pixel_graph(skeleton.skeleton, binary_mask=mask)

    assert image.size == (12, 9)
    assert skeleton.skeleton_pixels == 8
    assert graph.features["component_count"] == 1
    assert graph.features["endpoint_count"] == 2
