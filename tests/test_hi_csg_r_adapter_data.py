from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from src.htr.dataset_adapter import HICSGRAdapterDataset, collate_adapter_batch
from src.htr.vocab import CTCVocab
from src.htr.xaligned_hi_csg_r import (
    FEATURE_NAMES,
    QUALITY_FEATURE_INDICES,
    XAlignedFeatureNormalizer,
    file_sha256,
    save_feature_record,
)


def vocab() -> CTCVocab:
    return CTCVocab(
        {
            "blank_token": "<blank>",
            "blank_index": 0,
            "char_to_idx": {"a": 1},
            "idx_to_char": ["<blank>", "a"],
            "num_classes": 2,
        }
    )


def write_record(path: Path, sample_id: str, steps: int, value: float) -> None:
    features = np.full((steps, len(FEATURE_NAMES)), value, dtype=np.float32)
    features[:, 18] = 1.0
    save_feature_record(
        {
            "features": features,
            "raw_features": features,
            "quality": features[:, QUALITY_FEATURE_INDICES],
            "valid_mask": np.ones(steps, dtype=bool),
            "time_steps": steps,
            "original_width": steps * 4,
            "feature_names": np.asarray(FEATURE_NAMES),
            "quality_feature_names": np.asarray(
                [FEATURE_NAMES[index] for index in QUALITY_FEATURE_INDICES]
            ),
            "sample_id": sample_id,
            "graph_version": "hi_csg_r_v1",
            "feature_version": "hi_csg_r_xaligned_v1",
            "feature_builder_version": "1.0.0",
            "source_image_sha1": "test",
            "binarization": "test",
            "diagnostics": {},
        },
        path,
    )


def fixture_manifest(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for sample_id, width, value in (("a", 32, 1.0), ("b", 40, 3.0)):
        image_path = tmp_path / f"{sample_id}.png"
        Image.fromarray(np.full((16, width), 255, dtype=np.uint8)).save(image_path)
        feature_path = tmp_path / f"{sample_id}.npz"
        write_record(feature_path, sample_id, width // 4, value)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "text": "a",
                "dataset": "test",
                "xaligned_graph_npz": str(feature_path),
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest, tmp_path


def test_train_only_normalization_and_serialization(tmp_path: Path) -> None:
    manifest, _ = fixture_manifest(tmp_path)
    normalizer = XAlignedFeatureNormalizer.fit(manifest)
    assert normalizer.train_manifest_sha256 == file_sha256(manifest)
    assert np.isclose(normalizer.mean[0], (8 * 1.0 + 10 * 3.0) / 18)
    path = tmp_path / "normalizer.json"
    normalizer.to_path(path)
    restored = XAlignedFeatureNormalizer.from_path(path)
    assert restored == normalizer


def test_collate_pads_to_output_steps_not_image_width(tmp_path: Path) -> None:
    manifest, _ = fixture_manifest(tmp_path)
    normalizer = XAlignedFeatureNormalizer.fit(manifest)
    dataset = HICSGRAdapterDataset(manifest, vocab(), normalizer=normalizer)
    batch = collate_adapter_batch([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 1, 16, 40)
    assert batch["graph_features"].shape == (2, 10, 20)
    assert batch["graph_mask"][0].sum().item() == 8
    assert not batch["graph_mask"][0, 8:].any()


def test_shuffle_changes_only_graph_and_resamples_width(tmp_path: Path) -> None:
    manifest, _ = fixture_manifest(tmp_path)
    normalizer = XAlignedFeatureNormalizer.fit(manifest)
    plain = HICSGRAdapterDataset(manifest, vocab(), normalizer=normalizer)
    shuffled = HICSGRAdapterDataset(
        manifest,
        vocab(),
        normalizer=normalizer,
        shuffle_map={"a": "b", "b": "a"},
    )
    original = plain[0]
    changed = shuffled[0]
    assert original["sample_id"] == changed["sample_id"] == "a"
    assert original["text"] == changed["text"]
    assert changed["graph_sample_id"] == "b"
    assert changed["graph_features"].shape == original["graph_features"].shape
    assert not np.allclose(
        original["graph_features"].numpy(),
        changed["graph_features"].numpy(),
    )
