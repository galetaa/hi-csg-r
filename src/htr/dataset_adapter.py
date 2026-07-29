from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from src.htr.vocab import CTCVocab
from src.htr.xaligned_hi_csg_r import (
    FEATURE_NAMES,
    QUALITY_FEATURE_NAMES,
    XAlignedFeatureNormalizer,
    compute_output_steps,
    load_feature_record,
    read_jsonl,
    resample_feature_sequence,
    resolve_path,
)
from torch.utils.data import Dataset


def load_shuffle_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "mapping" in data:
        data = data["mapping"]
    if not isinstance(data, dict):
        raise ValueError("Shuffle map must be a JSON object or contain a mapping object")
    return {str(key): str(value) for key, value in data.items()}


class HICSGRAdapterDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        vocab: CTCVocab,
        *,
        normalizer: XAlignedFeatureNormalizer | None = None,
        mode: str = "m3_full",
        feature_field: str = "xaligned_graph_npz",
        shuffle_map: str | Path | dict[str, str] | None = None,
        strict_feature_version: str | None = "hi_csg_r_xaligned_v1",
    ) -> None:
        if mode not in {"m0_ft", "m2_geometry", "m3_full"}:
            raise ValueError(f"Unsupported adapter dataset mode: {mode}")
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        self.vocab = vocab
        self.normalizer = normalizer
        self.mode = mode
        self.feature_field = feature_field
        self.strict_feature_version = strict_feature_version
        if isinstance(shuffle_map, dict):
            self.shuffle_map = {str(key): str(value) for key, value in shuffle_map.items()}
        else:
            self.shuffle_map = load_shuffle_map(shuffle_map)
        self.row_by_sample_id = {str(row["sample_id"]): row for row in self.rows}
        if len(self.row_by_sample_id) != len(self.rows):
            raise ValueError(f"Duplicate sample_id values in {self.manifest_path}")
        if self.shuffle_map:
            row_ids = set(self.row_by_sample_id)
            missing = set(self.shuffle_map) - row_ids
            unmapped = row_ids - set(self.shuffle_map)
            donors = set(self.shuffle_map.values()) - set(self.row_by_sample_id)
            self_pairs = {
                target for target, donor in self.shuffle_map.items() if target == donor
            }
            if missing or unmapped or donors or self_pairs:
                raise ValueError(
                    f"Shuffle map does not match manifest: missing targets={len(missing)}, "
                    f"unmapped targets={len(unmapped)}, missing donors={len(donors)}, "
                    f"self pairs={len(self_pairs)}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _feature_row(self, row: dict[str, Any]) -> tuple[dict[str, Any], str]:
        sample_id = str(row["sample_id"])
        donor_id = self.shuffle_map.get(sample_id, sample_id)
        return self.row_by_sample_id[donor_id], donor_id

    def _load_graph_features(
        self,
        row: dict[str, Any],
        *,
        expected_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        if self.mode == "m0_ft":
            raw_features = np.zeros((expected_steps, len(FEATURE_NAMES)), dtype=np.float32)
            quality = np.zeros((expected_steps, len(QUALITY_FEATURE_NAMES)), dtype=np.float32)
            return (
                raw_features.copy(),
                raw_features,
                quality,
                np.ones(expected_steps, dtype=bool),
                str(row["sample_id"]),
            )

        if self.normalizer is None:
            raise ValueError(f"Normalizer is required for mode={self.mode}")
        feature_row, donor_id = self._feature_row(row)
        feature_value = feature_row.get(self.feature_field)
        if not feature_value:
            raise KeyError(f"{self.feature_field} missing for donor sample {donor_id}")
        feature_path = resolve_path(str(feature_value), self.manifest_path)
        record = load_feature_record(feature_path)

        if str(record["sample_id"]) != donor_id:
            raise ValueError(
                f"Feature record sample mismatch: manifest donor={donor_id}, "
                f"record={record['sample_id']}"
            )
        if self.strict_feature_version and str(record["feature_version"]) != self.strict_feature_version:
            raise ValueError(
                f"Feature version mismatch for {donor_id}: {record['feature_version']}"
            )
        names = tuple(str(value) for value in record["feature_names"].tolist())
        if names != FEATURE_NAMES:
            raise ValueError(f"Feature order mismatch for {donor_id}")

        raw_features = np.asarray(record["features"], dtype=np.float32)
        valid_mask = np.asarray(record["valid_mask"], dtype=bool)
        if raw_features.ndim != 2 or raw_features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Invalid x-aligned feature shape for donor {donor_id}: {raw_features.shape}"
            )
        if valid_mask.shape != (raw_features.shape[0],):
            raise ValueError(f"Invalid feature mask shape for {donor_id}: {valid_mask.shape}")
        if raw_features.shape[0] != expected_steps:
            if donor_id == str(row["sample_id"]):
                raise ValueError(
                    f"x-aligned shape mismatch for {donor_id}: {raw_features.shape[0]} "
                    f"vs expected {expected_steps}"
                )
            raw_features, valid_mask = resample_feature_sequence(
                raw_features,
                expected_steps,
                source_mask=valid_mask,
            )
        topology_enabled = self.mode == "m3_full"
        features, quality = self.normalizer.transform(
            raw_features,
            topology_enabled=topology_enabled,
        )
        features[~valid_mask] = 0.0
        quality[~valid_mask] = 0.0
        raw_features[~valid_mask] = 0.0
        return features, raw_features, quality, valid_mask, donor_id

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = resolve_path(str(row["image_path"]), self.manifest_path)
        with Image.open(image_path) as image_file:
            gray = image_file.convert("L")
            array = np.asarray(gray, dtype=np.float32)
        image = torch.from_numpy((255.0 - array) / 255.0).unsqueeze(0)
        width = int(image.shape[-1])
        output_steps = compute_output_steps(width)
        (
            graph_features,
            graph_raw_features,
            graph_quality,
            graph_mask,
            donor_id,
        ) = self._load_graph_features(row, expected_steps=output_steps)

        text = str(row["text"])
        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return {
            "image": image,
            "target": target,
            "text": text,
            "width": width,
            "output_steps": output_steps,
            "sample_id": str(row["sample_id"]),
            "graph_sample_id": donor_id,
            "dataset": str(row.get("dataset") or row.get("source_dataset") or "unknown"),
            "level": row.get("level"),
            "category": row.get("category"),
            "graph_features": torch.from_numpy(graph_features),
            "graph_raw_features": torch.from_numpy(graph_raw_features),
            "graph_quality": torch.from_numpy(graph_quality),
            "graph_mask": torch.from_numpy(graph_mask),
        }


def collate_adapter_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_height = max(int(item["image"].shape[1]) for item in batch)
    max_width = max(int(item["image"].shape[2]) for item in batch)
    max_steps = max(int(item["output_steps"]) for item in batch)

    images = torch.zeros((len(batch), 1, max_height, max_width), dtype=torch.float32)
    graph_features = torch.zeros(
        (len(batch), max_steps, len(FEATURE_NAMES)),
        dtype=torch.float32,
    )
    graph_raw_features = torch.zeros_like(graph_features)
    graph_quality = torch.zeros(
        (len(batch), max_steps, len(QUALITY_FEATURE_NAMES)),
        dtype=torch.float32,
    )
    graph_mask = torch.zeros((len(batch), max_steps), dtype=torch.bool)

    targets: list[torch.Tensor] = []
    widths: list[int] = []
    output_steps: list[int] = []
    target_lengths: list[int] = []

    for batch_index, item in enumerate(batch):
        image = item["image"]
        _, height, width = image.shape
        steps = int(item["output_steps"])
        if steps != compute_output_steps(int(width)):
            raise ValueError("Dataset output_steps does not match image width")
        if item["graph_features"].shape != (steps, len(FEATURE_NAMES)):
            raise ValueError("Graph feature shape does not match sample output length")
        images[batch_index, :, :height, :width] = image
        graph_features[batch_index, :steps] = item["graph_features"]
        graph_raw_features[batch_index, :steps] = item["graph_raw_features"]
        graph_quality[batch_index, :steps] = item["graph_quality"]
        graph_mask[batch_index, :steps] = item["graph_mask"]
        targets.append(item["target"])
        widths.append(int(width))
        output_steps.append(steps)
        target_lengths.append(int(len(item["target"])))

    return {
        "images": images,
        "widths": torch.tensor(widths, dtype=torch.long),
        "output_steps": torch.tensor(output_steps, dtype=torch.long),
        "graph_features": graph_features,
        "graph_raw_features": graph_raw_features,
        "graph_quality": graph_quality,
        "graph_mask": graph_mask,
        "targets": torch.cat(targets, dim=0),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": [str(item["text"]) for item in batch],
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "graph_sample_ids": [str(item["graph_sample_id"]) for item in batch],
        "datasets": [str(item["dataset"]) for item in batch],
        "levels": [item.get("level") for item in batch],
        "categories": [item.get("category") for item in batch],
    }
