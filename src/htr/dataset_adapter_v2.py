from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from src.htr.dataset_adapter import load_shuffle_map
from src.htr.model_hi_csg_r_late_correction_v2 import RISK_FEATURE_INDICES
from src.htr.vocab import CTCVocab
from src.htr.xaligned_hi_csg_r import (
    FEATURE_NAMES,
    XAlignedFeatureNormalizer,
    compute_output_steps,
    load_feature_record,
    read_jsonl,
    resample_feature_sequence,
    resolve_path,
)
from torch.utils.data import Dataset, Sampler


def core_domain(value: str) -> str:
    name = str(value).lower()
    if "cyrillic" in name:
        return "cyrillic"
    if "hkr" in name:
        return "hkr"
    if "school" in name:
        return "school"
    return str(value)


class HICSGRLateCorrectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        vocab: CTCVocab,
        normalizer: XAlignedFeatureNormalizer,
        *,
        feature_field: str = "xaligned_graph_npz",
        shuffle_map: str | Path | dict[str, str] | None = None,
        strict_feature_version: str = "hi_csg_r_xaligned_v1",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        self.vocab = vocab
        self.normalizer = normalizer
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
            expected = set(self.row_by_sample_id)
            if set(self.shuffle_map) != expected:
                raise ValueError("Shuffle map targets do not exactly match manifest")
            if set(self.shuffle_map.values()) - expected:
                raise ValueError("Shuffle map contains donors outside the manifest")
            if any(target == donor for target, donor in self.shuffle_map.items()):
                raise ValueError("Shuffle map contains self-pairs")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_features(
        self,
        row: dict[str, Any],
        expected_steps: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
    ]:
        sample_id = str(row["sample_id"])
        donor_id = self.shuffle_map.get(sample_id, sample_id)
        donor = self.row_by_sample_id[donor_id]
        value = donor.get(self.feature_field)
        if not value:
            raise KeyError(f"{self.feature_field} missing for {donor_id}")
        record = load_feature_record(resolve_path(str(value), self.manifest_path))
        if str(record["sample_id"]) != donor_id:
            raise ValueError(f"Feature record mismatch for donor {donor_id}")
        if str(record["feature_version"]) != self.strict_feature_version:
            raise ValueError(f"Feature version mismatch for donor {donor_id}")
        names = tuple(str(name) for name in record["feature_names"].tolist())
        if names != FEATURE_NAMES:
            raise ValueError(f"Feature order mismatch for donor {donor_id}")
        raw = np.asarray(record["features"], dtype=np.float32)
        valid = np.asarray(record["valid_mask"], dtype=bool)
        if raw.shape[0] != expected_steps:
            if donor_id == sample_id:
                raise ValueError(
                    f"Feature/output length mismatch for {sample_id}: "
                    f"{raw.shape[0]} vs {expected_steps}"
                )
            raw, valid = resample_feature_sequence(
                raw,
                expected_steps,
                source_mask=valid,
            )
        normalized, _ = self.normalizer.transform(raw, topology_enabled=True)
        nonempty = (
            (raw[:, 0] > 0)
            | (raw[:, 1] > 0)
            | (raw[:, 2] > 0)
            | (raw[:, 18] > 0)
        ) & valid
        # This post-normalization mask is the central v2 regression fix.
        normalized = normalized * nonempty[:, None].astype(np.float32)
        normalized[~valid] = 0.0
        raw[~valid] = 0.0
        risk = raw[:, RISK_FEATURE_INDICES].astype(np.float32)
        risk[~valid] = 0.0
        return normalized, raw, risk, valid, nonempty, donor_id

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = resolve_path(str(row["image_path"]), self.manifest_path)
        with Image.open(image_path) as image_file:
            gray = image_file.convert("L")
            array = np.asarray(gray, dtype=np.float32)
        image = torch.from_numpy((255.0 - array) / 255.0).unsqueeze(0)
        width = int(image.shape[-1])
        output_steps = compute_output_steps(width)
        normalized, raw, risk, valid, nonempty, donor_id = self._load_features(
            row,
            output_steps,
        )
        text = str(row["text"])
        return {
            "image": image,
            "width": width,
            "output_steps": output_steps,
            "target": torch.tensor(self.vocab.encode(text), dtype=torch.long),
            "text": text,
            "sample_id": str(row["sample_id"]),
            "graph_sample_id": donor_id,
            "dataset": str(row.get("dataset") or row.get("source_dataset") or "unknown"),
            "core_domain": core_domain(
                str(row.get("dataset") or row.get("source_dataset") or "unknown")
            ),
            "level": row.get("level"),
            "category": row.get("category"),
            "raw_graph_features": torch.from_numpy(raw),
            "normalized_graph_features": torch.from_numpy(normalized),
            "structural_risk_raw": torch.from_numpy(risk),
            "time_mask": torch.from_numpy(valid),
            "nonempty_graph_mask": torch.from_numpy(nonempty),
        }


def collate_late_correction_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(batch)
    max_height = max(int(item["image"].shape[1]) for item in batch)
    max_width = max(int(item["image"].shape[2]) for item in batch)
    max_steps = max(int(item["output_steps"]) for item in batch)
    images = torch.zeros((batch_size, 1, max_height, max_width), dtype=torch.float32)
    raw = torch.zeros((batch_size, max_steps, len(FEATURE_NAMES)), dtype=torch.float32)
    normalized = torch.zeros_like(raw)
    risk = torch.zeros((batch_size, max_steps, 3), dtype=torch.float32)
    time_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    nonempty = torch.zeros_like(time_mask)
    targets: list[torch.Tensor] = []
    widths: list[int] = []
    target_lengths: list[int] = []
    output_steps: list[int] = []

    for index, item in enumerate(batch):
        image = item["image"]
        _, height, width = image.shape
        steps = int(item["output_steps"])
        if steps != compute_output_steps(int(width)):
            raise ValueError("output_steps does not match image width")
        if item["normalized_graph_features"].shape != (steps, len(FEATURE_NAMES)):
            raise ValueError("Normalized feature shape does not match output length")
        images[index, :, :height, :width] = image
        raw[index, :steps] = item["raw_graph_features"]
        normalized[index, :steps] = item["normalized_graph_features"]
        risk[index, :steps] = item["structural_risk_raw"]
        time_mask[index, :steps] = item["time_mask"]
        nonempty[index, :steps] = item["nonempty_graph_mask"]
        targets.append(item["target"])
        widths.append(int(width))
        output_steps.append(steps)
        target_lengths.append(int(len(item["target"])))

    if torch.count_nonzero(normalized[time_mask & ~nonempty]).item() != 0:
        raise ValueError("Post-normalization empty-bin invariant was violated")
    padding_mask = ~time_mask
    return {
        "images": images,
        "widths": torch.tensor(widths, dtype=torch.long),
        "output_steps": torch.tensor(output_steps, dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "raw_graph_features": raw,
        "normalized_graph_features": normalized,
        "graph_features": normalized,
        "structural_risk_raw": risk,
        "time_mask": time_mask,
        "nonempty_graph_mask": nonempty,
        "padding_mask": padding_mask,
        "texts": [str(item["text"]) for item in batch],
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "graph_sample_ids": [str(item["graph_sample_id"]) for item in batch],
        "datasets": [str(item["dataset"]) for item in batch],
        "core_domains": [str(item["core_domain"]) for item in batch],
        "levels": [item.get("level") for item in batch],
        "categories": [item.get("category") for item in batch],
    }


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Deterministic approximately 1/3-per-domain batches."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        batch_size: int,
        *,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if batch_size < 3:
            raise ValueError("Domain-balanced batches require batch_size >= 3")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.by_domain: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            domain = core_domain(
                str(row.get("dataset") or row.get("source_dataset") or "unknown")
            )
            self.by_domain.setdefault(domain, []).append(index)
        if set(self.by_domain) != {"cyrillic", "hkr", "school"}:
            raise ValueError(
                "Domain-balanced sampler requires cyrillic, hkr, and school rows; "
                f"got {sorted(self.by_domain)}"
            )
        self.batch_count = (len(rows) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        domains = ("cyrillic", "hkr", "school")
        pools = {name: list(self.by_domain[name]) for name in domains}
        positions = {name: 0 for name in domains}
        if self.shuffle:
            for pool in pools.values():
                rng.shuffle(pool)

        def next_index(domain: str) -> int:
            position = positions[domain]
            if position >= len(pools[domain]):
                positions[domain] = 0
                position = 0
                if self.shuffle:
                    rng.shuffle(pools[domain])
            positions[domain] += 1
            return pools[domain][position]

        for batch_index in range(self.batch_count):
            base = self.batch_size // 3
            remainder = self.batch_size % 3
            counts = {domain: base for domain in domains}
            for offset in range(remainder):
                counts[domains[(batch_index + offset) % 3]] += 1
            batch = [
                next_index(domain)
                for domain in domains
                for _ in range(counts[domain])
            ]
            if self.shuffle:
                rng.shuffle(batch)
            yield batch
