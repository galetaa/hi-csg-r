from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.htr.vocab import CTCVocab


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class HTRDataset(Dataset):
    def __init__(self, manifest_path: str | Path, vocab: CTCVocab) -> None:
        self.rows = read_jsonl(manifest_path)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]

        img = Image.open(row["image_path"]).convert("L")
        arr = np.asarray(img, dtype=np.float32)

        # Foreground-high normalization: white background -> 0, dark ink -> high.
        arr = (255.0 - arr) / 255.0

        image = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

        text = str(row["text"])
        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)

        return {
            "image": image,
            "target": target,
            "text": text,
            "width": int(image.shape[-1]),
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "level": row.get("level"),
            "category": row.get("category"),
        }


def collate_htr_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    images = []
    widths = []
    targets = []
    target_lengths = []

    texts = []
    sample_ids = []
    datasets = []
    levels = []
    categories = []

    for item in batch:
        img = item["image"]
        _, h, w = img.shape

        padded = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        padded[:, :h, :w] = img

        images.append(padded)
        widths.append(w)
        targets.append(item["target"])
        target_lengths.append(len(item["target"]))

        texts.append(item["text"])
        sample_ids.append(item["sample_id"])
        datasets.append(item["dataset"])
        levels.append(item.get("level"))
        categories.append(item.get("category"))

    return {
        "images": torch.stack(images, dim=0),
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.cat(targets, dim=0),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
        "sample_ids": sample_ids,
        "datasets": datasets,
        "levels": levels,
        "categories": categories,
    }