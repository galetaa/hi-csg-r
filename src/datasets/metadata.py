from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
import json


@dataclass
class SampleMetadata:
    sample_id: str
    dataset: str
    source_id: str | None
    language: str
    script: str
    level: str

    image_path: str
    raw_transcription: str
    normalized_transcription: str

    writer_id: str | None = None
    ocr_image_path: str | None = None
    feature_image_path: str | None = None
    bbox: list[float] | None = None
    polygon: list[list[float]] | None = None
    split: str | None = None

    transcription_modes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records