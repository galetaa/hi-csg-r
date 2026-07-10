from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"sample_id", "dataset", "image_path"}


@dataclass(frozen=True)
class ManifestIssue:
    line: int
    code: str
    message: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield line_number, value


def validate_manifest(path: str | Path, *, check_images: bool = False) -> list[ManifestIssue]:
    manifest_path = Path(path)
    issues: list[ManifestIssue] = []
    seen_ids: set[str] = set()

    try:
        rows = iter_jsonl(manifest_path)
        for line_number, row in rows:
            missing = sorted(REQUIRED_FIELDS - row.keys())
            if missing:
                issues.append(
                    ManifestIssue(line_number, "required_fields_missing", ", ".join(missing))
                )

            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                issues.append(ManifestIssue(line_number, "sample_id_empty", "sample_id is empty"))
            elif sample_id in seen_ids:
                issues.append(
                    ManifestIssue(line_number, "sample_id_duplicate", f"duplicate: {sample_id}")
                )
            seen_ids.add(sample_id)

            text = row.get("text", row.get("normalized_transcription"))
            if text is None:
                issues.append(
                    ManifestIssue(
                        line_number, "text_missing", "text/normalized_transcription missing"
                    )
                )

            if check_images and row.get("image_path"):
                image_path = Path(str(row["image_path"]))
                if not image_path.is_absolute():
                    image_path = Path.cwd() / image_path
                if not image_path.exists():
                    issues.append(
                        ManifestIssue(line_number, "image_not_found", str(row["image_path"]))
                    )
    except (json.JSONDecodeError, ValueError) as exc:
        issues.append(ManifestIssue(0, "jsonl_invalid", str(exc)))

    return issues
