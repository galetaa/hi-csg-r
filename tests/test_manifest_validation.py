import json
from pathlib import Path

from src.pipeline.manifest import sha256_file, validate_manifest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_valid_manifest_passes(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        path,
        [{"sample_id": "s1", "dataset": "demo", "image_path": "x.png", "text": "тест"}],
    )
    assert validate_manifest(path) == []
    assert len(sha256_file(path)) == 64


def test_duplicate_ids_and_missing_text_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        path,
        [
            {"sample_id": "s1", "dataset": "demo", "image_path": "a.png", "text": "а"},
            {"sample_id": "s1", "dataset": "demo", "image_path": "b.png"},
        ],
    )
    codes = {issue.code for issue in validate_manifest(path)}
    assert codes == {"sample_id_duplicate", "text_missing"}
