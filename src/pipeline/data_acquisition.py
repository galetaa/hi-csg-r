from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.pipeline.registry import REPO_ROOT, load_yaml


@dataclass(frozen=True)
class AcquisitionResult:
    dataset_id: str
    status: str
    message: str


def dataset_registry() -> dict[str, dict[str, Any]]:
    items = load_yaml("research/datasets.yaml").get("datasets", [])
    return {str(item["id"]): item for item in items}


def select_datasets(ids: list[str]) -> list[dict[str, Any]]:
    registry = dataset_registry()
    selected_ids = ids or list(registry)
    unknown = sorted(set(selected_ids) - registry.keys())
    if unknown:
        raise ValueError(f"unknown dataset ids: {', '.join(unknown)}")
    return [registry[dataset_id] for dataset_id in selected_ids]


def acquisition_plan(ids: list[str]) -> list[AcquisitionResult]:
    results: list[AcquisitionResult] = []
    for dataset in select_datasets(ids):
        automation = dataset.get("automation", {})
        root = REPO_ROOT / str(dataset["expected_raw_root"])
        kind = str(automation.get("kind", "manual"))
        size = automation.get("approximate_download_size", "unknown")
        state = "present" if root.exists() else "absent"
        results.append(
            AcquisitionResult(
                str(dataset["id"]),
                "PLAN",
                f"kind={kind} raw={root.relative_to(REPO_ROOT)} state={state} size≈{size}",
            )
        )
    return results


def _ensure_download_target(root: Path, *, force: bool) -> None:
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"destination is not empty: {root}; use --force to resume/overwrite")
    root.mkdir(parents=True, exist_ok=True)


def _download_huggingface(dataset: dict[str, Any], *, force: bool, workers: int) -> str:
    from huggingface_hub import snapshot_download

    root = REPO_ROOT / str(dataset["expected_raw_root"])
    _ensure_download_target(root, force=force)
    automation = dataset["automation"]
    source = dataset["source"]
    return snapshot_download(
        repo_id=str(automation["repo_id"]),
        repo_type="dataset",
        revision=str(source["revision"]),
        local_dir=root,
        max_workers=workers,
        force_download=force,
    )


def _download_kaggle(dataset: dict[str, Any], *, force: bool) -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "Kaggle CLI not found; run `uv sync --group data-download` and retry via `uv run`"
        )
    root = REPO_ROOT / str(dataset["expected_raw_root"])
    _ensure_download_target(root, force=force)
    command = [
        executable,
        "datasets",
        "download",
        str(dataset["automation"]["dataset_ref"]),
        "-p",
        str(root),
        "--unzip",
    ]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return str(root)


def download_datasets(
    ids: list[str], *, execute: bool, force: bool = False, workers: int = 4
) -> list[AcquisitionResult]:
    if not execute:
        return acquisition_plan(ids)
    results: list[AcquisitionResult] = []
    for dataset in select_datasets(ids):
        dataset_id = str(dataset["id"])
        kind = str(dataset.get("automation", {}).get("kind", "manual"))
        try:
            if kind == "huggingface_snapshot":
                location = _download_huggingface(dataset, force=force, workers=workers)
                status = "DOWNLOADED"
                message = location
            elif kind == "kaggle":
                location = _download_kaggle(dataset, force=force)
                status = "DOWNLOADED"
                message = location
            else:
                status = "MANUAL"
                message = str(dataset["source"]["url"])
        except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
            status = "ERROR"
            message = str(exc)
        results.append(AcquisitionResult(dataset_id, status, message))
    return results


def manual_check(ids: list[str]) -> list[AcquisitionResult]:
    results: list[AcquisitionResult] = []
    for dataset in select_datasets(ids):
        expected = dataset.get("automation", {}).get("expected_archives", [])
        if not expected:
            continue
        root = REPO_ROOT / str(dataset["expected_raw_root"])
        missing = [name for name in expected if not (root / str(name)).is_file()]
        status = "PASS" if not missing else "MISSING"
        message = "all expected archives present" if not missing else ", ".join(missing)
        results.append(AcquisitionResult(str(dataset["id"]), status, message))
    return results


def _safe_target(base: Path, member_name: str) -> Path:
    target = (base / member_name).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ValueError(f"unsafe archive member: {member_name}")
    return target


def _extract_archive(archive: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                _safe_target(destination, member.filename)
            stream.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as stream:
            for tar_member in stream.getmembers():
                _safe_target(destination, tar_member.name)
                if tar_member.issym() or tar_member.islnk():
                    raise ValueError(f"archive links are not allowed: {tar_member.name}")
                if not tar_member.isfile() and not tar_member.isdir():
                    raise ValueError(f"special archive member is not allowed: {tar_member.name}")
            stream.extractall(destination)
        return
    raise ValueError(f"unsupported archive: {archive}")


def extract_datasets(ids: list[str], *, execute: bool) -> list[AcquisitionResult]:
    results: list[AcquisitionResult] = []
    patterns = ("*.zip", "*.tgz", "*.tar", "*.tar.gz")
    for dataset in select_datasets(ids):
        dataset_id = str(dataset["id"])
        raw = REPO_ROOT / str(dataset["expected_raw_root"])
        destination = REPO_ROOT / str(dataset["expected_interim_root"])
        archives = (
            sorted({path for pattern in patterns for path in raw.rglob(pattern)})
            if raw.exists()
            else []
        )
        if not archives:
            results.append(AcquisitionResult(dataset_id, "SKIP", "no supported archives found"))
            continue
        if not execute:
            results.append(
                AcquisitionResult(
                    dataset_id,
                    "PLAN",
                    f"extract {len(archives)} archive(s) to {destination.relative_to(REPO_ROOT)}",
                )
            )
            continue
        destination.mkdir(parents=True, exist_ok=True)
        try:
            for archive in archives:
                _extract_archive(archive, destination)
            results.append(AcquisitionResult(dataset_id, "EXTRACTED", f"archives={len(archives)}"))
        except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
            results.append(AcquisitionResult(dataset_id, "ERROR", str(exc)))
    return results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_local_checksums(dataset_id: str) -> Path:
    dataset = dataset_registry()[dataset_id]
    root = REPO_ROOT / str(dataset["expected_raw_root"])
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = [
        path for path in sorted(root.rglob("*")) if path.is_file() and ".cache" not in path.parts
    ]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset_id,
        "source": dataset["source"],
        "root": str(root.relative_to(REPO_ROOT)),
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    output = REPO_ROOT / "data/local_provenance" / f"{dataset_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
