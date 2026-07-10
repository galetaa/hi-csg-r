from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import src.pipeline.data_acquisition as acquisition
from src.pipeline.data_acquisition import (
    _download_huggingface,
    _download_kaggle,
    _extract_archive,
    acquisition_plan,
    dataset_registry,
    download_datasets,
    manual_check,
    select_datasets,
)


def test_registry_has_automated_and_manual_sources() -> None:
    registry = dataset_registry()
    assert registry["hwr200"]["automation"]["kind"] == "huggingface_snapshot"
    assert registry["school_notebooks"]["automation"]["kind"] == "huggingface_snapshot"
    assert registry["cyrillic_handwriting"]["automation"]["kind"] == "kaggle"
    assert registry["iam"]["automation"]["kind"] == "manual"
    assert registry["hkr_words"]["automation"]["kind"] == "manual"


def test_plan_and_download_without_execute_do_not_touch_network() -> None:
    planned = acquisition_plan(["hwr200", "iam"])
    dry_download = download_datasets(["hwr200", "iam"], execute=False)
    assert planned == dry_download
    assert all(result.status == "PLAN" for result in planned)


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        select_datasets(["not-a-dataset"])


def test_manual_check_lists_missing_archives() -> None:
    results = manual_check(["iam", "hkr_words"])
    assert {result.status for result in results} <= {"PASS", "MISSING"}
    assert {result.dataset_id for result in results} == {"iam", "hkr_words"}


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="unsafe archive member"):
        _extract_archive(archive, tmp_path / "destination")


def test_huggingface_download_uses_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(acquisition, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    dataset = dataset_registry()["hwr200"]
    _download_huggingface(dataset, force=False, workers=3)
    assert calls[0]["repo_id"] == "AntiplagiatCompany/HWR200"
    assert calls[0]["revision"] == dataset["source"]["revision"]
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["max_workers"] == 3


def test_kaggle_download_uses_versioned_dataset_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> None:
        commands.append(command)

    monkeypatch.setattr(acquisition, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(acquisition.shutil, "which", lambda _: "/usr/bin/kaggle")
    monkeypatch.setattr(acquisition.subprocess, "run", fake_run)
    dataset = dataset_registry()["cyrillic_handwriting"]
    _download_kaggle(dataset, force=False)
    assert commands[0][0:3] == ["/usr/bin/kaggle", "datasets", "download"]
    assert "constantinwerner/cyrillic-handwriting-dataset/4" in commands[0]
    assert "--unzip" in commands[0]
