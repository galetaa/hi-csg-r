from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.pipeline.registry import REPO_ROOT


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
            "milestone_tags": _git("tag", "--list", "milestone/*").splitlines(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "canonical_files": {
            "claims": file_sha256(REPO_ROOT / "research/claims.yaml"),
            "pipeline": file_sha256(REPO_ROOT / "research/pipeline.yaml"),
            "datasets": file_sha256(REPO_ROOT / "research/datasets.yaml"),
            "evidence": file_sha256(REPO_ROOT / "research/evidence.yaml"),
            "artifact_inventory": file_sha256(REPO_ROOT / "research/artifact_inventory.json"),
            "environment_profiles": file_sha256(REPO_ROOT / "research/environment_profiles.yaml"),
            "manuscript": file_sha256(REPO_ROOT / "article/HI_CSG_R_v11.docx"),
        },
    }


def write_provenance(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(collect_provenance(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
