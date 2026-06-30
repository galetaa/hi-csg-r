from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_cmd(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "cmd": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def optional_module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    return {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_names": [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def build_md(snapshot: dict[str, Any]) -> str:
    git = snapshot["git"]
    env = snapshot["environment"]
    dirty = bool(git["status_short"]["stdout"])
    lines = [
        "# Publication Reproducibility Snapshot v1",
        "",
        "## Repository",
        "",
        f"- commit: `{git['head']['stdout'] or 'unknown'}`",
        f"- dirty working tree: {'yes' if dirty else 'no'}",
        f"- branch: `{git['branch']['stdout'] or 'unknown'}`",
        "",
        "## Environment",
        "",
        f"- python: `{env['python_version']}`",
        f"- executable: `{env['python_executable']}`",
        f"- platform: `{env['platform']}`",
        f"- torch: `{env['torch'].get('version', 'not available')}`",
        f"- torch CUDA available: `{env['torch'].get('cuda_available')}`",
        f"- torch CUDA version: `{env['torch'].get('cuda_version')}`",
        f"- CUDA devices: `{env['torch'].get('device_names', [])}`",
        f"- transformers: `{env['transformers_version']}`",
        f"- numpy: `{env['numpy_version']}`",
        "",
        "## Git Status",
        "",
        "```text",
        git["status_short"]["stdout"],
        "```",
        "",
        "## Reproducibility Caution",
        "",
        "A dirty working tree means the snapshot is not a clean immutable release state. For publication, commit or archive the exact code and generated manifests used for the reported runs.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/htr_publication_v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pip_freeze = run_cmd([sys.executable, "-m", "pip", "freeze"])
    (out_dir / "pip_freeze.txt").write_text(
        pip_freeze["stdout"] + ("\n" if pip_freeze["stdout"] else ""),
        encoding="utf-8",
    )

    snapshot = {
        "git": {
            "head": run_cmd(["git", "rev-parse", "HEAD"]),
            "branch": run_cmd(["git", "branch", "--show-current"]),
            "status_short": run_cmd(["git", "status", "--short"]),
            "diff_stat": run_cmd(["git", "diff", "--stat"]),
        },
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch_info(),
            "transformers_version": optional_module_version("transformers"),
            "numpy_version": optional_module_version("numpy"),
        },
        "pip_freeze": {
            "path": str(out_dir / "pip_freeze.txt"),
            "returncode": pip_freeze["returncode"],
            "stderr": pip_freeze["stderr"],
        },
    }

    (out_dir / "reproducibility_snapshot_v1.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "reproducibility_snapshot_v1.md").write_text(
        build_md(snapshot),
        encoding="utf-8",
    )

    print(json.dumps({
        "out_json": str(out_dir / "reproducibility_snapshot_v1.json"),
        "out_md": str(out_dir / "reproducibility_snapshot_v1.md"),
        "pip_freeze": str(out_dir / "pip_freeze.txt"),
        "dirty": bool(snapshot["git"]["status_short"]["stdout"]),
        "commit": snapshot["git"]["head"]["stdout"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
