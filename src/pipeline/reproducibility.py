from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.registry import REPO_ROOT, load_yaml


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str
    message: str


def _select(value: Any, selector: str) -> Any:
    """Resolve a deliberately small dot selector; ``*`` maps over a list."""
    parts = selector.split(".") if selector else []
    current = value
    for index, part in enumerate(parts):
        if part == "*":
            if not isinstance(current, list):
                raise KeyError(f"'*' requires a list at {'.'.join(parts[:index])}")
            remainder = ".".join(parts[index + 1 :])
            return [_select(item, remainder) for item in current]
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(f"cannot descend into {part!r}")
    return current


def _apply_operation(value: Any, operation: str | None) -> Any:
    if operation is None:
        return value
    if operation == "mean":
        if not isinstance(value, list) or not value:
            raise ValueError("mean requires a non-empty list")
        return statistics.fmean(float(item) for item in value)
    raise ValueError(f"unsupported operation: {operation}")


def verify_evidence(path: str | Path = "research/evidence.yaml") -> list[CheckResult]:
    registry = load_yaml(path)
    default_tolerance = float(registry.get("default_tolerance", 1e-9))
    results: list[CheckResult] = []
    cache: dict[Path, Any] = {}

    for check in registry.get("checks", []):
        check_id = str(check["id"])
        source = REPO_ROOT / str(check["source"])
        try:
            if source not in cache:
                cache[source] = json.loads(source.read_text(encoding="utf-8"))
            actual = _apply_operation(
                _select(cache[source], str(check["selector"])), check.get("operation")
            )
            expected = check["expected"]
            if isinstance(expected, float):
                tolerance = float(check.get("tolerance", default_tolerance))
                passed = abs(float(actual) - expected) <= tolerance
            else:
                passed = actual == expected
            status = "PASS" if passed else "FAIL"
            message = f"expected={expected!r}, actual={actual!r}"
        except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            status = "FAIL"
            message = str(exc)
        results.append(CheckResult(check_id, status, message))

    for gap in registry.get("known_gaps", []):
        results.append(CheckResult(str(gap["id"]), "WARN", str(gap["statement"])))
    return results


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_inventory(paths: list[str]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for relative in sorted(dict.fromkeys(paths)):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_path(path)}
        )
    return {"schema_version": 1, "algorithm": "sha256", "artifacts": artifacts}


def write_artifact_inventory(spec_path: str | Path = "research/artifacts.yaml") -> Path:
    spec = load_yaml(spec_path)
    paths = [str(path) for path in spec.get("paths", [])]
    for pattern in spec.get("globs", []):
        paths.extend(
            str(path.relative_to(REPO_ROOT))
            for path in sorted(REPO_ROOT.glob(str(pattern)))
            if path.is_file()
        )
    output = REPO_ROOT / str(spec["output"])
    output.write_text(
        json.dumps(build_artifact_inventory(paths), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def verify_artifact_inventory(
    path: str | Path = "research/artifact_inventory.json",
) -> list[CheckResult]:
    inventory_path = REPO_ROOT / path
    data = json.loads(inventory_path.read_text(encoding="utf-8"))
    results: list[CheckResult] = []
    for artifact in data.get("artifacts", []):
        relative = str(artifact["path"])
        actual_path = REPO_ROOT / relative
        if not actual_path.is_file():
            results.append(CheckResult(relative, "FAIL", "file is missing"))
            continue
        actual = sha256_path(actual_path)
        expected = str(artifact["sha256"])
        status = "PASS" if actual == expected else "FAIL"
        results.append(CheckResult(relative, status, f"sha256={actual}"))
    return results


def audit_datasets(
    path: str | Path = "research/datasets.yaml", *, required_only: bool = False
) -> list[CheckResult]:
    registry = load_yaml(path)
    results: list[CheckResult] = []
    for dataset in registry.get("datasets", []):
        if required_only and not dataset.get("required_for_final_training", False):
            continue
        dataset_id = str(dataset["id"])
        roots = [dataset.get("expected_raw_root"), dataset.get("expected_interim_root")]
        present = [str(root) for root in roots if root and (REPO_ROOT / str(root)).exists()]
        access = str(dataset.get("source", {}).get("access", "unknown"))
        if present:
            results.append(CheckResult(dataset_id, "PASS", f"present: {', '.join(present)}"))
        else:
            results.append(CheckResult(dataset_id, "MISSING", f"access={access}"))
    return results


def full_reproduction_readiness() -> list[CheckResult]:
    results = audit_datasets(required_only=True)
    checkpoint_files = [
        path
        for suffix in ("*.pt", "*.pth", "*.ckpt", "*.safetensors")
        for path in REPO_ROOT.glob(f"outputs/**/{suffix}")
    ]
    status = "PASS" if checkpoint_files else "OPTIONAL_MISSING"
    results.append(
        CheckResult(
            "model_checkpoints",
            status,
            f"found={len(checkpoint_files)}; needed for re-evaluation, not training from scratch",
        )
    )
    for relative in (
        "uv.lock",
        "outputs/htr_publication_v3/pip_freeze.txt",
        "outputs/htr_publication_v3/reproducibility_snapshot_v1.json",
    ):
        present = (REPO_ROOT / relative).is_file()
        results.append(CheckResult(relative, "PASS" if present else "MISSING", "environment"))
    return results


def _read_json(relative: str) -> dict[str, Any]:
    data = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {relative}")
    return data


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def regenerate_key_tables(
    output_dir: str | Path = "reproducibility/generated",
) -> list[Path]:
    """Regenerate compact machine-readable tables directly from frozen evidence."""
    output = REPO_ROOT / output_dir
    seed = _read_json("outputs/final_result_package_v1/seed_confirmation_summary.json")
    primary_rows = [
        {
            "seed": row["seed"],
            "baseline_cer": row["baseline_cer"],
            "plus_10k_cer": row["plus_10k_cer"],
            "delta_cer": row["delta_cer"],
        }
        for row in seed["per_seed_deltas"]
    ]
    primary_rows.append(
        {
            "seed": "mean",
            "baseline_cer": seed["model_means"][0]["mean_cer"],
            "plus_10k_cer": seed["model_means"][1]["mean_cer"],
            "delta_cer": seed["delta_summary"]["mean_delta_cer"],
        }
    )
    primary_path = output / "primary_htr_3seed.csv"
    _write_csv(
        primary_path,
        ["seed", "baseline_cer", "plus_10k_cer", "delta_cer"],
        primary_rows,
    )

    domain = _read_json("outputs/final_result_package_v1/domainwise_seed_confirmation.json")
    domain_rows = [
        {
            "domain": row["domain"],
            "mean_baseline_cer": row["mean_baseline_cer"],
            "mean_plus_10k_cer": row["mean_plus_10k_cer"],
            "mean_delta_cer": row["mean_delta_cer"],
            "improved_seeds": row["improved_cer_seeds_n"],
        }
        for row in domain["domain_summary"]
        if row["domain"] != "school_notebooks_clean"
    ]
    domain_path = output / "domainwise_htr.csv"
    _write_csv(
        domain_path,
        [
            "domain",
            "mean_baseline_cer",
            "mean_plus_10k_cer",
            "mean_delta_cer",
            "improved_seeds",
        ],
        domain_rows,
    )

    structural = _read_json("outputs/iter2_structural_gold_v1/annotation_summary.json")
    structural_rows = [
        {"criterion": key, "n": structural["overall"]["n"], "rate": value}
        for key, value in structural["overall"]["rates"].items()
    ]
    structural_path = output / "structural_gold.csv"
    _write_csv(structural_path, ["criterion", "n", "rate"], structural_rows)

    selective = _read_json(
        "outputs/htr_graph_v1/selective_iter2_confidence_v1/selective_summary.json"
    )
    methods = selective["models"]["plus_10k_context"]["risk_methods"]
    selective_rows = [
        {
            "method": method,
            "risk_auc_exact_error_all": values["risk_auc_exact_error_all"],
            "risk_auc_exact_error_school": values["risk_auc_exact_error_school"],
        }
        for method, values in methods.items()
    ]
    selective_path = output / "selective_prediction.csv"
    _write_csv(
        selective_path,
        ["method", "risk_auc_exact_error_all", "risk_auc_exact_error_school"],
        selective_rows,
    )
    return [primary_path, domain_path, structural_path, selective_path]
