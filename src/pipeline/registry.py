from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_STATUSES = {
    "supported",
    "supported_within_audited_subset",
    "partially_supported",
    "exploratory",
    "not_supported",
    "false_by_design",
    "unresolved",
}


@dataclass(frozen=True)
class RegistryIssue:
    code: str
    message: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    return data


def validate_claim_registry(
    path: str | Path = "research/claims.yaml",
    *,
    check_evidence: bool = True,
) -> list[RegistryIssue]:
    data = load_yaml(path)
    claims = data.get("claims")
    issues: list[RegistryIssue] = []

    if not isinstance(claims, list) or not claims:
        return [RegistryIssue("claims_missing", "Registry must contain a non-empty claims list")]

    seen: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            issues.append(RegistryIssue("claim_not_mapping", f"Claim #{index} is not a mapping"))
            continue

        claim_id = str(claim.get("id", "")).strip()
        if not claim_id:
            issues.append(RegistryIssue("claim_id_missing", f"Claim #{index} has no id"))
        elif claim_id in seen:
            issues.append(RegistryIssue("claim_id_duplicate", f"Duplicate claim id: {claim_id}"))
        seen.add(claim_id)

        status = claim.get("status")
        if status not in ALLOWED_STATUSES:
            issues.append(
                RegistryIssue("claim_status_invalid", f"{claim_id}: invalid status {status!r}")
            )

        if not str(claim.get("statement", "")).strip():
            issues.append(
                RegistryIssue("claim_statement_missing", f"{claim_id}: statement is empty")
            )

        evidence = claim.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            issues.append(
                RegistryIssue("claim_evidence_missing", f"{claim_id}: no evidence entries")
            )
            continue

        for item in evidence:
            evidence_path = item.get("path") if isinstance(item, dict) else None
            if not evidence_path:
                issues.append(
                    RegistryIssue("evidence_path_missing", f"{claim_id}: malformed evidence entry")
                )
                continue
            if check_evidence and not (REPO_ROOT / str(evidence_path)).exists():
                issues.append(
                    RegistryIssue("evidence_not_found", f"{claim_id}: missing {evidence_path}")
                )

    return issues


def validate_pipeline_registry(
    path: str | Path = "research/pipeline.yaml",
) -> list[RegistryIssue]:
    data = load_yaml(path)
    stages = data.get("stages")
    issues: list[RegistryIssue] = []
    if not isinstance(stages, list) or not stages:
        return [RegistryIssue("pipeline_stages_missing", "Pipeline must contain stages")]

    seen: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            issues.append(RegistryIssue("pipeline_stage_invalid", "Stage is not a mapping"))
            continue
        stage_id = str(stage.get("id", "")).strip()
        if not stage_id:
            issues.append(RegistryIssue("pipeline_stage_id_missing", "Stage id is empty"))
        elif stage_id in seen:
            issues.append(RegistryIssue("pipeline_stage_id_duplicate", stage_id))
        seen.add(stage_id)

        implementation = stage.get("implementation")
        if not implementation or not (REPO_ROOT / str(implementation)).exists():
            issues.append(
                RegistryIssue(
                    "pipeline_implementation_missing",
                    f"{stage_id}: {implementation}",
                )
            )

    legacy_root = data.get("legacy_scripts", {}).get("root")
    if not legacy_root or not (REPO_ROOT / str(legacy_root)).is_dir():
        issues.append(RegistryIssue("legacy_root_missing", str(legacy_root)))
    return issues
