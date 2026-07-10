from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.pipeline.manifest import sha256_file, validate_manifest
from src.pipeline.provenance import write_provenance
from src.pipeline.registry import (
    REPO_ROOT,
    load_yaml,
    validate_claim_registry,
    validate_pipeline_registry,
)
from src.pipeline.reproducibility import (
    CheckResult,
    audit_datasets,
    full_reproduction_readiness,
    regenerate_key_tables,
    verify_artifact_inventory,
    verify_evidence,
    write_artifact_inventory,
)


def _verify_milestones() -> list[str]:
    import subprocess

    issues: list[str] = []
    data = load_yaml("research/milestones.yaml")
    for item in data.get("milestones", []):
        tag = item["tag"]
        expected = item["commit"]
        result = subprocess.run(
            ["git", "rev-list", "-n", "1", tag],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        actual = result.stdout.strip()
        if result.returncode != 0 or not actual.startswith(expected):
            issues.append(f"milestone {tag}: expected {expected}, got {actual or 'missing'}")
    return issues


def verify() -> int:
    issues = [f"claims: {issue.code}: {issue.message}" for issue in validate_claim_registry()]
    issues.extend(
        f"pipeline: {issue.code}: {issue.message}" for issue in validate_pipeline_registry()
    )
    issues.extend(_verify_milestones())

    claims = load_yaml("research/claims.yaml")
    manuscript = claims["canonical_manuscript"]
    manuscript_path = REPO_ROOT / manuscript["path"]
    if not manuscript_path.exists():
        issues.append(f"manuscript missing: {manuscript['path']}")
    elif sha256_file(manuscript_path) != manuscript["sha256"]:
        issues.append("canonical manuscript SHA-256 mismatch")

    if issues:
        for issue in issues:
            print(f"FAIL {issue}")
        return 1

    print("PASS canonical manuscript, claims, evidence paths, and milestone tags are valid")
    return 0


def show_status() -> int:
    claims = load_yaml("research/claims.yaml")
    print("HI-CSG-R frozen research status")
    print(f"manuscript: {claims['canonical_manuscript']['path']}")
    for claim in claims["claims"]:
        print(f"{claim['id']}: {claim['status']} ({claim['level']})")
    return 0


def _print_results(results: Sequence[CheckResult]) -> tuple[int, int, int]:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "MISSING": 0, "OPTIONAL_MISSING": 0}
    for result in results:
        status = result.status
        counts[status] = counts.get(status, 0) + 1
        print(f"{status} {result.check_id}: {result.message}")
    return counts["PASS"], counts["WARN"], counts["FAIL"] + counts["MISSING"]


def reproduce_lite() -> int:
    results = [*verify_evidence(), *verify_artifact_inventory()]
    passed, warnings, failures = _print_results(results)
    print(f"SUMMARY pass={passed} warn={warnings} fail={failures}")
    return 1 if failures else 0


def audit_data() -> int:
    results = audit_datasets()
    _print_results(results)
    # Absence is an audit result, not a broken verifier.
    return 0


def reproduce_full_check() -> int:
    results = full_reproduction_readiness()
    passed, warnings, missing = _print_results(results)
    print(f"READINESS pass={passed} warn={warnings} missing={missing}")
    if missing:
        print("BLOCKED training reproduction requires restored final-training datasets")
        return 2
    print("READY prerequisites for full reproduction are present")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HI-CSG-R canonical research verifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show canonical scientific claims")
    subparsers.add_parser("verify", help="Verify claims, evidence, manuscript, and milestones")
    subparsers.add_parser("reproduce-lite", help="Verify frozen evidence and artifact hashes")
    subparsers.add_parser("audit-data", help="Report local dataset availability and access mode")
    subparsers.add_parser("reproduce-full", help="Check full training/evaluation prerequisites")
    subparsers.add_parser("build-inventory", help="Regenerate SHA-256 artifact inventory")
    subparsers.add_parser("regenerate-tables", help="Regenerate key tables from frozen evidence")

    manifest_parser = subparsers.add_parser("validate-manifest", help="Validate a JSONL manifest")
    manifest_parser.add_argument("path", type=Path)
    manifest_parser.add_argument("--check-images", action="store_true")

    snapshot_parser = subparsers.add_parser("snapshot", help="Write a provenance snapshot")
    snapshot_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "status":
        return show_status()
    if args.command == "verify":
        return verify()
    if args.command == "reproduce-lite":
        return reproduce_lite()
    if args.command == "audit-data":
        return audit_data()
    if args.command == "reproduce-full":
        return reproduce_full_check()
    if args.command == "build-inventory":
        output = write_artifact_inventory()
        print(output.relative_to(REPO_ROOT))
        return 0
    if args.command == "regenerate-tables":
        for output in regenerate_key_tables():
            print(output.relative_to(REPO_ROOT))
        return 0
    if args.command == "snapshot":
        write_provenance(args.out)
        print(args.out)
        return 0
    if args.command == "validate-manifest":
        issues = validate_manifest(args.path, check_images=args.check_images)
        for issue in issues:
            print(f"{issue.line}: {issue.code}: {issue.message}")
        return 1 if issues else 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
