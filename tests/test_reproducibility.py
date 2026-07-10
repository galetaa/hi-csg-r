from __future__ import annotations

import random

import numpy as np
from src.pipeline.determinism import seed_everything
from src.pipeline.reproducibility import (
    audit_datasets,
    verify_artifact_inventory,
    verify_evidence,
)


def test_frozen_evidence_has_no_failures() -> None:
    results = verify_evidence()
    assert not [result for result in results if result.status == "FAIL"]
    assert any(result.status == "WARN" for result in results)


def test_artifact_inventory_matches_current_files() -> None:
    results = verify_artifact_inventory()
    assert len(results) >= 80
    assert all(result.status == "PASS" for result in results)


def test_dataset_audit_is_explicit_about_absent_local_data() -> None:
    results = audit_datasets()
    assert {result.status for result in results} <= {"PASS", "MISSING"}
    assert {result.check_id for result in results} >= {"iam", "hkr_words", "hwr200"}


def test_seed_policy_repeats_python_and_numpy_streams() -> None:
    seed_everything(42)
    first = (random.random(), float(np.random.random()))
    seed_everything(42)
    second = (random.random(), float(np.random.random()))
    assert first == second
