from pathlib import Path

import yaml
from src.pipeline.registry import (
    REPO_ROOT,
    load_yaml,
    validate_claim_registry,
    validate_pipeline_registry,
)


def test_reproducibility_registries_load() -> None:
    datasets = load_yaml("research/datasets.yaml")
    evidence = load_yaml("research/evidence.yaml")
    environments = load_yaml("research/environment_profiles.yaml")
    assert len(datasets["datasets"]) == 6
    assert len(evidence["checks"]) >= 12
    assert environments["determinism_policy"]["seeds"] == [42, 43, 44]


def test_canonical_claim_registry_is_valid() -> None:
    assert validate_claim_registry() == []


def test_canonical_pipeline_registry_is_valid() -> None:
    assert validate_pipeline_registry() == []


def test_v11_is_declared_as_canonical_manuscript() -> None:
    registry = load_yaml("research/claims.yaml")
    manuscript = registry["canonical_manuscript"]
    assert manuscript["path"] == "article/HI_CSG_R_v11.docx"
    assert (REPO_ROOT / manuscript["path"]).is_file()


def test_claim_ids_and_hypothesis_mapping_are_stable() -> None:
    registry = load_yaml("research/claims.yaml")
    mapping = {
        claim["manuscript_hypothesis"]: claim["id"]
        for claim in registry["claims"]
        if claim["manuscript_hypothesis"] is not None
    }
    assert mapping == {
        "H1": "H1-STRUCTURAL-DIAGNOSTICS",
        "H2": "H2-RELEVANT-DATA-AUGMENTATION",
        "H3": "H3-SELECTIVE-PREDICTION",
        "H4": "H4-GRAPH-FUSION",
    }

    primary = next(
        claim for claim in registry["claims"] if claim["id"] == "H2-RELEVANT-DATA-AUGMENTATION"
    )
    assert primary["headline_metrics"]["baseline_mean_cer"] == 0.152431
    assert primary["headline_metrics"]["augmented_mean_cer"] == 0.135479
    assert primary["headline_metrics"]["seeds"] == [42, 43, 44]


def test_registry_rejects_duplicate_claim_ids(tmp_path: Path) -> None:
    path = tmp_path / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "claims": [
                    {
                        "id": "DUPLICATE",
                        "status": "supported",
                        "statement": "one",
                        "evidence": [{"path": "docs/00_research_problem.md"}],
                    },
                    {
                        "id": "DUPLICATE",
                        "status": "exploratory",
                        "statement": "two",
                        "evidence": [{"path": "docs/00_research_problem.md"}],
                    },
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    codes = {issue.code for issue in validate_claim_registry(path)}
    assert "claim_id_duplicate" in codes
