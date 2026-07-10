"""Canonical verification layer for the frozen HI-CSG-R research project."""

from src.pipeline.registry import (
    RegistryIssue,
    load_yaml,
    validate_claim_registry,
    validate_pipeline_registry,
)

__all__ = [
    "RegistryIssue",
    "load_yaml",
    "validate_claim_registry",
    "validate_pipeline_registry",
]
