from __future__ import annotations

from src.htr.dataset_adapter_v2 import DomainBalancedBatchSampler
from tools.create_hi_csg_r_adapter_v2_split import (
    choose_group_subset,
    group_key,
    leakage_safe_group_keys,
)


def test_split_subset_selection_reaches_exact_target() -> None:
    groups = [(f"g{index}", [{}] * size) for index, size in enumerate([7, 5, 3, 2, 1])]
    selected, total = choose_group_subset(groups, 10, seed=20260730)
    assert total == 10
    assert sum(len(dict(groups)[name]) for name in selected) == 10


def test_group_key_prefers_page_over_image_path() -> None:
    row = {
        "dataset": "hkr_words",
        "sample_id": "s1",
        "image_path": "images/a.png",
        "writer_id": None,
        "source_metadata": {"page_id": "page-7"},
    }
    assert group_key(row) == "hkr_words|page|page-7"


def test_domain_balanced_sampler_is_reproducible() -> None:
    rows = [
        {"dataset": domain, "image_info": {"width": index + 1}}
        for domain in ("cyrillic_handwriting", "hkr_words", "school_notebooks_clean")
        for index in range(12)
    ]
    first = DomainBalancedBatchSampler(rows, 6, seed=42)
    second = DomainBalancedBatchSampler(rows, 6, seed=42)
    first.set_epoch(3)
    second.set_epoch(3)
    assert list(first) == list(second)
    for batch in first:
        domains = [rows[index]["dataset"] for index in batch]
        assert sum("cyrillic" in value for value in domains) == 2
        assert sum("hkr" in value for value in domains) == 2
        assert sum("school" in value for value in domains) == 2


def test_exact_hash_unions_distinct_hierarchy_groups() -> None:
    rows = [
        {
            "dataset": "hkr_words",
            "sample_id": "a",
            "image_path": "a.png",
            "source_metadata": {"page_id": "p1"},
            "xaligned_source_image_sha1": "same",
        },
        {
            "dataset": "hkr_words",
            "sample_id": "b",
            "image_path": "b.png",
            "source_metadata": {"page_id": "p2"},
            "xaligned_source_image_sha1": "same",
        },
    ]
    safe = leakage_safe_group_keys(rows)
    assert safe["a"] == safe["b"]
