import pytest
from src.datasets.splits import create_random_split, create_writer_independent_split


def test_random_split_is_deterministic_and_disjoint() -> None:
    records = [{"sample_id": f"s{i}"} for i in range(100)]
    first = create_random_split(records, seed=42)
    second = create_random_split(records, seed=42)
    assert first == second
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["val"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == {f"s{i}" for i in range(100)}


def test_writer_independent_split_never_splits_a_writer() -> None:
    records = [
        {"sample_id": f"{writer}-{sample}", "writer_id": writer}
        for writer in ("w1", "w2", "w3", "w4", "w5")
        for sample in range(3)
    ]
    split = create_writer_independent_split(records, train_ratio=0.6, val_ratio=0.2, seed=7)
    locations = {
        sample_id: split_name
        for split_name, sample_ids in split.items()
        for sample_id in sample_ids
    }
    for writer in ("w1", "w2", "w3", "w4", "w5"):
        assert len({locations[f"{writer}-{sample}"] for sample in range(3)}) == 1


def test_writer_independent_split_rejects_missing_writer() -> None:
    with pytest.raises(ValueError, match="writer_id is missing"):
        create_writer_independent_split([{"sample_id": "s1", "writer_id": None}])
