from __future__ import annotations

from collections import Counter, defaultdict

from src.datasets.metadata import read_jsonl


def main() -> None:
    records = read_jsonl("data/processed/iam/metadata.clean_splits.jsonl")

    print("split counts:", Counter(r["split"] for r in records))
    print("num records:", len(records))

    split_to_writers = defaultdict(set)
    writer_to_splits = defaultdict(set)

    missing_writer = 0

    for r in records:
        writer_id = r.get("writer_id")
        split = r.get("split")

        if not writer_id:
            missing_writer += 1
            continue

        split_to_writers[split].add(writer_id)
        writer_to_splits[writer_id].add(split)

    leakage = {w: s for w, s in writer_to_splits.items() if len(s) > 1}

    print("missing writer:", missing_writer)
    print("writers by split:", {k: len(v) for k, v in split_to_writers.items()})
    print("writer leakage:", len(leakage))

    if leakage:
        for writer_id, splits in list(leakage.items())[:20]:
            print(writer_id, splits)


if __name__ == "__main__":
    main()
