from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from src.datasets.metadata import read_jsonl


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    records = read_jsonl("data/processed/cyrillic_handwriting/metadata.final.jsonl")

    hash_to_records = defaultdict(list)

    for r in records:
        p = Path(r["image_path"])
        if not p.exists():
            continue

        h = sha256_file(p)
        hash_to_records[h].append(r)

    duplicate_groups = [v for v in hash_to_records.values() if len(v) > 1]

    print("duplicate groups:", len(duplicate_groups))
    print("duplicate samples:", sum(len(g) for g in duplicate_groups))

    leakage_groups = []
    for group in duplicate_groups:
        splits = {r["split"] for r in group}
        if len(splits) > 1:
            leakage_groups.append(group)

    print("cross-split duplicate groups:", len(leakage_groups))
    print("cross-split duplicate samples:", sum(len(g) for g in leakage_groups))

    for group in leakage_groups[:20]:
        print("\nGROUP")
        for r in group:
            print(
                r["sample_id"],
                r["split"],
                r["source_id"],
                repr(r["raw_transcription"]),
                r["image_path"],
            )


if __name__ == "__main__":
    main()