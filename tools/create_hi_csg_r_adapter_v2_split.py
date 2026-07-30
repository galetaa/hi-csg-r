from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.dataset_adapter_v2 import core_domain
from src.htr.xaligned_hi_csg_r import read_jsonl

SPLIT_VERSION = "hi_csg_r_adapter_v2_split_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def group_key(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset") or row.get("source_dataset") or "unknown")
    metadata = row.get("source_metadata") or {}
    candidates = (
        ("writer", row.get("writer_id")),
        ("page", metadata.get("page_id")),
        ("page", row.get("page_id")),
        ("source_group", row.get("source_group")),
        ("line_group", row.get("line_group_id")),
        ("source_file", metadata.get("source_image_file")),
        ("source_file", row.get("source_image_file")),
        ("source_path", row.get("source_path")),
        ("image_path", row.get("image_path")),
        ("sample", row.get("sample_id")),
    )
    for kind, value in candidates:
        if value not in {None, ""}:
            return f"{dataset}|{kind}|{Path(str(value)).as_posix().lower()}"
    raise ValueError("Row has no usable grouping identifier")


def leakage_safe_group_keys(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Union hierarchy groups that share an exact source-image SHA1."""
    base_keys = [group_key(row) for row in rows]
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_group: dict[str, int] = {}
    first_by_sha1: dict[str, int] = {}
    for index, (row, key) in enumerate(zip(rows, base_keys, strict=True)):
        if key in first_by_group:
            union(index, first_by_group[key])
        else:
            first_by_group[key] = index
        source_sha1 = str(row.get("xaligned_source_image_sha1") or "")
        if source_sha1:
            if source_sha1 in first_by_sha1:
                union(index, first_by_sha1[source_sha1])
            else:
                first_by_sha1[source_sha1] = index

    component_names: defaultdict[int, list[str]] = defaultdict(list)
    for index, key in enumerate(base_keys):
        component_names[find(index)].append(key)
    canonical = {
        root: min(names)
        for root, names in component_names.items()
    }
    return {
        str(row["sample_id"]): canonical[find(index)]
        for index, row in enumerate(rows)
    }


def choose_group_subset(
    groups: list[tuple[str, list[dict[str, Any]]]],
    target: int,
    *,
    seed: int,
) -> tuple[set[str], int]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, rows) in enumerate(shuffled):
        size = len(rows)
        additions: dict[int, tuple[int, ...]] = {}
        for total, chosen in list(reachable.items()):
            candidate = total + size
            if candidate <= target and candidate not in reachable:
                additions[candidate] = (*chosen, index)
        reachable.update(additions)
        if target in reachable:
            break
    selected_total = max(reachable)
    selected = {shuffled[index][0] for index in reachable[selected_total]}
    return selected, selected_total


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def slim_output_row(row: dict[str, Any], split: str, safe_group: str) -> dict[str, Any]:
    keep = (
        "sample_id",
        "dataset",
        "source_dataset",
        "level",
        "language",
        "script",
        "image_path",
        "text",
        "text_len",
        "category",
        "image_info",
        "xaligned_graph_npz",
        "xaligned_graph_version",
        "xaligned_feature_dim",
        "xaligned_feature_source",
        "xaligned_time_steps",
        "xaligned_source_image_sha1",
    )
    output = {key: row.get(key) for key in keep if key in row}
    output.update(
        {
            "source_split": row.get("split"),
            "split": split,
            "adapter_v2_split": split,
            "adapter_v2_group_id": safe_group,
            "adapter_v2_split_version": SPLIT_VERSION,
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--features_out_dir")
    parser.add_argument("--dev_per_domain", type=int, default=1000)
    parser.add_argument("--holdout_per_domain", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    source = Path(args.train_manifest)
    rows = read_jsonl(source)
    safe_group_by_sample = leakage_safe_group_keys(rows)
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    group_domains: dict[str, str] = {}
    for row in rows:
        key = safe_group_by_sample[str(row["sample_id"])]
        domain = core_domain(str(row.get("dataset") or row.get("source_dataset")))
        if domain not in {"cyrillic", "hkr", "school"}:
            raise ValueError(f"Unsupported core domain: {domain}")
        if key in group_domains and group_domains[key] != domain:
            raise ValueError(f"Group crosses core domains: {key}")
        grouped[key].append(row)
        group_domains[key] = domain

    assignments: dict[str, str] = {}
    selection_summary: dict[str, Any] = {}
    for domain_index, domain in enumerate(("cyrillic", "hkr", "school")):
        domain_groups = [
            (key, values)
            for key, values in grouped.items()
            if group_domains[key] == domain
        ]
        holdout_groups, holdout_n = choose_group_subset(
            domain_groups,
            args.holdout_per_domain,
            seed=args.seed + 100 * domain_index + 1,
        )
        remaining = [
            item for item in domain_groups if item[0] not in holdout_groups
        ]
        dev_groups, dev_n = choose_group_subset(
            remaining,
            args.dev_per_domain,
            seed=args.seed + 100 * domain_index + 2,
        )
        for key, _ in domain_groups:
            assignments[key] = (
                "holdout"
                if key in holdout_groups
                else "dev"
                if key in dev_groups
                else "train"
            )
        selection_summary[domain] = {
            "source_samples": sum(len(values) for _, values in domain_groups),
            "groups": len(domain_groups),
            "dev_target": args.dev_per_domain,
            "dev_actual": dev_n,
            "holdout_target": args.holdout_per_domain,
            "holdout_actual": holdout_n,
        }

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "holdout": [],
    }
    for row in rows:
        safe_group = safe_group_by_sample[str(row["sample_id"])]
        name = assignments[safe_group]
        split_rows[name].append(slim_output_row(row, name, safe_group))

    out_dir = Path(args.out_dir)
    features_out = (
        Path(args.features_out_dir)
        if args.features_out_dir
        else out_dir.parent / "features"
    )
    manifest_hashes: dict[str, str] = {}
    for name, values in split_rows.items():
        random.Random(args.seed + len(name)).shuffle(values)
        split_path = out_dir / f"{name}.jsonl"
        feature_path = features_out / f"{name}.jsonl"
        write_jsonl(split_path, values)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        if feature_path.exists() or feature_path.is_symlink():
            feature_path.unlink()
        feature_path.symlink_to(os.path.relpath(split_path, feature_path.parent))
        manifest_hashes[name] = sha256_file(split_path)

    summary = {
        "status": "CREATED",
        "version": SPLIT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source),
        "source_samples": len(rows),
        "split_counts": {name: len(values) for name, values in split_rows.items()},
        "domain_selection": selection_summary,
        "manifest_sha256": manifest_hashes,
        "group_hierarchy": [
            "writer_id",
            "source_metadata.page_id",
            "page_id",
            "source_group",
            "line_group_id",
            "source_image_file",
            "source_path",
            "image_path",
            "sample_id",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
