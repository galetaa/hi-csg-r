from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.dataset_adapter_v2 import core_domain
from src.htr.xaligned_hi_csg_r import read_jsonl, resolve_path

from tools.create_hi_csg_r_adapter_v2_split import group_key


def image_sha1(row: dict[str, Any], manifest: Path) -> str:
    cached = row.get("xaligned_source_image_sha1")
    if cached:
        return str(cached)
    digest = hashlib.sha1()
    with resolve_path(str(row["image_path"]), manifest).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pair_overlap(sets: dict[str, set[str]]) -> dict[str, int]:
    names = ("train", "dev", "holdout")
    return {
        f"{left}_{right}": len(sets[left] & sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--expected_train", type=int, default=35498)
    parser.add_argument("--expected_dev", type=int, default=3000)
    parser.add_argument("--expected_holdout", type=int, default=1500)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    manifests = {name: split_dir / f"{name}.jsonl" for name in ("train", "dev", "holdout")}
    rows = {name: read_jsonl(path) for name, path in manifests.items()}
    sample_sets = {
        name: {str(row["sample_id"]) for row in values}
        for name, values in rows.items()
    }
    path_sets = {
        name: {str(row["image_path"]) for row in values}
        for name, values in rows.items()
    }
    group_sets = {
        name: {group_key(row) for row in values}
        for name, values in rows.items()
    }
    hash_sets = {
        name: {image_sha1(row, manifests[name]) for row in values}
        for name, values in rows.items()
    }
    counts = {name: len(values) for name, values in rows.items()}
    expected = {
        "train": args.expected_train,
        "dev": args.expected_dev,
        "holdout": args.expected_holdout,
    }
    duplicate_ids = {
        name: len(values) - len(sample_sets[name])
        for name, values in rows.items()
    }
    missing_images = {
        name: sum(
            not resolve_path(str(row["image_path"]), manifests[name]).exists()
            for row in values
        )
        for name, values in rows.items()
    }
    missing_features = {
        name: sum(
            not resolve_path(str(row["xaligned_graph_npz"]), manifests[name]).exists()
            for row in values
        )
        for name, values in rows.items()
    }
    domain_counts = {
        name: dict(
            sorted(
                Counter(
                    core_domain(str(row.get("dataset") or row.get("source_dataset")))
                    for row in values
                ).items()
            )
        )
        for name, values in rows.items()
    }
    overlaps = {
        "sample_id": pair_overlap(sample_sets),
        "image_path": pair_overlap(path_sets),
        "group": pair_overlap(group_sets),
        "image_sha1": pair_overlap(hash_sets),
    }
    failures: list[str] = []
    if counts != expected:
        failures.append(f"split counts differ: expected={expected}, actual={counts}")
    if any(duplicate_ids.values()):
        failures.append(f"duplicate sample ids: {duplicate_ids}")
    if any(missing_images.values()):
        failures.append(f"missing images: {missing_images}")
    if any(missing_features.values()):
        failures.append(f"missing features: {missing_features}")
    for kind, values in overlaps.items():
        if any(values.values()):
            failures.append(f"{kind} overlap: {values}")
    for name in ("dev", "holdout"):
        domain_target = 1000 if name == "dev" else 500
        if any(domain_counts[name].get(domain, 0) != domain_target for domain in ("cyrillic", "hkr", "school")):
            failures.append(f"{name} domain counts differ: {domain_counts[name]}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "counts": counts,
        "domain_counts": domain_counts,
        "duplicate_sample_ids": duplicate_ids,
        "missing_images": missing_images,
        "missing_features": missing_features,
        "overlaps": overlaps,
        "near_duplicate_audit": {
            "status": "NOT_AVAILABLE",
            "note": "No frozen perceptual-hash infrastructure exists; exact SHA1 overlap is enforced.",
        },
        "failures": failures,
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# HI-CSG-R adapter v2 split audit",
        "",
        f"**Status:** {report['status']}",
        "",
        "## Counts",
        "",
        "| Split | Total | Cyrillic | HKR | School |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "dev", "holdout"):
        values = domain_counts[name]
        lines.append(
            f"| {name} | {counts[name]} | {values.get('cyrillic', 0)} | "
            f"{values.get('hkr', 0)} | {values.get('school', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            f"- sample overlap: `{overlaps['sample_id']}`",
            f"- path overlap: `{overlaps['image_path']}`",
            f"- group overlap: `{overlaps['group']}`",
            f"- SHA1 overlap: `{overlaps['image_sha1']}`",
            f"- missing images: `{missing_images}`",
            f"- missing feature records: `{missing_features}`",
            f"- failures: `{failures}`",
        ]
    )
    (output / "split_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

