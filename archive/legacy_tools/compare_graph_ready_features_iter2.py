from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


KEYS_OF_INTEREST = [
    "fg_fraction",
    "bbox_area_frac",
    "cc_count",
    "cc_area_mean",
    "cc_area_max_frac",
    "skel_pixels",
    "skel_fraction",
    "skel_components",
    "graph_endpoint_count",
    "graph_branchpoint_count",
    "endpoint_per_100_skel",
    "branchpoint_per_100_skel",
    "dir_h_frac",
    "dir_v_frac",
    "dir_diag_down_frac",
    "dir_diag_up_frac",
    "stroke_width_mean",
    "stroke_width_p90",
    "warning_count",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    names = row["graph_feature_names"]
    values = row["graph_features"]

    return {
        str(name): float(value)
        for name, value in zip(names, values)
    }


def q(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "p05": None,
            "median": None,
            "p95": None,
        }

    arr = np.asarray(values, dtype=np.float64)

    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def compare_split(
    old_path: Path,
    new_path: Path,
) -> dict[str, Any]:
    old_rows = read_jsonl(old_path)
    new_rows = read_jsonl(new_path)

    old_by_id = {
        str(row["sample_id"]): row
        for row in old_rows
        if row.get("dataset") == "school_notebooks_clean"
    }

    new_by_id = {
        str(row["sample_id"]): row
        for row in new_rows
        if row.get("dataset") == "school_notebooks_clean"
    }

    common = sorted(set(old_by_id) & set(new_by_id))
    old_only = sorted(set(old_by_id) - set(new_by_id))
    new_only = sorted(set(new_by_id) - set(old_by_id))

    if not common:
        raise RuntimeError(
            f"No common School sample_ids between {old_path} and {new_path}"
        )

    deltas: dict[str, list[float]] = {
        key: []
        for key in KEYS_OF_INTEREST
    }
    old_values: dict[str, list[float]] = {
        key: []
        for key in KEYS_OF_INTEREST
    }
    new_values: dict[str, list[float]] = {
        key: []
        for key in KEYS_OF_INTEREST
    }

    changed_feature_names = 0

    for sample_id in common:
        old_row = old_by_id[sample_id]
        new_row = new_by_id[sample_id]

        if old_row["graph_feature_names"] != new_row["graph_feature_names"]:
            changed_feature_names += 1

        old_f = feature_dict(old_row)
        new_f = feature_dict(new_row)

        for key in KEYS_OF_INTEREST:
            if key not in old_f or key not in new_f:
                continue

            old_v = float(old_f[key])
            new_v = float(new_f[key])

            old_values[key].append(old_v)
            new_values[key].append(new_v)
            deltas[key].append(new_v - old_v)

    return {
        "n_old_school": len(old_by_id),
        "n_new_school": len(new_by_id),
        "n_common": len(common),
        "old_only_count": len(old_only),
        "new_only_count": len(new_only),
        "changed_feature_names": changed_feature_names,
        "features": {
            key: {
                "old": q(old_values[key]),
                "new": q(new_values[key]),
                "delta": q(deltas[key]),
            }
            for key in KEYS_OF_INTEREST
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.5f}"
    except Exception:
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_root", required=True)
    parser.add_argument("--new_root", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    old_root = Path(args.old_root)
    new_root = Path(args.new_root)

    result: dict[str, Any] = {
        "old_root": str(old_root),
        "new_root": str(new_root),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        result["splits"][split] = compare_split(
            old_root / f"{split}.jsonl",
            new_root / f"{split}.jsonl",
        )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Paired School graph feature comparison - Iteration 2")
    lines.append("")

    for split, split_result in result["splits"].items():
        lines.append(f"## {split}")
        lines.append("")
        lines.append(
            f"common school samples: {split_result['n_common']}; "
            f"feature-name mismatches: {split_result['changed_feature_names']}"
        )
        lines.append("")
        lines.append("| feature | old mean | new mean | delta mean | old p95 | new p95 |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        for key in KEYS_OF_INTEREST:
            row = split_result["features"][key]
            lines.append(
                f"| `{key}` | "
                f"{fmt(row['old']['mean'])} | "
                f"{fmt(row['new']['mean'])} | "
                f"{fmt(row['delta']['mean'])} | "
                f"{fmt(row['old']['p95'])} | "
                f"{fmt(row['new']['p95'])} |"
            )

        lines.append("")

    out_md.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("wrote:", out_json)
    print("wrote:", out_md)


if __name__ == "__main__":
    main()
