from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_CHANGE_KEYS = [
    "fg_fraction",
    "dir_h_frac",
    "graph_branchpoint_count",
    "stroke_width_mean",
    "stroke_width_p90",
]

ANNOTATION_FIELDS = [
    "usable",
    "ink_loss",
    "line_residual",
    "neighbor_text_removed",
    "skeleton_follows_ink",
    "notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    names = row.get("graph_feature_names") or []
    values = row.get("graph_features") or []

    return {
        str(name): float(value)
        for name, value in zip(names, values)
    }


def get_text(row: dict[str, Any]) -> str:
    for key in [
        "text",
        "target",
        "label",
        "transcription",
        "normalized_transcription",
    ]:
        if row.get(key) is not None:
            return str(row[key])

    return ""


def load_school_rows(root: Path) -> dict[str, dict[str, Any]]:
    by_id = {}

    for split in ["train", "val", "test"]:
        path = root / f"{split}.jsonl"

        for row in read_jsonl(path):
            if row.get("dataset") != "school_notebooks_clean":
                continue

            sample_id = str(row["sample_id"])
            item = dict(row)
            item["_split"] = split
            item["_features"] = feature_dict(row)
            by_id[sample_id] = item

    return by_id


def load_diagnostics(root: Path) -> dict[str, dict[str, Any]]:
    by_id = {}

    for split in ["train", "val", "test"]:
        path = root / "diagnostics" / f"{split}.jsonl"

        for row in read_jsonl(path):
            sample_id = str(row["sample_id"])
            item = dict(row)
            item["_split"] = split
            by_id[sample_id] = item

    return by_id


def q(values: list[float], frac: float) -> float:
    if not values:
        return 0.0

    return float(np.quantile(np.asarray(values, dtype=np.float64), frac))


def feature_change_score(
    old_features: dict[str, float],
    new_features: dict[str, float],
    scales: dict[str, float],
) -> float:
    score = 0.0

    for key in FEATURE_CHANGE_KEYS:
        if key not in old_features or key not in new_features:
            continue

        scale = max(scales.get(key, 0.0), 1e-9)
        score += abs(new_features[key] - old_features[key]) / scale

    return float(score)


def take(
    rows: list[dict[str, Any]],
    *,
    group: str,
    n: int,
    selected: set[str],
) -> list[dict[str, Any]]:
    out = []

    for row in rows:
        sample_id = str(row["sample_id"])

        if sample_id in selected:
            continue

        item = dict(row)
        item["validation_group"] = group
        out.append(item)
        selected.add(sample_id)

        if len(out) >= n:
            break

    if len(out) < n:
        raise RuntimeError(
            f"Only selected {len(out)} samples for {group}; requested {n}"
        )

    return out


def build_output_row(
    row: dict[str, Any],
    *,
    old_rows: dict[str, dict[str, Any]],
    new_rows: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    old_row = old_rows[sample_id]
    new_row = new_rows[sample_id]
    diag = diagnostics.get(sample_id, {})

    old_f = old_row["_features"]
    new_f = new_row["_features"]

    out = {
        "sample_id": sample_id,
        "validation_group": row["validation_group"],
        "split": new_row.get("_split", ""),
        "dataset": new_row.get("dataset", ""),
        "level": new_row.get("level", ""),
        "category": new_row.get("category", ""),
        "image_path": new_row.get("image_path", ""),
        "target": get_text(new_row),
        "old_fg_fraction": old_f.get("fg_fraction", ""),
        "new_fg_fraction": new_f.get("fg_fraction", ""),
        "delta_fg_fraction": new_f.get("fg_fraction", 0.0) - old_f.get("fg_fraction", 0.0),
        "old_dir_h_frac": old_f.get("dir_h_frac", ""),
        "new_dir_h_frac": new_f.get("dir_h_frac", ""),
        "delta_dir_h_frac": new_f.get("dir_h_frac", 0.0) - old_f.get("dir_h_frac", 0.0),
        "old_branchpoint_count": old_f.get("graph_branchpoint_count", ""),
        "new_branchpoint_count": new_f.get("graph_branchpoint_count", ""),
        "old_warning_count": old_f.get("warning_count", ""),
        "new_warning_count": new_f.get("warning_count", ""),
        "ruling_response_mean": diag.get("ruling_response_mean", ""),
        "ruling_response_p95": diag.get("ruling_response_p95", ""),
        "feature_change_score": row.get("feature_change_score", ""),
        "selection_reason": row.get("selection_reason", ""),
    }

    for field in ANNOTATION_FIELDS:
        out[field] = ""

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_root", required=True)
    parser.add_argument("--new_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--n_per_group", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    old_root = Path(args.old_root)
    new_root = Path(args.new_root)

    old_rows = load_school_rows(old_root)
    new_rows = load_school_rows(new_root)
    diagnostics = load_diagnostics(new_root)

    common_ids = sorted(set(old_rows) & set(new_rows) & set(diagnostics))

    if not common_ids:
        raise RuntimeError("No common School sample_ids")

    old_fg_values = [
        old_rows[sid]["_features"].get("fg_fraction", 0.0)
        for sid in common_ids
    ]

    old_fg_p95 = q(old_fg_values, 0.95)

    diff_values_by_key: dict[str, list[float]] = {
        key: []
        for key in FEATURE_CHANGE_KEYS
    }

    candidates = []

    for sample_id in common_ids:
        old_row = old_rows[sample_id]
        new_row = new_rows[sample_id]
        diag = diagnostics[sample_id]

        old_f = old_row["_features"]
        new_f = new_row["_features"]

        for key in FEATURE_CHANGE_KEYS:
            if key in old_f and key in new_f:
                diff_values_by_key[key].append(abs(new_f[key] - old_f[key]))

        candidates.append({
            "sample_id": sample_id,
            "old_row": old_row,
            "new_row": new_row,
            "diag": diag,
            "old_features": old_f,
            "new_features": new_f,
        })

    scales = {
        key: max(q(values, 0.95), 1e-9)
        for key, values in diff_values_by_key.items()
    }

    enriched = []
    for item in candidates:
        old_f = item["old_features"]
        new_f = item["new_features"]

        score = feature_change_score(
            old_f,
            new_f,
            scales,
        )

        row = dict(item["new_row"])
        row["feature_change_score"] = score
        row["ruling_response_mean"] = float(item["diag"].get("ruling_response_mean", 0.0))
        row["old_warning_count"] = float(old_f.get("warning_count", 0.0))
        row["old_fg_fraction"] = float(old_f.get("fg_fraction", 0.0))
        row["selection_reason"] = ""
        enriched.append(row)

    selected: set[str] = set()
    selected_rows: list[dict[str, Any]] = []

    binary_issue = sorted(
        [
            row for row in enriched
            if row["old_warning_count"] > 0
            or row["old_fg_fraction"] >= old_fg_p95
        ],
        key=lambda row: (
            row["old_warning_count"],
            row["old_fg_fraction"],
        ),
        reverse=True,
    )
    for row in binary_issue:
        row["selection_reason"] = "old_warning_or_foreground_heavy"
    selected_rows.extend(take(
        binary_issue,
        group="old_binary_issue",
        n=args.n_per_group,
        selected=selected,
    ))

    high_ruling = sorted(
        enriched,
        key=lambda row: float(row["ruling_response_mean"]),
        reverse=True,
    )
    for row in high_ruling:
        row["selection_reason"] = "top_ruling_response_mean"
    selected_rows.extend(take(
        high_ruling,
        group="high_ruling_response",
        n=args.n_per_group,
        selected=selected,
    ))

    high_change = sorted(
        enriched,
        key=lambda row: float(row["feature_change_score"]),
        reverse=True,
    )
    for row in high_change:
        row["selection_reason"] = "top_feature_change_score"
    selected_rows.extend(take(
        high_change,
        group="high_feature_change",
        n=args.n_per_group,
        selected=selected,
    ))

    stable = [
        row for row in enriched
        if str(row["sample_id"]) not in selected
    ]
    stable.sort(
        key=lambda row: (
            abs(float(row["feature_change_score"])),
            abs(float(row["ruling_response_mean"]) - q([
                float(r["ruling_response_mean"])
                for r in enriched
            ], 0.50)),
        )
    )

    stable_pool = stable[: max(args.n_per_group * 8, args.n_per_group)]
    rng = random.Random(args.seed)
    rng.shuffle(stable_pool)
    for row in stable_pool:
        row["selection_reason"] = "low_change_stable_control"
    selected_rows.extend(take(
        stable_pool,
        group="random_stable_control",
        n=args.n_per_group,
        selected=selected,
    ))

    output_rows = [
        build_output_row(
            row,
            old_rows=old_rows,
            new_rows=new_rows,
            diagnostics=diagnostics,
        )
        for row in selected_rows
    ]

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "old_root": str(old_root),
        "new_root": str(new_root),
        "seed": args.seed,
        "n_per_group": args.n_per_group,
        "selected_n": len(output_rows),
        "groups": {
            group: sum(row["validation_group"] == group for row in output_rows)
            for group in sorted({row["validation_group"] for row in output_rows})
        },
        "old_fg_p95": old_fg_p95,
        "feature_change_scales": scales,
        "annotation_fields": ANNOTATION_FIELDS,
    }

    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_csv)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()
