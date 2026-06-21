from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from skimage.morphology import skeletonize

from src.preprocessing.school_rectangular_v2 import (
    SchoolCocoSource,
    extract_school_lineaware_v3,
)
from tools import extract_htr_graph_features as base


VERSION = "school_rectangular_whitebalance_lineaware_postpoly_v3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_text(row: dict[str, Any]) -> str:
    for key in [
        "text",
        "target",
        "label",
        "transcription",
        "normalized_transcription",
    ]:
        value = row.get(key)

        if value is not None:
            return str(value)

    return ""


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    arr = np.asarray(values, dtype=np.float64)

    return {
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def compute_graph_row(
    row: dict[str, Any],
    coco: SchoolCocoSource,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extracted = extract_school_lineaware_v3(row, coco)

    fg = extracted["foreground"].astype(bool)
    skel = skeletonize(fg)

    h, w = fg.shape
    text = get_text(row)

    features: dict[str, Any] = {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(w / max(h, 1)),
        "text_len": int(len(text)),
        "fg_fraction": float(fg.mean()) if fg.size else 0.0,
    }

    features.update(base.bbox_features(fg))
    features.update(base.component_features(fg))
    features.update(base.graph_features(fg, skel))

    warnings = base.warnings_for(
        fg,
        skel,
        int(features["cc_count"]),
    )

    features["warning_count"] = len(warnings)

    missing = [
        key for key in base.FEATURE_KEYS
        if key not in features
    ]

    if missing:
        raise RuntimeError(
            f"Missing graph features for {row.get('sample_id')}: {missing}"
        )

    out = dict(row)
    out["graph_features"] = [
        float(features[key])
        for key in base.FEATURE_KEYS
    ]
    out["graph_feature_names"] = list(base.FEATURE_KEYS)
    out["graph_warning_count"] = len(warnings)
    out["graph_warnings"] = warnings
    out["graph_binarization"] = VERSION
    out["graph_preprocessing_version"] = VERSION
    out["school_foreground_metadata"] = extracted["method_metadata"]

    ruling_response = extracted["ruling_response"]

    diagnostics = {
        "sample_id": str(row.get("sample_id", "")),
        "dataset": str(row.get("dataset", "")),
        "split": str(row.get("split", "")),
        "width": int(w),
        "height": int(h),
        "text_len": int(len(text)),
        "fg_fraction": float(features["fg_fraction"]),
        "skel_fraction": float(features["skel_fraction"]),
        "cc_count": int(features["cc_count"]),
        "skel_components": int(features["skel_components"]),
        "endpoint_count": int(features["graph_endpoint_count"]),
        "branchpoint_count": int(features["graph_branchpoint_count"]),
        "dir_h_frac": float(features["dir_h_frac"]),
        "stroke_width_mean": float(features["stroke_width_mean"]),
        "warning_count": len(warnings),
        "warnings": ";".join(warnings),
        "ruling_response_mean": float(np.mean(ruling_response)),
        "ruling_response_p95": float(np.quantile(ruling_response, 0.95)),
    }

    return out, diagnostics


def process_split(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    coco: SchoolCocoSource,
    limit: int | None,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)

    if limit is not None:
        rows = rows[:limit]

    output_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        try:
            if row.get("dataset") == "school_notebooks_clean":
                out, diag = compute_graph_row(row, coco)
                output_rows.append(out)
                diagnostics.append(diag)
            else:
                output_rows.append(row)

        except Exception as exc:
            failures.append({
                "index": index - 1,
                "sample_id": row.get("sample_id"),
                "error": repr(exc),
            })

        if index % 500 == 0:
            print(f"[{split}] {index}/{len(rows)}")

    if failures:
        fail_path = diagnostics_path.with_suffix(".failures.json")
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"{split}: {len(failures)} failures; first={failures[:3]}"
        )

    write_jsonl(output_rows, output_path)
    write_jsonl(diagnostics, diagnostics_path)

    summary = {
        "split": split,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "diagnostics_path": str(diagnostics_path),
        "n_rows": len(output_rows),
        "school_n": len(diagnostics),
        "version": VERSION,
        "fg_fraction": quantiles([
            float(row["fg_fraction"])
            for row in diagnostics
        ]),
        "skel_fraction": quantiles([
            float(row["skel_fraction"])
            for row in diagnostics
        ]),
        "cc_count": quantiles([
            float(row["cc_count"])
            for row in diagnostics
        ]),
        "skel_components": quantiles([
            float(row["skel_components"])
            for row in diagnostics
        ]),
        "endpoint_count": quantiles([
            float(row["endpoint_count"])
            for row in diagnostics
        ]),
        "branchpoint_count": quantiles([
            float(row["branchpoint_count"])
            for row in diagnostics
        ]),
        "dir_h_frac": quantiles([
            float(row["dir_h_frac"])
            for row in diagnostics
        ]),
        "stroke_width_mean": quantiles([
            float(row["stroke_width_mean"])
            for row in diagnostics
        ]),
        "ruling_response_mean": quantiles([
            float(row["ruling_response_mean"])
            for row in diagnostics
        ]),
        "warning_rate": mean([
            float(row["warning_count"] > 0)
            for row in diagnostics
        ]),
        "warning_counts": {
            warning: sum(
                warning in str(row["warnings"]).split(";")
                for row in diagnostics
            )
            for warning in sorted({
                warning
                for row in diagnostics
                for warning in str(row["warnings"]).split(";")
                if warning
            })
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        required=True,
        help="Existing graph-ready mixed manifest root.",
    )
    parser.add_argument(
        "--school_raw_dir",
        required=True,
    )
    parser.add_argument(
        "--out_root",
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    diagnostics_root = out_root / "diagnostics"

    coco = SchoolCocoSource(args.school_raw_dir)

    summaries = []

    for split in ["train", "val", "test"]:
        print(f"\n=== {split} ===")

        summary = process_split(
            split=split,
            input_path=input_root / f"{split}.jsonl",
            output_path=out_root / f"{split}.jsonl",
            diagnostics_path=diagnostics_root / f"{split}.jsonl",
            coco=coco,
            limit=args.limit,
        )

        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    vocab_path = input_root / "vocab.json"

    if vocab_path.exists():
        (out_root / "vocab.json").write_text(
            vocab_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    final_summary = {
        "version": VERSION,
        "input_root": str(input_root),
        "out_root": str(out_root),
        "items": summaries,
    }

    (out_root / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nwrote:", out_root)


if __name__ == "__main__":
    main()
