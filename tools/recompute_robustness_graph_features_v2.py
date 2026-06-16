from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

from tools import extract_htr_graph_features as base


VERSION = "robustness_graph_features_v2_recomputed"
SCHOOL_MAX_FG = 0.35
SCHOOL_MIN_OBJECT = 4


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
    for key in ["text", "target", "label", "transcription"]:
        if key in row:
            return str(row[key])
    return ""


def school_dark_auto(arr: np.ndarray) -> tuple[np.ndarray, str]:
    fg145 = arr < 145

    if fg145.any():
        fg145 = remove_small_objects(
            fg145.astype(bool),
            min_size=SCHOOL_MIN_OBJECT,
        )

    if float(fg145.mean()) <= SCHOOL_MAX_FG:
        return fg145.astype(bool), "school_dark_auto_145"

    fg120 = arr < 120

    if fg120.any():
        fg120 = remove_small_objects(
            fg120.astype(bool),
            min_size=SCHOOL_MIN_OBJECT,
        )

    return fg120.astype(bool), "school_dark_auto_120"


def extract_mask(
    arr: np.ndarray,
    dataset: str,
    sauvola_window: int,
) -> tuple[np.ndarray, str]:
    if dataset == "school_notebooks_clean":
        return school_dark_auto(arr)

    method = base.DATASET_BINARIZATION.get(dataset, "otsu")
    fg = base.binarize(
        arr,
        method=method,
        sauvola_window=sauvola_window,
    )

    return fg.astype(bool), method


def process_one(
    row: dict[str, Any],
    manifest_path_str: str,
    sauvola_window: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(manifest_path_str)

    image_value = base.get_image_value(row)
    image_path = base.resolve_image_path(image_value, manifest_path)
    arr = base.load_gray(image_path)

    dataset = str(
        row.get("dataset")
        or row.get("source_dataset")
        or "unknown"
    )

    fg, method = extract_mask(
        arr,
        dataset=dataset,
        sauvola_window=sauvola_window,
    )

    skel = skeletonize(fg)

    h, w = arr.shape
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
            f"Missing features for {row.get('sample_id')}: {missing}"
        )

    out = dict(row)
    out["graph_features"] = [
        float(features[key])
        for key in base.FEATURE_KEYS
    ]
    out["graph_feature_names"] = list(base.FEATURE_KEYS)
    out["graph_warning_count"] = len(warnings)
    out["graph_warnings"] = warnings
    out["graph_binarization"] = method
    out["graph_preprocessing_version"] = VERSION
    out["graph_features_recomputed_from"] = str(image_path)

    diagnostic = {
        "sample_id": str(row.get("sample_id", "")),
        "dataset": dataset,
        "method": method,
        "fg_fraction": float(features["fg_fraction"]),
        "skel_fraction": float(features["skel_fraction"]),
        "warning_count": len(warnings),
    }

    return out, diagnostic


def process_manifest(
    input_path: Path,
    output_path: Path,
    *,
    workers: int,
    sauvola_window: int,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)

    output_rows: list[dict[str, Any] | None] = [None] * len(rows)
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if workers <= 1:
        for idx, row in enumerate(rows):
            try:
                out, diag = process_one(
                    row,
                    str(input_path),
                    sauvola_window,
                )
                output_rows[idx] = out
                diagnostics.append(diag)
            except Exception as exc:
                failures.append({
                    "index": idx,
                    "sample_id": row.get("sample_id"),
                    "error": repr(exc),
                })

            if (idx + 1) % 500 == 0:
                print(
                    f"[{input_path.stem}] "
                    f"{idx + 1}/{len(rows)}"
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_one,
                    row,
                    str(input_path),
                    sauvola_window,
                ): idx
                for idx, row in enumerate(rows)
            }

            for done, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                idx = futures[future]

                try:
                    out, diag = future.result()
                    output_rows[idx] = out
                    diagnostics.append(diag)
                except Exception as exc:
                    failures.append({
                        "index": idx,
                        "sample_id": rows[idx].get("sample_id"),
                        "error": repr(exc),
                    })

                if done % 500 == 0:
                    print(
                        f"[{input_path.stem}] "
                        f"{done}/{len(rows)}"
                    )

    good_rows = [
        row for row in output_rows
        if row is not None
    ]

    if failures:
        raise RuntimeError(
            f"{input_path}: {len(failures)} failures; "
            f"first={failures[:3]}"
        )

    write_jsonl(good_rows, output_path)

    school_rows = [
        d for d in diagnostics
        if d["dataset"] == "school_notebooks_clean"
    ]

    summary = {
        "condition": input_path.stem,
        "input_manifest": str(input_path),
        "output_manifest": str(output_path),
        "n": len(good_rows),
        "failure_n": len(failures),
        "fg_fraction_mean": float(np.mean([
            d["fg_fraction"] for d in diagnostics
        ])),
        "skel_fraction_mean": float(np.mean([
            d["skel_fraction"] for d in diagnostics
        ])),
        "warning_rate": float(np.mean([
            d["warning_count"] > 0
            for d in diagnostics
        ])),
        "school_n": len(school_rows),
        "school_threshold_120_rate": (
            float(np.mean([
                d["method"] == "school_dark_auto_120"
                for d in school_rows
            ]))
            if school_rows else 0.0
        ),
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sauvola_window", type=int, default=25)
    args = parser.parse_args()

    input_dir = Path(args.input_manifest_dir)
    out_dir = Path(args.out_dir)
    output_manifest_dir = out_dir / "manifests"
    output_manifest_dir.mkdir(parents=True, exist_ok=True)

    manifests = sorted(input_dir.glob("*.jsonl"))
    if not manifests:
        raise RuntimeError(f"No manifests found in {input_dir}")

    summaries = []

    for idx, input_path in enumerate(manifests, start=1):
        print(
            f"\n=== {idx}/{len(manifests)} "
            f"{input_path.stem} ==="
        )

        output_path = output_manifest_dir / input_path.name

        summary = process_manifest(
            input_path,
            output_path,
            workers=args.workers,
            sauvola_window=args.sauvola_window,
        )
        summaries.append(summary)

        print(json.dumps(summary, ensure_ascii=False, indent=2))

    final = {
        "version": VERSION,
        "input_manifest_dir": str(input_dir),
        "output_manifest_dir": str(output_manifest_dir),
        "condition_n": len(summaries),
        "items": summaries,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nwrote:", out_dir / "summary.json")


if __name__ == "__main__":
    main()