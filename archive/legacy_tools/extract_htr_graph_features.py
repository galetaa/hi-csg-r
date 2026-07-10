from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import remove_small_objects, skeletonize


DATASET_BINARIZATION = {
    "cyrillic_handwriting": "otsu",
    "hkr_words": "otsu",
    "school_notebooks_clean": "sauvola",
}

IMAGE_KEYS = [
    "image_path",
    "path",
    "crop_path",
    "file_path",
    "img_path",
    "image",
]

FEATURE_KEYS = [
    "width",
    "height",
    "aspect_ratio",
    "text_len",

    "fg_fraction",
    "bbox_x0_frac",
    "bbox_y0_frac",
    "bbox_w_frac",
    "bbox_h_frac",
    "bbox_area_frac",

    "cc_count",
    "cc_area_mean",
    "cc_area_median",
    "cc_area_max_frac",

    "skel_pixels",
    "skel_fraction",
    "skel_components",

    "graph_nodes",
    "graph_edges_8n",
    "graph_avg_degree",
    "graph_endpoint_count",
    "graph_branchpoint_count",
    "graph_isolated_count",

    "endpoint_per_100_skel",
    "branchpoint_per_100_skel",

    "degree_hist_0",
    "degree_hist_1",
    "degree_hist_2",
    "degree_hist_3",
    "degree_hist_4",
    "degree_hist_5plus",

    "dir_h_frac",
    "dir_v_frac",
    "dir_diag_down_frac",
    "dir_diag_up_frac",

    "stroke_width_mean",
    "stroke_width_std",
    "stroke_width_p50",
    "stroke_width_p90",

    "warning_count",
]


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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "dataset",
        "split",
        "image_path",
        "text",
        "binarization",
        "warnings",
    ] + FEATURE_KEYS

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def stable_id(row: dict[str, Any], dataset: str, split: str, idx: int) -> str:
    for key in ["sample_id", "id", "uid", "record_id"]:
        if row.get(key):
            return str(row[key])

    raw = "|".join([
        dataset,
        split,
        str(idx),
        str(row.get("image_path", row.get("path", ""))),
        str(row.get("text", "")),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "label", "transcription", "target"]:
        if key in row:
            return str(row[key])
    return ""


def get_image_value(row: dict[str, Any]) -> str:
    for key in IMAGE_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"No image path key found. Available keys: {sorted(row.keys())}")


def resolve_image_path(value: str, manifest_path: Path) -> Path:
    p = Path(value)

    candidates = [
        p,
        Path.cwd() / p,
        manifest_path.parent / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Image not found: {value}")


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    return arr


def binarize(arr: np.ndarray, method: str, sauvola_window: int) -> np.ndarray:
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)

    if method == "otsu":
        try:
            t = threshold_otsu(arr)
        except ValueError:
            t = 127
        fg = arr < t

    elif method == "sauvola":
        window = sauvola_window
        if window % 2 == 0:
            window += 1
        window = max(7, window)

        # window cannot be larger than image dimensions in a useful way
        min_dim = min(arr.shape)
        if min_dim >= 7:
            window = min(window, min_dim if min_dim % 2 == 1 else min_dim - 1)
            window = max(7, window)
            t = threshold_sauvola(arr, window_size=window)
            fg = arr < t
        else:
            try:
                t = threshold_otsu(arr)
            except ValueError:
                t = 127
            fg = arr < t

    else:
        raise ValueError(f"Unknown binarization method: {method}")

    # Safety: handwriting foreground should normally be minority.
    if fg.mean() > 0.5:
        fg = ~fg

    return fg.astype(bool)


def bbox_features(fg: np.ndarray) -> dict[str, float]:
    ys, xs = np.where(fg)
    h, w = fg.shape

    if len(xs) == 0:
        return {
            "bbox_x0_frac": 0.0,
            "bbox_y0_frac": 0.0,
            "bbox_w_frac": 0.0,
            "bbox_h_frac": 0.0,
            "bbox_area_frac": 0.0,
        }

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1

    return {
        "bbox_x0_frac": x0 / max(w, 1),
        "bbox_y0_frac": y0 / max(h, 1),
        "bbox_w_frac": bw / max(w, 1),
        "bbox_h_frac": bh / max(h, 1),
        "bbox_area_frac": (bw * bh) / max(w * h, 1),
    }


def component_features(mask: np.ndarray) -> dict[str, float]:
    struct = np.ones((3, 3), dtype=np.uint8)
    labels, n = ndi.label(mask, structure=struct)

    if n == 0:
        return {
            "cc_count": 0,
            "cc_area_mean": 0.0,
            "cc_area_median": 0.0,
            "cc_area_max_frac": 0.0,
        }

    areas = np.bincount(labels.ravel())[1:]
    total = max(mask.size, 1)

    return {
        "cc_count": int(n),
        "cc_area_mean": float(np.mean(areas)),
        "cc_area_median": float(np.median(areas)),
        "cc_area_max_frac": float(np.max(areas) / total),
    }


def skeleton_degree(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    neigh = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    # subtract self
    degree = neigh - skel.astype(np.uint8)
    return degree


def direction_features(skel: np.ndarray) -> dict[str, float]:
    if skel.size == 0:
        total = 0
    else:
        h_pairs = int(np.logical_and(skel[:, :-1], skel[:, 1:]).sum()) if skel.shape[1] > 1 else 0
        v_pairs = int(np.logical_and(skel[:-1, :], skel[1:, :]).sum()) if skel.shape[0] > 1 else 0
        dd_pairs = int(np.logical_and(skel[:-1, :-1], skel[1:, 1:]).sum()) if min(skel.shape) > 1 else 0
        du_pairs = int(np.logical_and(skel[1:, :-1], skel[:-1, 1:]).sum()) if min(skel.shape) > 1 else 0
        total = h_pairs + v_pairs + dd_pairs + du_pairs

    if total == 0:
        return {
            "dir_h_frac": 0.0,
            "dir_v_frac": 0.0,
            "dir_diag_down_frac": 0.0,
            "dir_diag_up_frac": 0.0,
        }

    return {
        "dir_h_frac": h_pairs / total,
        "dir_v_frac": v_pairs / total,
        "dir_diag_down_frac": dd_pairs / total,
        "dir_diag_up_frac": du_pairs / total,
    }


def stroke_width_features(fg: np.ndarray, skel: np.ndarray) -> dict[str, float]:
    if not skel.any():
        return {
            "stroke_width_mean": 0.0,
            "stroke_width_std": 0.0,
            "stroke_width_p50": 0.0,
            "stroke_width_p90": 0.0,
        }

    dist = ndi.distance_transform_edt(fg)
    widths = 2.0 * dist[skel]

    if widths.size == 0:
        return {
            "stroke_width_mean": 0.0,
            "stroke_width_std": 0.0,
            "stroke_width_p50": 0.0,
            "stroke_width_p90": 0.0,
        }

    return {
        "stroke_width_mean": float(np.mean(widths)),
        "stroke_width_std": float(np.std(widths)),
        "stroke_width_p50": float(np.percentile(widths, 50)),
        "stroke_width_p90": float(np.percentile(widths, 90)),
    }


def graph_features(fg: np.ndarray, skel: np.ndarray) -> dict[str, float]:
    skel_pixels = int(skel.sum())

    if skel_pixels == 0:
        base = {
            "skel_pixels": 0,
            "skel_fraction": 0.0,
            "skel_components": 0,
            "graph_nodes": 0,
            "graph_edges_8n": 0,
            "graph_avg_degree": 0.0,
            "graph_endpoint_count": 0,
            "graph_branchpoint_count": 0,
            "graph_isolated_count": 0,
            "endpoint_per_100_skel": 0.0,
            "branchpoint_per_100_skel": 0.0,
            "degree_hist_0": 0.0,
            "degree_hist_1": 0.0,
            "degree_hist_2": 0.0,
            "degree_hist_3": 0.0,
            "degree_hist_4": 0.0,
            "degree_hist_5plus": 0.0,
        }
        base.update(direction_features(skel))
        base.update(stroke_width_features(fg, skel))
        return base

    degree = skeleton_degree(skel)
    deg_vals = degree[skel].astype(np.int32)

    # undirected 8-neighbor edge count
    edge_count = int(deg_vals.sum() // 2)

    endpoint_count = int((deg_vals == 1).sum())
    branch_count = int((deg_vals >= 3).sum())
    isolated_count = int((deg_vals == 0).sum())

    _, skel_cc = ndi.label(skel, structure=np.ones((3, 3), dtype=np.uint8))

    hist_total = max(len(deg_vals), 1)
    out = {
        "skel_pixels": skel_pixels,
        "skel_fraction": float(skel_pixels / max(skel.size, 1)),
        "skel_components": int(skel_cc),

        "graph_nodes": skel_pixels,
        "graph_edges_8n": edge_count,
        "graph_avg_degree": float(deg_vals.mean()) if len(deg_vals) else 0.0,
        "graph_endpoint_count": endpoint_count,
        "graph_branchpoint_count": branch_count,
        "graph_isolated_count": isolated_count,

        "endpoint_per_100_skel": 100.0 * endpoint_count / max(skel_pixels, 1),
        "branchpoint_per_100_skel": 100.0 * branch_count / max(skel_pixels, 1),

        "degree_hist_0": float((deg_vals == 0).sum() / hist_total),
        "degree_hist_1": float((deg_vals == 1).sum() / hist_total),
        "degree_hist_2": float((deg_vals == 2).sum() / hist_total),
        "degree_hist_3": float((deg_vals == 3).sum() / hist_total),
        "degree_hist_4": float((deg_vals == 4).sum() / hist_total),
        "degree_hist_5plus": float((deg_vals >= 5).sum() / hist_total),
    }

    out.update(direction_features(skel))
    out.update(stroke_width_features(fg, skel))
    return out


def warnings_for(fg: np.ndarray, skel: np.ndarray, cc_count: int) -> list[str]:
    warnings = []
    fg_frac = float(fg.mean()) if fg.size else 0.0
    skel_pixels = int(skel.sum())

    if fg_frac < 0.001:
        warnings.append("very_low_foreground")
    if fg_frac > 0.45:
        warnings.append("very_high_foreground")
    if skel_pixels < 5:
        warnings.append("too_few_skeleton_pixels")
    if cc_count > 120:
        warnings.append("many_components")

    return warnings


def process_one(task: tuple[int, dict[str, Any], str, str, str, int, int]) -> dict[str, Any]:
    idx, row, manifest_path_str, dataset, split, min_object_size, sauvola_window = task
    manifest_path = Path(manifest_path_str)

    text = get_text(row)
    img_value = get_image_value(row)
    img_path = resolve_image_path(img_value, manifest_path)

    arr = load_gray(img_path)
    h, w = arr.shape[:2]

    method = DATASET_BINARIZATION.get(dataset, "otsu")
    fg = binarize(arr, method=method, sauvola_window=sauvola_window)

    if min_object_size > 0 and fg.any():
        fg = remove_small_objects(fg, min_size=min_object_size)

    skel = skeletonize(fg)

    out: dict[str, Any] = {
        "sample_id": stable_id(row, dataset, split, idx),
        "dataset": dataset,
        "split": split,
        "image_path": str(img_path),
        "text": text,
        "binarization": method,
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(w / max(h, 1)),
        "text_len": int(len(text)),
        "fg_fraction": float(fg.mean()) if fg.size else 0.0,
    }

    out.update(bbox_features(fg))
    out.update(component_features(fg))
    out.update(graph_features(fg, skel))

    warns = warnings_for(fg, skel, int(out["cc_count"]))
    out["warnings"] = ";".join(warns)
    out["warning_count"] = len(warns)

    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and r[key] != ""]
        return float(statistics.mean(vals)) if vals else 0.0

    return {
        "n": len(rows),
        "warning_rows": sum(1 for r in rows if int(r.get("warning_count", 0)) > 0),
        "warning_rate": sum(1 for r in rows if int(r.get("warning_count", 0)) > 0) / max(len(rows), 1),
        "fg_fraction_mean": mean_of("fg_fraction"),
        "skel_pixels_mean": mean_of("skel_pixels"),
        "endpoint_per_100_skel_mean": mean_of("endpoint_per_100_skel"),
        "branchpoint_per_100_skel_mean": mean_of("branchpoint_per_100_skel"),
        "stroke_width_mean": mean_of("stroke_width_mean"),
    }


def process_manifest(
    manifest_path: Path,
    dataset: str,
    split: str,
    out_dir: Path,
    num_workers: int,
    min_object_size: int,
    sauvola_window: int,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)

    tasks = [
        (idx, row, str(manifest_path), dataset, split, min_object_size, sauvola_window)
        for idx, row in enumerate(rows)
    ]

    good: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if num_workers <= 1:
        for i, task in enumerate(tasks, 1):
            try:
                good.append(process_one(task))
            except Exception as e:
                failures.append({
                    "idx": task[0],
                    "dataset": dataset,
                    "split": split,
                    "error": repr(e),
                })
            if i % 1000 == 0:
                print(f"[{dataset}/{split}] {i}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(process_one, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    good.append(fut.result())
                except Exception as e:
                    failures.append({
                        "dataset": dataset,
                        "split": split,
                        "error": repr(e),
                    })
                if i % 1000 == 0:
                    print(f"[{dataset}/{split}] {i}/{len(tasks)}")

    good.sort(key=lambda r: r["sample_id"])

    stem = f"{dataset}_{split}"
    jsonl_path = out_dir / f"{stem}_features.jsonl"
    csv_path = out_dir / f"{stem}_features.csv"
    failures_path = out_dir / f"{stem}_failures.jsonl"

    write_jsonl(good, jsonl_path)
    write_csv(good, csv_path)
    write_jsonl(failures, failures_path)

    summary = {
        "dataset": dataset,
        "split": split,
        "manifest": str(manifest_path),
        "expected_n": len(rows),
        "features_n": len(good),
        "failures_n": len(failures),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "failures": str(failures_path),
        "stats": summarize_rows(good),
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_dir", default="data/experiments/htr_graph_v1/subsets/tri10k")
    parser.add_argument("--out_dir", default="data/experiments/htr_graph_v1/features/tri10k")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--min_object_size", type=int, default=0)
    parser.add_argument("--sauvola_window", type=int, default=25)
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        "cyrillic_handwriting",
        "hkr_words",
        "school_notebooks_clean",
    ]
    splits = ["train", "val", "test"]

    summaries = []

    for dataset in datasets:
        for split in splits:
            manifest = subset_dir / dataset / f"{split}.jsonl"
            print(f"\n=== extracting {dataset}/{split} ===")
            summary = process_manifest(
                manifest_path=manifest,
                dataset=dataset,
                split=split,
                out_dir=out_dir,
                num_workers=args.num_workers,
                min_object_size=args.min_object_size,
                sauvola_window=args.sauvola_window,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))

    final = {
        "subset_dir": str(subset_dir),
        "out_dir": str(out_dir),
        "num_workers": args.num_workers,
        "min_object_size": args.min_object_size,
        "sauvola_window": args.sauvola_window,
        "items": summaries,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nwrote:", out_dir / "summary.json")


if __name__ == "__main__":
    main()