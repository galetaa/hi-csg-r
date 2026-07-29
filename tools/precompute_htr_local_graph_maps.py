from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import remove_small_objects, skeletonize


IMAGE_KEYS = ["image_path", "path", "crop_path", "file_path", "img_path", "image"]

DATASET_BINARIZATION = {
    "cyrillic_handwriting": "otsu",
    "hkr_words": "otsu",
    "school_notebooks_clean": "sauvola",
}


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


def get_image_value(row: dict[str, Any]) -> str:
    for key in IMAGE_KEYS:
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    raise KeyError(f"No image path key found: {sorted(row.keys())}")


def resolve_image_path(value: str, manifest_path: Path) -> Path:
    p = Path(value)
    candidates = [p, Path.cwd() / p, manifest_path.parent / p]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(value)


def binarize(arr: np.ndarray, method: str, sauvola_window: int) -> np.ndarray:
    if method == "otsu":
        try:
            t = threshold_otsu(arr)
        except ValueError:
            t = 127
        fg = arr < t

    elif method == "sauvola":
        min_dim = min(arr.shape)
        if min_dim >= 7:
            window = sauvola_window
            if window % 2 == 0:
                window += 1
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
        raise ValueError(method)

    if fg.mean() > 0.5:
        fg = ~fg

    return fg.astype(bool)


def process_one(task):
    idx, row, manifest_path_str, dataset, split, maps_root_str, min_object_size, sauvola_window = task

    manifest_path = Path(manifest_path_str)
    maps_root = Path(maps_root_str)

    img_path = resolve_image_path(get_image_value(row), manifest_path)

    arr_u8 = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    method_dataset = str(row.get("dataset") or row.get("source_dataset") or dataset)
    if method_dataset == "mixed":
        method_dataset = str(row.get("source_dataset") or row.get("dataset") or "mixed")

    method = DATASET_BINARIZATION.get(method_dataset, "otsu")

    fg = binarize(arr_u8, method=method, sauvola_window=sauvola_window)

    if min_object_size > 0 and fg.any():
        fg = remove_small_objects(fg, min_size=min_object_size)

    skel = skeletonize(fg)

    dist = ndi.distance_transform_edt(fg).astype(np.float32)
    if dist.max() > 0:
        dist = dist / dist.max()

    sid = str(row.get("sample_id", f"{dataset}_{split}_{idx:08d}"))

    out_dir = maps_root / dataset / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sid}.npz"

    np.savez_compressed(
        out_path,
        fg=fg.astype(np.uint8),
        skel=skel.astype(np.uint8),
        dist=dist.astype(np.float32),
        binarization=np.array(method),
    )

    new_row = dict(row)
    new_row["local_graph_npz"] = str(out_path)
    new_row["local_graph_channels"] = ["fg", "skel", "dist"]
    new_row["local_graph_binarization"] = method
    new_row["local_graph_method_dataset"] = method_dataset

    return new_row


def process_manifest(
    manifest_path: Path,
    dataset: str,
    split: str,
    out_manifest: Path,
    maps_root: Path,
    num_workers: int,
    min_object_size: int,
    sauvola_window: int,
):
    rows = read_jsonl(manifest_path)

    tasks = [
        (
            idx,
            row,
            str(manifest_path),
            dataset,
            split,
            str(maps_root),
            min_object_size,
            sauvola_window,
        )
        for idx, row in enumerate(rows)
    ]

    out_rows = []
    failures = []

    if num_workers <= 1:
        for i, task in enumerate(tasks, 1):
            try:
                out_rows.append(process_one(task))
            except Exception as e:
                failures.append({"idx": task[0], "error": repr(e)})
            if i % 1000 == 0:
                print(f"[{dataset}/{split}] {i}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(process_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    out_rows.append(fut.result())
                except Exception as e:
                    failures.append({"error": repr(e)})
                if i % 1000 == 0:
                    print(f"[{dataset}/{split}] {i}/{len(tasks)}")

    # Preserve deterministic order by sample_id after multiprocessing.
    out_rows.sort(key=lambda r: str(r.get("sample_id", "")))

    write_jsonl(out_rows, out_manifest)
    write_jsonl(failures, out_manifest.with_suffix(".failures.jsonl"))

    return {
        "dataset": dataset,
        "split": split,
        "source_manifest": str(manifest_path),
        "out_manifest": str(out_manifest),
        "expected_n": len(rows),
        "written_n": len(out_rows),
        "failures_n": len(failures),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_ready_dir", default="data/experiments/htr_graph_v1/graph_ready/tri10k_mixed")
    parser.add_argument("--out_dir", default="data/experiments/htr_graph_v1/local_graph_ready/tri10k_mixed")
    parser.add_argument("--maps_root", default="data/experiments/htr_graph_v1/local_maps/tri10k_mixed")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--min_object_size", type=int, default=0)
    parser.add_argument("--sauvola_window", type=int, default=25)
    args = parser.parse_args()

    graph_ready = Path(args.graph_ready_dir)
    out_dir = Path(args.out_dir)
    maps_root = Path(args.maps_root)

    datasets = ["cyrillic_handwriting", "hkr_words", "school_notebooks_clean"]
    splits = ["train", "val", "test"]

    summaries = []

    # Combined manifests.
    for split in splits:
        print(f"\n=== combined/{split} ===")
        summaries.append(
            process_manifest(
                manifest_path=graph_ready / f"{split}.jsonl",
                dataset="mixed",
                split=split,
                out_manifest=out_dir / f"{split}.jsonl",
                maps_root=maps_root,
                num_workers=args.num_workers,
                min_object_size=args.min_object_size,
                sauvola_window=args.sauvola_window,
            )
        )

    # Per-dataset eval manifests.
    for dataset in datasets:
        for split in splits:
            print(f"\n=== {dataset}/{split} ===")
            summaries.append(
                process_manifest(
                    manifest_path=graph_ready / "eval_manifests" / f"{dataset}_{split}.jsonl",
                    dataset=dataset,
                    split=split,
                    out_manifest=out_dir / "eval_manifests" / f"{dataset}_{split}.jsonl",
                    maps_root=maps_root,
                    num_workers=args.num_workers,
                    min_object_size=args.min_object_size,
                    sauvola_window=args.sauvola_window,
                )
            )

    # Copy vocab.
    vocab_src = graph_ready / "vocab.json"
    vocab_dst = out_dir / "vocab.json"
    vocab_dst.parent.mkdir(parents=True, exist_ok=True)
    vocab_dst.write_text(vocab_src.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "graph_ready_dir": str(graph_ready),
        "out_dir": str(out_dir),
        "maps_root": str(maps_root),
        "items": summaries,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()