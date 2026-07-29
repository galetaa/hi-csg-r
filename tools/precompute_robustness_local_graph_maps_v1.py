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


DATASET_BINARIZATION = {
    "cyrillic_handwriting": "otsu",
    "hkr_words": "otsu",
    "school_notebooks_clean": "sauvola",
    "iam": "otsu",
    "iam_lines": "otsu",
    "iam_words": "otsu",
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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(value: str, manifest_path: Path) -> Path:
    p = Path(value)
    candidates = [
        p,
        Path.cwd() / p,
        manifest_path.parent / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(value)


def get_image_path(row: dict[str, Any], manifest_path: Path) -> Path:
    for key in ["image_path", "path", "img_path"]:
        value = row.get(key)
        if isinstance(value, str) and value:
            return resolve_path(value, manifest_path)

    raise KeyError(f"Row has no image path field. Keys: {sorted(row.keys())}")


def get_sample_id(row: dict[str, Any], idx: int) -> str:
    sid = row.get("clean_sample_id") or row.get("sample_id") or f"sample_{idx:08d}"
    sid = str(sid)
    return (
        sid.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def method_for_row(row: dict[str, Any]) -> str:
    dataset = str(
        row.get("source_dataset")
        or row.get("dataset")
        or row.get("local_graph_method_dataset")
        or ""
    )

    if dataset == "mixed":
        dataset = str(row.get("source_dataset") or "")

    return DATASET_BINARIZATION.get(dataset, "otsu")


def safe_sauvola_window(h: int, w: int, requested: int) -> int:
    m = min(h, w, requested)
    if m < 3:
        return 3
    if m % 2 == 0:
        m -= 1
    return max(m, 3)


def binarize_ink(arr_u8: np.ndarray, method: str, sauvola_window: int) -> np.ndarray:
    # Convert white-background image into foreground-high ink image.
    ink = 255 - arr_u8

    if method == "sauvola":
        h, w = ink.shape
        win = safe_sauvola_window(h, w, sauvola_window)
        thr = threshold_sauvola(ink, window_size=win)
        fg = ink > thr
    elif method == "otsu":
        if ink.max() == ink.min():
            fg = np.zeros_like(ink, dtype=bool)
        else:
            thr = threshold_otsu(ink)
            fg = ink > thr
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return fg.astype(bool)


def process_one(task: tuple[int, dict[str, Any], str, str, str, int, int]) -> dict[str, Any]:
    idx, row, manifest_path_s, condition, maps_root_s, min_object_size, sauvola_window = task

    manifest_path = Path(manifest_path_s)
    maps_root = Path(maps_root_s)

    img_path = get_image_path(row, manifest_path)
    arr_u8 = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)

    method = method_for_row(row)
    fg = binarize_ink(arr_u8, method=method, sauvola_window=sauvola_window)

    if min_object_size > 0 and fg.any():
        fg = remove_small_objects(fg, min_size=min_object_size)

    skel = skeletonize(fg)

    dist = ndi.distance_transform_edt(fg).astype(np.float32)
    if float(dist.max()) > 0:
        dist = dist / float(dist.max())

    sid = get_sample_id(row, idx)

    out_dir = maps_root / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sid}.npz"

    np.savez_compressed(
        out_path,
        fg=fg.astype(np.uint8),
        skel=skel.astype(np.uint8),
        dist=dist.astype(np.float32),
        binarization=np.array(method),
        condition=np.array(condition),
    )

    new_row = dict(row)
    new_row["local_graph_npz"] = str(out_path)
    new_row["local_graph_channels"] = ["fg", "skel", "dist"]
    new_row["local_graph_binarization"] = method
    new_row["local_graph_source"] = "distorted_image" if condition != "clean" else "clean_image"
    new_row["local_graph_condition"] = condition

    return new_row


def process_manifest(
    *,
    manifest_path: Path,
    condition: str,
    out_manifest: Path,
    maps_root: Path,
    num_workers: int,
    min_object_size: int,
    sauvola_window: int,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)

    tasks = [
        (
            idx,
            row,
            str(manifest_path),
            condition,
            str(maps_root),
            min_object_size,
            sauvola_window,
        )
        for idx, row in enumerate(rows)
    ]

    out_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if num_workers <= 1:
        for i, task in enumerate(tasks, 1):
            try:
                out_rows.append(process_one(task))
            except Exception as e:
                failures.append({"idx": task[0], "sample_id": task[1].get("sample_id"), "error": repr(e)})

            if i % 1000 == 0:
                print(f"[{condition}] {i}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(process_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    out_rows.append(fut.result())
                except Exception as e:
                    failures.append({"error": repr(e)})

                if i % 1000 == 0:
                    print(f"[{condition}] {i}/{len(tasks)}")

    out_rows.sort(key=lambda r: str(r.get("sample_id", "")))

    write_jsonl(out_rows, out_manifest)
    write_jsonl(failures, out_manifest.with_suffix(".failures.jsonl"))

    return {
        "condition": condition,
        "source_manifest": str(manifest_path),
        "out_manifest": str(out_manifest),
        "expected_n": len(rows),
        "written_n": len(out_rows),
        "failures_n": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_manifest", required=True)
    parser.add_argument("--conditions_file", required=True)
    parser.add_argument("--graph_ready_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--maps_root", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--min_object_size", type=int, default=0)
    parser.add_argument("--sauvola_window", type=int, default=25)
    args = parser.parse_args()

    clean_manifest = Path(args.clean_manifest)
    conditions_file = Path(args.conditions_file)
    graph_ready_dir = Path(args.graph_ready_dir)
    out_dir = Path(args.out_dir)
    maps_root = Path(args.maps_root)

    conditions = [
        line.strip()
        for line in conditions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    summaries = []

    print("\n=== clean ===")
    summaries.append(
        process_manifest(
            manifest_path=clean_manifest,
            condition="clean",
            out_manifest=out_dir / "clean.jsonl",
            maps_root=maps_root,
            num_workers=args.num_workers,
            min_object_size=args.min_object_size,
            sauvola_window=args.sauvola_window,
        )
    )

    for cond in conditions:
        manifest = graph_ready_dir / f"{cond}.jsonl"

        print(f"\n=== {cond} ===")
        summaries.append(
            process_manifest(
                manifest_path=manifest,
                condition=cond,
                out_manifest=out_dir / f"{cond}.jsonl",
                maps_root=maps_root,
                num_workers=args.num_workers,
                min_object_size=args.min_object_size,
                sauvola_window=args.sauvola_window,
            )
        )

    summary = {
        "clean_manifest": str(clean_manifest),
        "conditions_file": str(conditions_file),
        "graph_ready_dir": str(graph_ready_dir),
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