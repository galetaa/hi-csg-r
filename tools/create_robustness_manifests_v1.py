from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


IMAGE_KEYS = ["image_path", "path", "crop_path", "file_path", "img_path", "image"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def get_image_value(row: dict[str, Any]) -> str:
    for key in IMAGE_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"No image path key found. Keys: {sorted(row.keys())}")


def resolve_image_path(value: str, manifest_path: Path) -> Path:
    p = Path(value)
    candidates = [
        p,
        Path.cwd() / p,
        manifest_path.parent / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def stable_distorted_id(sample_id: str, distortion: str, level: str) -> str:
    raw = f"{sample_id}|{distortion}|{level}"
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{sample_id}__{distortion}_{level}_{suffix}"


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def save_gray(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def gaussian_blur(arr: np.ndarray, ksize: int) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(arr, (ksize, ksize), 0)


def add_gaussian_noise(arr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=arr.shape)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def change_contrast(arr: np.ndarray, alpha: float) -> np.ndarray:
    # alpha < 1.0 lowers contrast around mid-gray.
    out = (arr.astype(np.float32) - 127.5) * alpha + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def ink_morphology(arr: np.ndarray, op: str, ksize: int) -> np.ndarray:
    # Work on ink intensity: black ink in grayscale becomes high signal here.
    ink = 255 - arr
    kernel = np.ones((ksize, ksize), dtype=np.uint8)

    if op == "erode_ink":
        ink2 = cv2.erode(ink, kernel, iterations=1)
    elif op == "dilate_ink":
        ink2 = cv2.dilate(ink, kernel, iterations=1)
    else:
        raise ValueError(op)

    return 255 - ink2


def distort(arr: np.ndarray, distortion: str, level: str, rng: np.random.Generator) -> np.ndarray:
    if distortion == "blur":
        ksize = {"mild": 3, "medium": 5, "strong": 7}[level]
        return gaussian_blur(arr, ksize)

    if distortion == "noise":
        sigma = {"mild": 8.0, "medium": 16.0, "strong": 24.0}[level]
        return add_gaussian_noise(arr, sigma, rng)

    if distortion == "low_contrast":
        alpha = {"mild": 0.75, "medium": 0.55, "strong": 0.40}[level]
        return change_contrast(arr, alpha)

    if distortion == "thin_strokes":
        ksize = {"mild": 2, "medium": 2, "strong": 3}[level]
        return ink_morphology(arr, "erode_ink", ksize)

    if distortion == "thick_strokes":
        ksize = {"mild": 2, "medium": 2, "strong": 3}[level]
        return ink_morphology(arr, "dilate_ink", ksize)

    raise ValueError(f"Unknown distortion: {distortion}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--distortions",
        default="blur,noise,low_contrast,thin_strokes,thick_strokes",
    )
    parser.add_argument(
        "--levels",
        default="mild,medium,strong",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    rows = read_jsonl(manifest_path)
    if args.limit is not None:
        rng_py = random.Random(args.seed)
        rows = list(rows)
        rng_py.shuffle(rows)
        rows = rows[: args.limit]

    distortions = [x.strip() for x in args.distortions.split(",") if x.strip()]
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]

    rng = np.random.default_rng(args.seed)

    summary: dict[str, Any] = {
        "source_manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "source_n": len(rows),
        "distortions": distortions,
        "levels": levels,
        "items": [],
    }

    for distortion_name in distortions:
        for level in levels:
            out_rows: list[dict[str, Any]] = []
            condition = f"{distortion_name}_{level}"
            images_dir = out_dir / "images" / condition

            for i, row in enumerate(rows):
                original_value = get_image_value(row)
                original_path = resolve_image_path(original_value, manifest_path)
                arr = load_gray(original_path)

                original_sample_id = str(row.get("sample_id", row.get("id", i)))
                new_sample_id = stable_distorted_id(
                    original_sample_id,
                    distortion_name,
                    level,
                )

                distorted = distort(arr, distortion_name, level, rng)
                out_image_path = images_dir / f"{new_sample_id}.png"
                save_gray(distorted, out_image_path)

                new_row = dict(row)
                new_row["sample_id"] = new_sample_id
                new_row["clean_sample_id"] = original_sample_id
                new_row["image_path"] = str(out_image_path)
                new_row["original_image_path"] = str(original_path)
                new_row["distortion"] = distortion_name
                new_row["distortion_level"] = level
                new_row["robustness_condition"] = condition
                new_row["split"] = args.split

                if args.dataset_name is not None:
                    new_row["dataset"] = args.dataset_name

                out_rows.append(new_row)

            out_manifest = out_dir / "manifests" / f"{condition}.jsonl"
            write_jsonl(out_rows, out_manifest)

            summary["items"].append(
                {
                    "condition": condition,
                    "distortion": distortion_name,
                    "level": level,
                    "n": len(out_rows),
                    "manifest": str(out_manifest),
                }
            )

            print(
                json.dumps(
                    {
                        "condition": condition,
                        "n": len(out_rows),
                        "manifest": str(out_manifest),
                    },
                    ensure_ascii=False,
                )
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote:", summary_path)


if __name__ == "__main__":
    main()
