from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp_bbox(
    bbox: list[float],
    *,
    width: int,
    height: int,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [
        int(round(float(value)))
        for value in bbox[:4]
    ]

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)

    if x1 <= x0:
        x1 = min(width, x0 + 1)

    if y1 <= y0:
        y1 = min(height, y0 + 1)

    return x0, y0, x1, y1


def resize_keep_height(
    image: Image.Image,
    *,
    target_height: int,
) -> Image.Image:
    scale = target_height / max(image.height, 1)

    target_width = max(
        1,
        int(round(image.width * scale)),
    )

    return image.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )


def render_one(
    row: dict[str, Any],
    *,
    source_images_root: Path,
    images_out_dir: Path,
    target_height: int,
    pad_x_frac: float,
    pad_y_frac: float,
) -> dict[str, Any]:
    source_image_file = str(row["source_image_file"])
    source_path = source_images_root / source_image_file

    if not source_path.exists():
        matches = list(
            source_images_root.rglob(source_image_file)
        )

        if len(matches) == 1:
            source_path = matches[0]
        else:
            raise FileNotFoundError(
                f"{source_image_file}; matches={matches[:5]}"
            )

    with Image.open(source_path) as src:
        src = src.convert("RGB")

        bbox = row["bbox_xyxy"]

        line_height = max(
            1,
            int(round(float(bbox[3]) - float(bbox[1]))),
        )

        pad_x = int(round(line_height * pad_x_frac))
        pad_y = int(round(line_height * pad_y_frac))

        crop_box = clamp_bbox(
            bbox,
            width=src.width,
            height=src.height,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        crop = src.crop(crop_box)
        crop = resize_keep_height(
            crop,
            target_height=target_height,
        )

        rel_path = Path("images") / f"{row['sample_id']}.png"
        out_path = images_out_dir.parent / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        crop.save(out_path)

    out = dict(row)
    out["image_path"] = str(out_path)
    out["render_metadata"] = {
        "renderer": "school_full_line_raw_rgb_v1",
        "source_image_file": source_image_file,
        "source_path": str(source_path),
        "source_bbox_xyxy": row["bbox_xyxy"],
        "crop_bbox_xyxy": list(crop_box),
        "target_height": target_height,
        "pad_x_frac": pad_x_frac,
        "pad_y_frac": pad_y_frac,
    }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--school_raw_dir", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--target_height", type=int, default=64)
    parser.add_argument("--pad_x_frac", type=float, default=0.35)
    parser.add_argument("--pad_y_frac", type=float, default=0.25)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    out_jsonl = Path(args.out_jsonl)

    source_images_root = (
        Path(args.school_raw_dir)
        / "images"
        / "images"
    )

    if not source_images_root.exists():
        raise FileNotFoundError(source_images_root)

    rows = read_jsonl(input_path)

    images_out_dir = out_jsonl.parent / "images"
    rendered = []
    failures = []

    for index, row in enumerate(rows, start=1):
        try:
            rendered.append(
                render_one(
                    row,
                    source_images_root=source_images_root,
                    images_out_dir=images_out_dir,
                    target_height=args.target_height,
                    pad_x_frac=args.pad_x_frac,
                    pad_y_frac=args.pad_y_frac,
                )
            )
        except Exception as exc:
            failures.append({
                "index": index - 1,
                "sample_id": row.get("sample_id"),
                "error": repr(exc),
            })

        if index % 500 == 0:
            print(f"{index}/{len(rows)}")

    if failures:
        fail_path = out_jsonl.with_suffix(".failures.json")
        fail_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"render failures={len(failures)}; first={failures[:3]}"
        )

    write_jsonl(rendered, out_jsonl)

    summary = {
        "input_jsonl": str(input_path),
        "out_jsonl": str(out_jsonl),
        "n": len(rendered),
        "target_height": args.target_height,
        "pad_x_frac": args.pad_x_frac,
        "pad_y_frac": args.pad_y_frac,
        "renderer": "school_full_line_raw_rgb_v1",
    }

    (out_jsonl.parent / "render_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
