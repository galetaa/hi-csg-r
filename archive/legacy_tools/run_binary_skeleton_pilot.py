from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from src.datasets.metadata import read_jsonl
from src.graph.binarization import (
    binarize_array,
    load_grayscale_array,
    mask_to_pil,
)
from src.graph.diagnostics import make_binary_skeleton_diagnostics
from src.graph.graph_io import write_json
from src.graph.grid_suppression import suppress_long_grid_lines
from src.graph.skeletonize import skeleton_to_pil, skeletonize_mask

BASE_METHODS = ["otsu", "sauvola", "adaptive_gaussian"]
GRID_METHODS = {"otsu", "sauvola"}
PAGE_LIKE_DATASETS = {"hwr200", "hkr_forms"}


def resize_for_graph_pilot(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    w, h = img.size
    m = max(w, h)

    if m <= max_side:
        return img, 1.0

    scale = max_side / m
    new_size = (
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )
    return img.resize(new_size, Image.Resampling.BICUBIC), scale


def get_effective_max_side(
    dataset: str,
    line_max_side: int,
    page_max_side: int,
) -> int:
    if dataset in PAGE_LIKE_DATASETS:
        return page_max_side
    return line_max_side


def add_common_diagnostic_fields(
    diagnostics: dict,
    *,
    pilot_id: str,
    input_path: Path,
    feature_out: Path,
    binary_path: Path,
    skeleton_path: Path,
    scale: float,
    selection_reason: list,
) -> dict:
    diagnostics["pilot_id"] = pilot_id
    diagnostics["input_image_path"] = str(input_path)
    diagnostics["feature_scaled_path"] = str(feature_out)
    diagnostics["binary_path"] = str(binary_path)
    diagnostics["skeleton_path"] = str(skeleton_path)
    diagnostics["scale"] = scale
    diagnostics["selection_reason"] = selection_reason
    return diagnostics


def process_gridless_variant(
    *,
    record: dict,
    dataset: str,
    level: str,
    pilot_id: str,
    input_path: Path,
    feature_out: Path,
    out_dir: Path,
    arr_shape: tuple[int, int],
    base_method: str,
    binary_mask,
    binary_threshold_info: dict,
    scale: float,
    grid_horizontal_length: int,
    grid_vertical_length: int,
    grid_line_width: int,
) -> tuple[str, dict]:
    method_label = f"{base_method}_gridless"

    grid_result = suppress_long_grid_lines(
        binary_mask,
        horizontal_length=grid_horizontal_length,
        vertical_length=grid_vertical_length,
        line_width=grid_line_width,
    )
    skel_gridless = skeletonize_mask(grid_result.cleaned_mask)

    binary_path = out_dir / f"binary_{method_label}.png"
    grid_mask_path = out_dir / f"grid_mask_{base_method}.png"
    skeleton_path = out_dir / f"skeleton_{method_label}.png"
    diagnostics_path = out_dir / f"diagnostics_{method_label}.json"

    mask_to_pil(grid_result.cleaned_mask).save(binary_path)
    mask_to_pil(grid_result.grid_mask).save(grid_mask_path)
    skeleton_to_pil(skel_gridless.skeleton).save(skeleton_path)

    diagnostics = make_binary_skeleton_diagnostics(
        sample_id=record["sample_id"],
        dataset=dataset,
        level=level,
        image_shape=arr_shape,
        foreground_ratio=float(grid_result.cleaned_mask.mean()),
        skeleton_pixels=skel_gridless.skeleton_pixels,
        foreground_pixels=int(grid_result.cleaned_mask.sum()),
        method=method_label,
        threshold_info={
            **binary_threshold_info,
            "base_method": base_method,
            "grid_removed_pixels": grid_result.removed_pixels,
            "grid_removed_ratio": grid_result.removed_ratio,
            "grid_horizontal_length": grid_horizontal_length,
            "grid_vertical_length": grid_vertical_length,
            "grid_line_width": grid_line_width,
        },
        scale=scale,
        extra={
            "grid_removed_pixels": grid_result.removed_pixels,
            "grid_removed_ratio": grid_result.removed_ratio,
            "grid_mask_path": str(grid_mask_path),
        },
    )

    diagnostics = add_common_diagnostic_fields(
        diagnostics,
        pilot_id=pilot_id,
        input_path=input_path,
        feature_out=feature_out,
        binary_path=binary_path,
        skeleton_path=skeleton_path,
        scale=scale,
        selection_reason=record.get("selection_reason", []),
    )

    write_json(diagnostics, diagnostics_path)

    return method_label, diagnostics


def process_one(
    record: dict,
    out_root: Path,
    *,
    line_max_side: int,
    page_max_side: int,
    grid_suppression: bool,
    grid_horizontal_length: int,
    grid_vertical_length: int,
    grid_line_width: int,
) -> dict:
    pilot_id = record["pilot_id"]
    dataset = record["dataset"]
    level = record["level"]
    input_path = Path(record["input_image_path"])

    out_dir = out_root / dataset / pilot_id
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_max_side = get_effective_max_side(
        dataset=dataset,
        line_max_side=line_max_side,
        page_max_side=page_max_side,
    )

    img = Image.open(input_path).convert("L")
    img_scaled, scale = resize_for_graph_pilot(img, max_side=effective_max_side)

    feature_out = out_dir / "feature_scaled.png"
    img_scaled.save(feature_out)

    arr = load_grayscale_array(str(feature_out))

    method_reports = {}

    for method in BASE_METHODS:
        try:
            binary = binarize_array(arr, method=method)
            skel = skeletonize_mask(binary.mask)

            binary_path = out_dir / f"binary_{method}.png"
            skeleton_path = out_dir / f"skeleton_{method}.png"
            diagnostics_path = out_dir / f"diagnostics_{method}.json"

            mask_to_pil(binary.mask).save(binary_path)
            skeleton_to_pil(skel.skeleton).save(skeleton_path)

            diagnostics = make_binary_skeleton_diagnostics(
                sample_id=record["sample_id"],
                dataset=dataset,
                level=level,
                image_shape=arr.shape,
                foreground_ratio=binary.foreground_ratio,
                skeleton_pixels=skel.skeleton_pixels,
                foreground_pixels=skel.foreground_pixels,
                method=method,
                threshold_info=binary.threshold_info,
                scale=scale,
            )

            diagnostics = add_common_diagnostic_fields(
                diagnostics,
                pilot_id=pilot_id,
                input_path=input_path,
                feature_out=feature_out,
                binary_path=binary_path,
                skeleton_path=skeleton_path,
                scale=scale,
                selection_reason=record.get("selection_reason", []),
            )

            write_json(diagnostics, diagnostics_path)
            method_reports[method] = diagnostics

            if grid_suppression and dataset in PAGE_LIKE_DATASETS and method in GRID_METHODS:
                try:
                    method_label, gridless_diagnostics = process_gridless_variant(
                        record=record,
                        dataset=dataset,
                        level=level,
                        pilot_id=pilot_id,
                        input_path=input_path,
                        feature_out=feature_out,
                        out_dir=out_dir,
                        arr_shape=arr.shape,
                        base_method=method,
                        binary_mask=binary.mask,
                        binary_threshold_info=binary.threshold_info,
                        scale=scale,
                        grid_horizontal_length=grid_horizontal_length,
                        grid_vertical_length=grid_vertical_length,
                        grid_line_width=grid_line_width,
                    )
                    method_reports[method_label] = gridless_diagnostics

                except Exception as exc:
                    method_label = f"{method}_gridless"
                    method_reports[method_label] = {
                        "pilot_id": pilot_id,
                        "sample_id": record["sample_id"],
                        "dataset": dataset,
                        "level": level,
                        "binarization": method_label,
                        "warnings": ["grid_suppression_failed"],
                        "error": repr(exc),
                    }

        except Exception as exc:
            method_reports[method] = {
                "pilot_id": pilot_id,
                "sample_id": record["sample_id"],
                "dataset": dataset,
                "level": level,
                "binarization": method,
                "warnings": ["binarization_or_skeleton_failed"],
                "error": repr(exc),
            }

    summary = {
        "pilot_id": pilot_id,
        "sample_id": record["sample_id"],
        "dataset": dataset,
        "level": level,
        "input_image_path": str(input_path),
        "output_dir": str(out_dir),
        "scale": scale,
        "effective_max_side": effective_max_side,
        "page_like": dataset in PAGE_LIKE_DATASETS,
        "grid_suppression_enabled": grid_suppression,
        "methods": method_reports,
    }

    write_json(summary, out_dir / "binary_skeleton_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", default="data/pilot/graph_pilot_v1.jsonl")
    parser.add_argument("--out_dir", default="outputs/graph_pilot")

    # New preferred args.
    parser.add_argument("--line_max_side", type=int, default=10000)
    parser.add_argument("--page_max_side", type=int, default=2400)

    # Backward-compatible legacy arg. If passed, it overrides both.
    parser.add_argument("--max_side", type=int, default=None)

    parser.add_argument("--grid_suppression", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--grid_horizontal_length", type=int, default=35)
    parser.add_argument("--grid_vertical_length", type=int, default=35)
    parser.add_argument("--grid_line_width", type=int, default=1)

    args = parser.parse_args()

    if args.max_side is not None:
        line_max_side = args.max_side
        page_max_side = args.max_side
    else:
        line_max_side = args.line_max_side
        page_max_side = args.page_max_side

    records = read_jsonl(args.pilot)

    if args.limit is not None:
        records = records[: args.limit]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []

    for idx, record in enumerate(records, start=1):
        summary = process_one(
            record,
            out_root,
            line_max_side=line_max_side,
            page_max_side=page_max_side,
            grid_suppression=args.grid_suppression,
            grid_horizontal_length=args.grid_horizontal_length,
            grid_vertical_length=args.grid_vertical_length,
            grid_line_width=args.grid_line_width,
        )
        summaries.append(summary)

        if idx % 25 == 0:
            print(f"processed {idx}/{len(records)}")

    all_summary_path = out_root / "binary_skeleton_pilot_summary.json"
    all_summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {all_summary_path}")
    print(f"Processed records: {len(summaries)}")
    print(f"line_max_side={line_max_side}")
    print(f"page_max_side={page_max_side}")
    print(f"grid_suppression={args.grid_suppression}")


if __name__ == "__main__":
    main()
