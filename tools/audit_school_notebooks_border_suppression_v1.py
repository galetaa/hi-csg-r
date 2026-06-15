from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage.filters import threshold_sauvola
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def sauvola_fg(arr_u8: np.ndarray, window: int = 25) -> np.ndarray:
    ink = 255 - arr_u8

    h, w = ink.shape
    win = min(window, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    thr = threshold_sauvola(ink, window_size=win)
    return ink > thr


def component_touches_border(bbox: tuple[int, int, int, int], h: int, w: int, margin: int) -> bool:
    y0, x0, y1, x1 = bbox
    return x0 <= margin or y0 <= margin or x1 >= w - margin or y1 >= h - margin


def suppress_border_artifacts(
    fg: np.ndarray,
    *,
    margin: int = 3,
    min_area_frac: float = 0.015,
    long_frac: float = 0.35,
    high_fill_frac: float = 0.45,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Removes only large dense foreground components touching the crop border.

    Intended for school_notebooks border/crop artifacts, not as a universal rule.
    """
    h, w = fg.shape
    lab = label(fg, connectivity=2)

    cleaned = fg.copy()
    removed_components = 0
    removed_area = 0
    kept_border_components = 0

    for prop in regionprops(lab):
        y0, x0, y1, x1 = prop.bbox
        area = int(prop.area)
        bh = y1 - y0
        bw = x1 - x0

        touches = component_touches_border(prop.bbox, h, w, margin)
        if not touches:
            continue

        area_frac = area / max(h * w, 1)
        width_frac = bw / max(w, 1)
        height_frac = bh / max(h, 1)
        fill_frac = area / max(bw * bh, 1)

        is_large_border_artifact = (
            area_frac >= min_area_frac
            and (
                width_frac >= long_frac
                or height_frac >= long_frac
                or fill_frac >= high_fill_frac
            )
        )

        if is_large_border_artifact:
            cleaned[lab == prop.label] = False
            removed_components += 1
            removed_area += area
        else:
            kept_border_components += 1

    stats = {
        "removed_components": removed_components,
        "removed_area": removed_area,
        "removed_area_frac": removed_area / max(h * w, 1),
        "kept_border_components": kept_border_components,
    }

    return cleaned, stats


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def fg_to_img(fg: np.ndarray) -> Image.Image:
    return Image.fromarray((255 - fg.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")


def skel_to_img(skel: np.ndarray) -> Image.Image:
    return Image.fromarray((255 - skel.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")


def overlay_skel(orig: Image.Image, skel: np.ndarray) -> Image.Image:
    arr = np.asarray(orig.convert("RGB")).copy()
    arr[skel] = np.array([255, 0, 0], dtype=np.uint8)
    return Image.fromarray(arr)


def draw_card(row: dict[str, Any], result: dict[str, Any], *, view_w: int, view_h: int) -> Image.Image:
    img_path = Path(row["image_path"])
    orig_l = ImageOps.autocontrast(Image.open(img_path).convert("L"))
    arr = np.asarray(orig_l, dtype=np.uint8)

    old_fg = sauvola_fg(arr)
    new_fg, _ = suppress_border_artifacts(old_fg)

    old_skel = skeletonize(old_fg)
    new_skel = skeletonize(new_fg)

    views = [
        ("original", orig_l.convert("RGB")),
        ("old binary", fg_to_img(old_fg)),
        ("new binary", fg_to_img(new_fg)),
        ("old skeleton", skel_to_img(old_skel)),
        ("new skeleton", skel_to_img(new_skel)),
        ("new overlay", overlay_skel(orig_l.convert("RGB"), new_skel)),
    ]

    header_h = 95
    label_h = 20
    card_w = 3 * view_w
    card_h = header_h + 2 * (view_h + label_h)

    card = Image.new("RGB", (card_w, card_h), "white")
    draw = ImageDraw.Draw(card)

    header = [
        f"id: {row.get('sample_id')}",
        f"cell: {row.get('audit_cell')} | CER={safe_float(row.get('cer')):.3f} | risk={safe_float(row.get('structural_risk_score')):.3f}",
        f"target: {row.get('target')}",
        f"pred: {row.get('pred')}",
        f"removed_components={result['removed_components']} removed_area_frac={result['removed_area_frac']:.4f}",
    ]

    y = 6
    for line in header:
        draw.text((8, y), str(line)[:150], fill="black")
        y += 16

    for i, (title, im) in enumerate(views):
        col = i % 3
        row_i = i // 3
        x = col * view_w
        y = header_h + row_i * (view_h + label_h)

        draw.text((x + 6, y), title, fill="black")
        card.paste(fit(im, view_w, view_h), (x, y + label_h))

    draw.rectangle((0, 0, card_w - 1, card_h - 1), outline="black")
    return card


def make_contact_sheet(rows: list[dict[str, Any]], results: list[dict[str, Any]], path: Path) -> None:
    view_w = 300
    view_h = 95

    cards = [
        draw_card(row, result, view_w=view_w, view_h=view_h)
        for row, result in zip(rows, results)
    ]

    if not cards:
        return

    cols = 1
    sheet_w = cards[0].width
    sheet_h = sum(c.height for c in cards)

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

    y = 0
    for c in cards:
        sheet.paste(c, (0, y))
        y += c.height

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def process_row(row: dict[str, Any]) -> dict[str, Any]:
    img_path = Path(row["image_path"])
    orig_l = Image.open(img_path).convert("L")
    arr = np.asarray(orig_l, dtype=np.uint8)

    old_fg = sauvola_fg(arr)
    new_fg, stats = suppress_border_artifacts(old_fg)

    old_skel = skeletonize(old_fg)
    new_skel = skeletonize(new_fg)

    out = {
        "sample_id": row.get("sample_id"),
        "audit_cell": row.get("audit_cell"),
        "dataset": row.get("dataset"),
        "cer": row.get("cer"),
        "risk": row.get("structural_risk_score"),
        "image_path": row.get("image_path"),
        "old_fg_fraction": float(old_fg.mean()),
        "new_fg_fraction": float(new_fg.mean()),
        "fg_fraction_delta": float(new_fg.mean() - old_fg.mean()),
        "old_skel_fraction": float(old_skel.mean()),
        "new_skel_fraction": float(new_skel.mean()),
        "skel_fraction_delta": float(new_skel.mean() - old_skel.mean()),
        **stats,
    }

    return out


def make_md(results: list[dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("# School notebooks border suppression audit — v1")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This is a preprocessing sanity check for school-notebooks samples where manual audit "
        "identified crop/border artifacts being binarized as foreground."
    )
    lines.append("")

    if results:
        lines.append("## 2. Aggregate")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| n | {len(results)} |")
        lines.append(f"| mean removed components | {np.mean([r['removed_components'] for r in results]):.3f} |")
        lines.append(f"| mean removed area frac | {np.mean([r['removed_area_frac'] for r in results]):.5f} |")
        lines.append(f"| mean old fg fraction | {np.mean([r['old_fg_fraction'] for r in results]):.5f} |")
        lines.append(f"| mean new fg fraction | {np.mean([r['new_fg_fraction'] for r in results]):.5f} |")
        lines.append(f"| mean old skeleton fraction | {np.mean([r['old_skel_fraction'] for r in results]):.5f} |")
        lines.append(f"| mean new skeleton fraction | {np.mean([r['new_skel_fraction'] for r in results]):.5f} |")
        lines.append("")

    lines.append("## 3. Interpretation rule")
    lines.append("")
    lines.append(
        "If the new binary/skeleton removes crop-border components without erasing real handwriting, "
        "then school-notebooks failures should be reported as fixable preprocessing artifacts."
    )
    lines.append("")
    lines.append(
        "If it erases real handwriting, the rule is too aggressive and should not be integrated."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="school_notebooks_clean")
    args = parser.parse_args()

    rows_all = read_csv(Path(args.annotations_csv))
    rows = [r for r in rows_all if r.get("dataset") == args.dataset]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [process_row(r) for r in rows]

    write_csv(results, out_dir / "border_suppression_metrics.csv")
    make_contact_sheet(rows, results, out_dir / "border_suppression_contact_sheet.png")
    make_md(results, out_dir / "border_suppression_report.md")

    summary = {
        "annotations_csv": args.annotations_csv,
        "dataset": args.dataset,
        "n": len(rows),
        "outputs": {
            "metrics_csv": str(out_dir / "border_suppression_metrics.csv"),
            "contact_sheet_png": str(out_dir / "border_suppression_contact_sheet.png"),
            "report_md": str(out_dir / "border_suppression_report.md"),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
