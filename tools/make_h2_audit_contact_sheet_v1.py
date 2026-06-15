from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import skeletonize


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def binarize(arr: np.ndarray, dataset: str) -> np.ndarray:
    ink = 255 - arr

    if dataset == "school_notebooks_clean":
        h, w = ink.shape
        win = min(25, h, w)
        if win % 2 == 0:
            win -= 1
        win = max(win, 3)
        thr = threshold_sauvola(ink, window_size=win)
        return ink > thr

    if ink.max() == ink.min():
        return np.zeros_like(ink, dtype=bool)

    thr = threshold_otsu(ink)
    return ink > thr


def load_views(row: dict[str, Any], thumb_w: int, thumb_h: int) -> tuple[Image.Image, Image.Image, Image.Image]:
    img_path = Path(row["image_path"])
    img = Image.open(img_path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)

    dataset = str(row.get("dataset") or "")
    fg = binarize(arr, dataset)
    skel = skeletonize(fg)

    orig = ImageOps.autocontrast(img).convert("RGB")

    fg_img = Image.fromarray((255 - fg.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")
    skel_img = Image.fromarray((255 - skel.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")

    views = []
    for v in [orig, fg_img, skel_img]:
        v.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        x = (thumb_w - v.width) // 2
        y = (thumb_h - v.height) // 2
        canvas.paste(v, (x, y))
        views.append(canvas)

    return views[0], views[1], views[2]


def draw_card(row: dict[str, Any], thumb_w: int, thumb_h: int, card_w: int, card_h: int) -> Image.Image:
    card = Image.new("RGB", (card_w, card_h), "white")
    draw = ImageDraw.Draw(card)

    try:
        orig, fg, skel = load_views(row, thumb_w, thumb_h)
    except Exception as e:
        draw.text((10, 10), f"IMAGE ERROR: {e}", fill="black")
        return card

    x0 = 10
    y0 = 10

    card.paste(orig, (x0, y0))
    card.paste(fg, (x0 + thumb_w + 8, y0))
    card.paste(skel, (x0 + 2 * (thumb_w + 8), y0))

    text_y = y0 + thumb_h + 8
    lines = [
        f"{row.get('audit_cell', '')}",
        f"id: {row.get('sample_id', '')}",
        f"dataset: {row.get('dataset', '')} | {row.get('level', '')} | {row.get('category', '')}",
        f"CER={safe_float(row.get('cer')):.3f} risk={safe_float(row.get('structural_risk_score')):.3f}",
        f"T: {row.get('target', '')}",
        f"P: {row.get('pred', '')}",
    ]

    for line in lines:
        draw.text((10, text_y), line[:95], fill="black")
        text_y += 14

    draw.rectangle((0, 0, card_w - 1, card_h - 1), outline="black")
    return card


def make_cell_sheet(rows: list[dict[str, Any]], out_path: Path, cols: int, thumb_w: int, thumb_h: int) -> None:
    card_w = 3 * thumb_w + 2 * 8 + 20
    card_h = thumb_h + 110

    sheet_w = cols * card_w
    sheet_h = ((len(rows) + cols - 1) // cols) * card_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

    for i, row in enumerate(rows):
        card = draw_card(row, thumb_w, thumb_h, card_w, card_h)
        x = (i % cols) * card_w
        y = (i // cols) * card_h
        sheet.paste(card, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_html(rows: list[dict[str, Any]], out_path: Path) -> None:
    by_cell: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        by_cell.setdefault(str(row.get("audit_cell", "unknown")), []).append(row)

    lines = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'>")
    lines.append("<title>H2 audit contact sheet</title>")
    lines.append("<style>")
    lines.append("body{font-family:system-ui,sans-serif;margin:24px;}")
    lines.append("table{border-collapse:collapse;width:100%;font-size:13px;}")
    lines.append("td,th{border:1px solid #ccc;padding:6px;vertical-align:top;}")
    lines.append("img{max-width:360px;image-rendering:auto;}")
    lines.append(".cell{margin-top:32px;}")
    lines.append("</style></head><body>")
    lines.append("<h1>H2 audit contact sheet</h1>")
    lines.append("<p>Each card shows original / binarized foreground / skeleton. Use this only for audit candidate review, not population-level estimates.</p>")

    for cell, group in sorted(by_cell.items()):
        lines.append(f"<div class='cell'><h2>{html.escape(cell)} — n={len(group)}</h2>")
        lines.append(f"<p><a href='{html.escape(cell)}.png'>Open PNG sheet</a></p>")
        lines.append("<table>")
        lines.append("<tr><th>sample</th><th>image</th><th>target / pred</th><th>metrics</th></tr>")

        for row in group:
            img_path = html.escape(str(row.get("image_path", "")))
            sample_id = html.escape(str(row.get("sample_id", "")))
            target = html.escape(str(row.get("target", "")))
            pred = html.escape(str(row.get("pred", "")))
            dataset = html.escape(str(row.get("dataset", "")))
            level = html.escape(str(row.get("level", "")))
            category = html.escape(str(row.get("category", "")))
            cer = safe_float(row.get("cer"))
            risk = safe_float(row.get("structural_risk_score"))

            lines.append("<tr>")
            lines.append(f"<td><b>{sample_id}</b><br>{dataset}<br>{level}<br>{category}</td>")
            lines.append(f"<td><img src='../../../../{img_path}'></td>")
            lines.append(f"<td><b>T:</b> {target}<br><b>P:</b> {pred}</td>")
            lines.append(f"<td>CER={cer:.3f}<br>risk={risk:.3f}</td>")
            lines.append("</tr>")

        lines.append("</table></div>")

    lines.append("</body></html>")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--thumb_w", type=int, default=260)
    parser.add_argument("--thumb_h", type=int, default=90)
    args = parser.parse_args()

    rows = read_csv(Path(args.candidates_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault(str(row.get("audit_cell", "unknown")), []).append(row)

    for cell, group in sorted(by_cell.items()):
        make_cell_sheet(
            group,
            out_dir / f"{cell}.png",
            cols=args.cols,
            thumb_w=args.thumb_w,
            thumb_h=args.thumb_h,
        )

    make_html(rows, out_dir / "index.html")

    summary = {
        "candidates_csv": args.candidates_csv,
        "out_dir": str(out_dir),
        "n": len(rows),
        "cells": {k: len(v) for k, v in sorted(by_cell.items())},
        "outputs": ["index.html"] + [f"{cell}.png" for cell in sorted(by_cell)],
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()