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


VARIANTS = [
    ("baseline", None, None),
    ("darkcc_thr20_area005", 20, 0.005),
    ("darkcc_thr35_area005", 35, 0.005),
    ("darkcc_thr50_area005", 50, 0.005),
    ("darkcc_thr35_area020", 35, 0.020),
    ("darkcc_thr50_area020", 50, 0.020),
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def touches_border(bbox: tuple[int, int, int, int], h: int, w: int, margin: int = 1) -> bool:
    y0, x0, y1, x1 = bbox
    return y0 <= margin or x0 <= margin or y1 >= h - margin or x1 >= w - margin


def repair_dark_border_components(
    arr: np.ndarray,
    *,
    dark_thr: int,
    min_area_frac: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Replace large dark connected components touching image border with white.

    This targets black polygon/crop padding BEFORE binarization.
    It is intentionally diagnostic, not automatically integrated.
    """
    h, w = arr.shape
    dark = arr <= dark_thr
    lab = label(dark, connectivity=2)

    out = arr.copy()
    removed_components = 0
    removed_area = 0
    removed_bboxes = []

    for prop in regionprops(lab):
        area = int(prop.area)
        area_frac = area / max(h * w, 1)

        if not touches_border(prop.bbox, h, w, margin=1):
            continue

        if area_frac < min_area_frac:
            continue

        out[lab == prop.label] = 255
        removed_components += 1
        removed_area += area
        removed_bboxes.append(tuple(int(x) for x in prop.bbox))

    return out, {
        "removed_components": removed_components,
        "removed_area": removed_area,
        "removed_area_frac": removed_area / max(h * w, 1),
        "removed_bboxes": removed_bboxes,
    }


def sauvola_fg(arr: np.ndarray, window: int = 25) -> np.ndarray:
    ink = 255 - arr

    h, w = ink.shape
    win = min(window, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    thr = threshold_sauvola(ink, window_size=win)
    return ink > thr


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def fg_to_img(fg: np.ndarray) -> Image.Image:
    return Image.fromarray((255 - fg.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")


def overlay_skel(gray: np.ndarray, skel: np.ndarray) -> Image.Image:
    base = Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")
    arr = np.asarray(base).copy()
    arr[skel] = np.array([255, 0, 0], dtype=np.uint8)
    return Image.fromarray(arr)


def make_variant_panel(
    title: str,
    arr_repaired: np.ndarray,
    stats: dict[str, Any],
    *,
    panel_w: int,
    panel_h: int,
) -> tuple[Image.Image, dict[str, Any]]:
    fg = sauvola_fg(arr_repaired)
    skel = skeletonize(fg)

    gray_img = Image.fromarray(arr_repaired.astype(np.uint8), mode="L").convert("RGB")
    bin_img = fg_to_img(fg)
    over_img = overlay_skel(arr_repaired, skel)

    view_h = (panel_h - 42) // 3

    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)

    draw.text((5, 4), title, fill="black")
    draw.text(
        (5, 20),
        f"fg={fg.mean():.3f} skel={skel.mean():.3f} rem={stats.get('removed_area_frac', 0):.3f}",
        fill="black",
    )

    panel.paste(fit(gray_img, panel_w, view_h), (0, 42))
    panel.paste(fit(bin_img, panel_w, view_h), (0, 42 + view_h))
    panel.paste(fit(over_img, panel_w, view_h), (0, 42 + 2 * view_h))

    draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline="black")

    metrics = {
        "fg_fraction": float(fg.mean()),
        "skel_fraction": float(skel.mean()),
        **stats,
    }

    return panel, metrics


def process_sample(
    row: dict[str, Any],
    *,
    panel_w: int,
    panel_h: int,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    img_path = Path(row["image_path"])
    gray = ImageOps.autocontrast(Image.open(img_path).convert("L"))
    arr = np.asarray(gray, dtype=np.uint8)

    sample_header_h = 88
    cols = len(VARIANTS) + 1
    card_w = cols * panel_w
    card_h = sample_header_h + panel_h

    card = Image.new("RGB", (card_w, card_h), "white")
    draw = ImageDraw.Draw(card)

    header_lines = [
        f"id: {row.get('sample_id')}",
        f"cell: {row.get('audit_cell')} | CER={safe_float(row.get('cer')):.3f} | risk={safe_float(row.get('structural_risk_score')):.3f}",
        f"target: {row.get('target')}",
        f"pred: {row.get('pred')}",
        f"path: {row.get('image_path')}",
    ]

    y = 4
    for line in header_lines:
        draw.text((6, y), str(line)[:180], fill="black")
        y += 16

    # Original panel
    orig_panel = Image.new("RGB", (panel_w, panel_h), "white")
    d = ImageDraw.Draw(orig_panel)
    d.text((5, 4), "original grayscale", fill="black")
    d.text((5, 20), f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}", fill="black")
    orig_panel.paste(fit(gray.convert("RGB"), panel_w, panel_h - 42), (0, 42))
    d.rectangle((0, 0, panel_w - 1, panel_h - 1), outline="black")
    card.paste(orig_panel, (0, sample_header_h))

    metrics_rows: list[dict[str, Any]] = []

    for i, (name, thr, area_frac) in enumerate(VARIANTS, start=1):
        if name == "baseline":
            repaired = arr.copy()
            stats = {
                "removed_components": 0,
                "removed_area": 0,
                "removed_area_frac": 0.0,
                "removed_bboxes": [],
            }
        else:
            repaired, stats = repair_dark_border_components(
                arr,
                dark_thr=int(thr),
                min_area_frac=float(area_frac),
            )

        panel, m = make_variant_panel(
            name,
            repaired,
            stats,
            panel_w=panel_w,
            panel_h=panel_h,
        )

        card.paste(panel, (i * panel_w, sample_header_h))

        metrics_rows.append({
            "sample_id": row.get("sample_id"),
            "dataset": row.get("dataset"),
            "audit_cell": row.get("audit_cell"),
            "image_path": row.get("image_path"),
            "variant": name,
            "dark_thr": "" if thr is None else thr,
            "min_area_frac": "" if area_frac is None else area_frac,
            "fg_fraction": m["fg_fraction"],
            "skel_fraction": m["skel_fraction"],
            "removed_components": m["removed_components"],
            "removed_area": m["removed_area"],
            "removed_area_frac": m["removed_area_frac"],
        })

    draw.rectangle((0, 0, card_w - 1, card_h - 1), outline="black")
    return card, metrics_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_sheet(cards: list[Image.Image], path: Path) -> None:
    if not cards:
        return

    w = max(c.width for c in cards)
    h = sum(c.height for c in cards)

    sheet = Image.new("RGB", (w, h), "white")

    y = 0
    for c in cards:
        sheet.paste(c, (0, y))
        y += c.height

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="school_notebooks_clean")
    parser.add_argument("--panel_w", type=int, default=220)
    parser.add_argument("--panel_h", type=int, default=230)
    args = parser.parse_args()

    rows = read_csv(Path(args.annotations_csv))
    rows = [r for r in rows if r.get("dataset") == args.dataset]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    metrics = []

    for i, r in enumerate(rows, 1):
        print(f"{i}/{len(rows)} {r.get('sample_id')}")
        card, mr = process_sample(r, panel_w=args.panel_w, panel_h=args.panel_h)
        cards.append(card)
        metrics.extend(mr)

    make_sheet(cards, out_dir / "dark_padding_repair_contact_sheet.png")
    write_csv(metrics, out_dir / "dark_padding_repair_metrics.csv")

    summary = {
        "annotations_csv": args.annotations_csv,
        "dataset": args.dataset,
        "n_samples": len(rows),
        "variants": [v[0] for v in VARIANTS],
        "outputs": {
            "contact_sheet": str(out_dir / "dark_padding_repair_contact_sheet.png"),
            "metrics_csv": str(out_dir / "dark_padding_repair_metrics.csv"),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
