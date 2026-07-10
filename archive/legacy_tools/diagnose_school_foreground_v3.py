from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects


VARIANTS = [
    "baseline_sauvola",
    "global_dark_120",
    "global_dark_145",
    "bgdiff_blur15_delta12",
    "bgdiff_blur15_delta18",
    "bgdiff_blur25_delta12",
    "bgdiff_blur25_delta18",
    "contrast_autocontrast_bgdiff_delta18",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def sauvola_fg(arr: np.ndarray, window: int = 25) -> np.ndarray:
    ink = 255 - arr
    h, w = ink.shape
    win = min(window, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    thr = threshold_sauvola(ink, window_size=win)
    return ink > thr


def bgdiff_fg(arr: np.ndarray, blur_radius: int, delta: int) -> np.ndarray:
    """
    Foreground = pixels darker than their local blurred background.

    This targets gray polygon/background patches:
    gray background should be close to local background,
    real ink should be much darker.
    """
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    bg = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32)

    diff = bg - arr.astype(np.float32)
    fg = diff >= float(delta)

    # remove tiny specks, keep handwriting-scale components
    if fg.any():
        fg = remove_small_objects(fg.astype(bool), min_size=4)

    return fg.astype(bool)


def make_variant(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "baseline_sauvola":
        return sauvola_fg(arr)

    if name == "global_dark_120":
        return arr < 120

    if name == "global_dark_145":
        return arr < 145

    if name == "bgdiff_blur15_delta12":
        return bgdiff_fg(arr, blur_radius=15, delta=12)

    if name == "bgdiff_blur15_delta18":
        return bgdiff_fg(arr, blur_radius=15, delta=18)

    if name == "bgdiff_blur25_delta12":
        return bgdiff_fg(arr, blur_radius=25, delta=12)

    if name == "bgdiff_blur25_delta18":
        return bgdiff_fg(arr, blur_radius=25, delta=18)

    if name == "contrast_autocontrast_bgdiff_delta18":
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        ac = ImageOps.autocontrast(img, cutoff=1)
        ac_arr = np.asarray(ac, dtype=np.uint8)
        return bgdiff_fg(ac_arr, blur_radius=25, delta=18)

    raise ValueError(name)


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
    rgb = np.asarray(base).copy()
    rgb[skel] = np.array([255, 0, 0], dtype=np.uint8)
    return Image.fromarray(rgb)


def make_panel(
    *,
    title: str,
    gray: np.ndarray,
    fg: np.ndarray,
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    skel = skeletonize(fg)

    gray_img = Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")
    bin_img = fg_to_img(fg)
    over_img = overlay_skel(gray, skel)

    view_h = (panel_h - 42) // 3

    panel = Image.new("RGB", (panel_w, panel_h), "white")
    d = ImageDraw.Draw(panel)

    d.text((5, 4), title[:36], fill="black")
    d.text((5, 20), f"fg={fg.mean():.3f} skel={skel.mean():.3f}", fill="black")

    panel.paste(fit(gray_img, panel_w, view_h), (0, 42))
    panel.paste(fit(bin_img, panel_w, view_h), (0, 42 + view_h))
    panel.paste(fit(over_img, panel_w, view_h), (0, 42 + 2 * view_h))

    d.rectangle((0, 0, panel_w - 1, panel_h - 1), outline="black")
    return panel


def process_sample(
    row: dict[str, Any],
    *,
    panel_w: int,
    panel_h: int,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    img_path = Path(row["image_path"])
    gray_img = Image.open(img_path).convert("L")
    gray = np.asarray(gray_img, dtype=np.uint8)

    header_h = 92
    cols = 1 + len(VARIANTS)
    card_w = cols * panel_w
    card_h = header_h + panel_h

    card = Image.new("RGB", (card_w, card_h), "white")
    d = ImageDraw.Draw(card)

    header = [
        f"id: {row.get('sample_id')}",
        f"cell: {row.get('audit_cell')} | CER={safe_float(row.get('cer')):.3f} | risk={safe_float(row.get('structural_risk_score')):.3f}",
        f"target: {row.get('target')}",
        f"pred: {row.get('pred')}",
        f"path: {row.get('image_path')}",
    ]

    y = 4
    for line in header:
        d.text((6, y), str(line)[:180], fill="black")
        y += 16

    orig_panel = Image.new("RGB", (panel_w, panel_h), "white")
    od = ImageDraw.Draw(orig_panel)
    od.text((5, 4), "original", fill="black")
    od.text((5, 20), f"min={gray.min()} max={gray.max()} mean={gray.mean():.1f}", fill="black")
    orig_panel.paste(fit(gray_img.convert("RGB"), panel_w, panel_h - 42), (0, 42))
    od.rectangle((0, 0, panel_w - 1, panel_h - 1), outline="black")
    card.paste(orig_panel, (0, header_h))

    metric_rows = []

    for idx, name in enumerate(VARIANTS, start=1):
        try:
            fg = make_variant(gray, name)
        except Exception:
            fg = np.zeros_like(gray, dtype=bool)

        skel = skeletonize(fg)
        panel = make_panel(title=name, gray=gray, fg=fg, panel_w=panel_w, panel_h=panel_h)
        card.paste(panel, (idx * panel_w, header_h))

        metric_rows.append({
            "sample_id": row.get("sample_id"),
            "audit_cell": row.get("audit_cell"),
            "dataset": row.get("dataset"),
            "image_path": row.get("image_path"),
            "variant": name,
            "fg_fraction": float(fg.mean()),
            "skel_fraction": float(skel.mean()),
        })

    d.rectangle((0, 0, card_w - 1, card_h - 1), outline="black")
    return card, metric_rows


def make_sheet(cards: list[Image.Image], out_path: Path) -> None:
    if not cards:
        return

    w = max(c.width for c in cards)
    h = sum(c.height for c in cards)

    sheet = Image.new("RGB", (w, h), "white")

    y = 0
    for c in cards:
        sheet.paste(c, (0, y))
        y += c.height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="school_notebooks_clean")
    parser.add_argument("--panel_w", type=int, default=210)
    parser.add_argument("--panel_h", type=int, default=235)
    args = parser.parse_args()

    rows = read_csv(Path(args.annotations_csv))
    rows = [r for r in rows if r.get("dataset") == args.dataset]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    metrics = []

    for i, r in enumerate(rows, 1):
        print(f"{i}/{len(rows)} {r.get('sample_id')}")
        card, ms = process_sample(r, panel_w=args.panel_w, panel_h=args.panel_h)
        cards.append(card)
        metrics.extend(ms)

    make_sheet(cards, out_dir / "school_foreground_v3_contact_sheet.png")
    write_csv(metrics, out_dir / "school_foreground_v3_metrics.csv")

    summary = {
        "dataset": args.dataset,
        "n": len(rows),
        "variants": VARIANTS,
        "outputs": {
            "contact_sheet": str(out_dir / "school_foreground_v3_contact_sheet.png"),
            "metrics_csv": str(out_dir / "school_foreground_v3_metrics.csv"),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
