from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import remove_small_objects, skeletonize
from src.preprocessing.school_foreground_v4 import (
    global_affine_whiten_foreground,
    load_rgb_and_support,
    masked_local_whiten_foreground,
    school_dark_auto_v2,
)

VARIANTS = [
    "school_dark_auto_v2",
    "global_affine_whitening",
    "masked_local_whitening",
]

RAW_BY_ID: dict[str, dict[str, Any]] = {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_raw_index(
    root: Path,
) -> dict[str, dict[str, Any]]:
    result = {}

    for split in [
        "train",
        "val",
        "test",
    ]:
        for row in read_jsonl(root / f"{split}.jsonl"):
            result[str(row["sample_id"])] = row

    return result


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cleanup_dark_mask(fg: np.ndarray) -> np.ndarray:
    fg = fg.astype(bool)
    if fg.any():
        fg = remove_small_objects(fg, min_size=4)
    return fg.astype(bool)


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
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    bg = np.asarray(
        img.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
        dtype=np.float32,
    )

    diff = bg - arr.astype(np.float32)
    fg = diff >= float(delta)

    if fg.any():
        fg = remove_small_objects(fg.astype(bool), min_size=4)

    return fg.astype(bool)


def cleanup_mask(
    fg: np.ndarray,
) -> np.ndarray:
    fg = fg.astype(bool)

    if fg.any():
        fg = remove_small_objects(
            fg,
            min_size=4,
        )

    return fg.astype(bool)


def otsu_dark(
    arr: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(
        arr,
        dtype=np.uint8,
    )

    if int(arr.max()) <= int(arr.min()):
        return np.zeros_like(
            arr,
            dtype=bool,
        )

    threshold = threshold_otsu(arr)

    return cleanup_mask(arr < threshold)


def percentile_normalized(
    arr: np.ndarray,
) -> np.ndarray:
    low = float(np.percentile(arr, 2))
    high = float(np.percentile(arr, 90))

    if high <= low + 1:
        return arr.copy()

    normalized = (arr.astype(np.float32) - low) * 255.0 / (high - low)

    return np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)


def division_normalized(
    arr: np.ndarray,
) -> np.ndarray:
    sigma = max(
        5.0,
        min(arr.shape) / 6.0,
    )

    background = cv2.GaussianBlur(
        arr,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    denominator = np.maximum(
        background,
        1,
    ).astype(np.uint8)

    normalized = cv2.divide(
        arr,
        denominator,
        scale=255,
    )

    return normalized.astype(np.uint8)


def bgdiff_mask(
    arr: np.ndarray,
) -> np.ndarray:
    sigma = max(
        5.0,
        min(arr.shape) / 6.0,
    )

    background = cv2.GaussianBlur(
        arr,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    ).astype(np.float32)

    difference = np.clip(
        background - arr.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    if int(difference.max()) == 0:
        return np.zeros_like(
            difference,
            dtype=bool,
        )

    threshold = max(
        float(threshold_otsu(difference)),
        6.0,
    )

    return cleanup_mask(difference > threshold)


def make_variant(
    gray: np.ndarray,
    rgb: np.ndarray,
    support: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    if name == "school_dark_auto_v2":
        foreground = school_dark_auto_v2(gray)
        return foreground, gray, None

    if name == "global_affine_whitening":
        return global_affine_whiten_foreground(
            gray,
            support,
        )

    if name == "masked_local_whitening":
        return masked_local_whiten_foreground(
            gray,
            support,
        )

    raise ValueError(name)


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def fg_to_img(fg: np.ndarray) -> Image.Image:
    return Image.fromarray(
        (255 - fg.astype(np.uint8) * 255).astype(np.uint8),
        mode="L",
    ).convert("RGB")


def overlay_skel(gray: np.ndarray, skel: np.ndarray) -> Image.Image:
    base = Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")
    rgb = np.asarray(base).copy()
    rgb[skel] = np.array([255, 0, 0], dtype=np.uint8)
    return Image.fromarray(rgb)


def data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_panel(gray: np.ndarray, fg: np.ndarray, title: str, w: int, h: int) -> Image.Image:
    skel = skeletonize(fg)

    gray_img = Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")
    bin_img = fg_to_img(fg)
    over_img = overlay_skel(gray, skel)

    panel = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(panel)

    header_h = 42
    view_h = (h - header_h) // 3

    d.text((5, 4), title[:42], fill="black")
    d.text((5, 20), f"fg={fg.mean():.3f} skel={skel.mean():.3f}", fill="black")

    panel.paste(fit(gray_img, w, view_h), (0, header_h))
    panel.paste(fit(bin_img, w, view_h), (0, header_h + view_h))
    panel.paste(fit(over_img, w, view_h), (0, header_h + 2 * view_h))

    d.rectangle((0, 0, w - 1, h - 1), outline="black")
    return panel


def prepare_item(row: dict[str, Any], panel_w: int, panel_h: int) -> dict[str, Any]:
    img_path = Path(row["image_path"])
    gray_img = Image.open(img_path).convert("L")
    gray = np.asarray(gray_img, dtype=np.uint8)

    raw_row = RAW_BY_ID.get(str(row["sample_id"]))

    if raw_row is not None:
        rgb, support = load_rgb_and_support(
            raw_row,
            target_shape=gray.shape,
        )
    else:
        rgb = np.repeat(
            gray[:, :, None],
            3,
            axis=2,
        )

        support = np.ones_like(
            gray,
            dtype=bool,
        )

    original = ImageOps.autocontrast(gray_img).convert("RGB")

    variants: dict[str, Any] = {}

    for name in VARIANTS:
        try:
            fg, display_gray, adaptive_threshold = make_variant(
                gray,
                rgb,
                support,
                name,
            )
            skel = skeletonize(fg)
            panel = make_panel(
                display_gray,
                fg,
                (name if adaptive_threshold is None else (f"{name} T={adaptive_threshold:.1f}")),
                panel_w,
                panel_h,
            )

            variants[name] = {
                "image": data_url(panel),
                "fg_fraction": float(fg.mean()),
                "skel_fraction": float(skel.mean()),
            }
        except Exception as e:
            err = Image.new("RGB", (panel_w, panel_h), "white")
            d = ImageDraw.Draw(err)
            d.text((10, 10), f"{name} ERROR", fill="black")
            d.text((10, 30), str(e)[:80], fill="black")
            variants[name] = {
                "image": data_url(err),
                "fg_fraction": None,
                "skel_fraction": None,
            }

    return {
        "sample_id": row.get("sample_id", ""),
        "audit_cell": row.get("audit_cell", ""),
        "dataset": row.get("dataset", ""),
        "level": row.get("level", ""),
        "category": row.get("category", ""),
        "cer": row.get("cer", ""),
        "risk": row.get("structural_risk_score", ""),
        "target": (row.get("target") or row.get("text") or ""),
        "pred": row.get("pred", ""),
        "image_path": row.get("image_path", ""),
        "original": data_url(fit(original, panel_w * 2, 120)),
        "variants": variants,
    }


HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>School foreground v3 browser</title>
<style>
body {
  font-family: system-ui, sans-serif;
  margin: 0;
  background: #f5f5f5;
  color: #111;
}
header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: #111;
  color: white;
  padding: 10px 16px;
  display: flex;
  gap: 10px;
  align-items: center;
}
button {
  font-size: 14px;
  padding: 7px 10px;
  border-radius: 6px;
  border: 1px solid #aaa;
  background: white;
  cursor: pointer;
}
button.primary {
  background: #111;
  color: white;
  border-color: white;
}
main {
  display: grid;
  grid-template-columns: 1fr 340px;
  gap: 14px;
  padding: 14px;
}
.panel {
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 12px;
}
.meta {
  font-size: 14px;
  line-height: 1.4;
}
.textbox {
  margin-top: 8px;
  font-size: 17px;
  background: #fafafa;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 8px;
}
.variants {
  margin-top: 12px;
  display: grid;
  grid-template-columns: repeat(2, minmax(320px, 1fr));
  gap: 10px;
}
.variant {
  border: 2px solid #ddd;
  border-radius: 8px;
  padding: 8px;
}
.variant.selected {
  border-color: #111;
  background: #eef6ff;
}
.variant img {
  width: 100%;
  background: white;
  border: 1px solid #ccc;
}
.variant .stats {
  font-size: 12px;
  color: #555;
}
.controls h3 {
  margin: 14px 0 6px;
}
.row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.good { background: #ddf5dd; }
.partial { background: #fff0c2; }
.bad { background: #ffdada; }
.selectedButton {
  outline: 3px solid #111;
}
textarea {
  width: 100%;
  min-height: 80px;
}
.sample-list {
  max-height: 250px;
  overflow: auto;
  border: 1px solid #ddd;
  border-radius: 8px;
}
.sample-list button {
  display: block;
  width: 100%;
  border: 0;
  border-bottom: 1px solid #eee;
  border-radius: 0;
  text-align: left;
}
.sample-list button.done { background: #e8f5e9; }
.sample-list button.current { background: #e3f2fd; }
.help {
  font-size: 13px;
  line-height: 1.35;
  background: #fafafa;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 8px;
}
.small {
  font-size: 12px;
  color: #555;
}
</style>
</head>
<body>

<header>
  <button onclick="prevItem()">← Prev</button>
  <button onclick="nextItem()">Next →</button>
  <button class="primary" onclick="exportCSV()">Export CSV</button>
  <button onclick="exportJSON()">Export JSON</button>
  <span id="progress"></span>
</header>

<main>
  <section class="panel">
    <div class="meta" id="meta"></div>
    <div class="textbox" id="text"></div>
    <div style="margin-top:10px;">
      <b>Original</b><br>
      <img id="original" style="max-width:100%;border:1px solid #ccc;background:white;">
    </div>
    <div class="variants" id="variants"></div>
  </section>

  <aside class="panel controls">
    <div class="help">
      <b>Задача:</b><br>
      выбрать вариант, где binary содержит в основном буквы,
      а overlay/skeleton идёт по буквам, не по серой плашке.<br><br>
      Если стало только чуть лучше — partial_fix.<br>
      Если всё ещё плохо — bad_fix.
    </div>

    <h3>Fix grade</h3>
    <div class="row">
      <button class="good" id="grade_good_fix" onclick="setField('fix_grade','good_fix')">good_fix</button>
      <button class="partial" id="grade_partial_fix" onclick="setField('fix_grade','partial_fix')">partial_fix</button>
      <button class="bad" id="grade_bad_fix" onclick="setField('fix_grade','bad_fix')">bad_fix</button>
    </div>

    <h3>Real ink erased?</h3>
    <div class="row">
      <button id="erased_0" onclick="setField('real_ink_erased','0')">no</button>
      <button id="erased_1" onclick="setField('real_ink_erased','1')">yes</button>
    </div>

    <h3>Background/blob after best?</h3>
    <div class="row">
      <button id="border_after_0" onclick="setField('border_artifact_after','0')">no</button>
      <button id="border_after_1" onclick="setField('border_artifact_after','1')">yes</button>
    </div>

    <h3>Skeleton follows ink after best?</h3>
    <div class="row">
      <button id="follows_after_1" onclick="setField('skeleton_follows_ink_after','1')">yes</button>
      <button id="follows_after_0" onclick="setField('skeleton_follows_ink_after','0')">no</button>
    </div>

    <h3>Quick presets</h3>
    <div class="row">
      <button class="good" onclick="presetGood()">good</button>
      <button class="partial" onclick="presetPartial()">partial</button>
      <button class="bad" onclick="presetBad()">bad</button>
    </div>

    <h3>Notes</h3>
    <textarea id="notes" oninput="setField('notes', this.value)"></textarea>

    <div class="row">
      <button class="primary" onclick="saveNext()">Save + Next</button>
    </div>

    <h3>Samples</h3>
    <div class="sample-list" id="sampleList"></div>

    <p class="small">
      Клавиши: ←/→ навигация, G good, P partial, B bad.
    </p>
  </aside>
</main>

<script id="data" type="application/json">__DATA__</script>

<script>
const data = JSON.parse(document.getElementById("data").textContent);
const items = data.items;
const variants = data.variants;
const storageKey = "school_foreground_v3_" + data.storage_id;

let annotations = JSON.parse(localStorage.getItem(storageKey) || "{}");
let current = 0;

function defaultAnn(item) {
  return {
    sample_id: item.sample_id,
    audit_cell: item.audit_cell,
    dataset: item.dataset,
    image_path: item.image_path,
    target: item.target,
    pred: item.pred,
    best_variant: "",
    fix_grade: "",
    real_ink_erased: "",
    border_artifact_after: "",
    skeleton_follows_ink_after: "",
    notes: ""
  };
}

function annFor(item) {
  if (!annotations[item.sample_id]) annotations[item.sample_id] = defaultAnn(item);
  return annotations[item.sample_id];
}

function save() {
  localStorage.setItem(storageKey, JSON.stringify(annotations));
  updateProgress();
}

function setField(k, v) {
  annFor(items[current])[k] = String(v);
  save();
  render();
}

function selectVariant(v) {
  annFor(items[current]).best_variant = v;
  save();
  render();
}

function presetGood() {
  const a = annFor(items[current]);
  a.fix_grade = "good_fix";
  a.real_ink_erased = "0";
  a.border_artifact_after = "0";
  a.skeleton_follows_ink_after = "1";
  save();
  render();
}

function presetPartial() {
  const a = annFor(items[current]);
  a.fix_grade = "partial_fix";
  a.real_ink_erased = "0";
  a.border_artifact_after = "1";
  a.skeleton_follows_ink_after = "0";
  save();
  render();
}

function presetBad() {
  const a = annFor(items[current]);
  a.fix_grade = "bad_fix";
  a.real_ink_erased = "0";
  a.border_artifact_after = "1";
  a.skeleton_follows_ink_after = "0";
  save();
  render();
}

function isDone(a) {
  return a.best_variant && a.fix_grade && a.real_ink_erased && a.border_artifact_after && a.skeleton_follows_ink_after;
}

function updateProgress() {
  let done = 0;
  for (const item of items) {
    const a = annotations[item.sample_id];
    if (a && isDone(a)) done++;
  }
  document.getElementById("progress").textContent = `${done}/${items.length} done`;
}

function prevItem() {
  current = Math.max(0, current - 1);
  render();
}

function nextItem() {
  current = Math.min(items.length - 1, current + 1);
  render();
}

function saveNext() {
  save();
  nextItem();
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderButtons(a) {
  document.querySelectorAll("button").forEach(b => b.classList.remove("selectedButton"));

  function mark(id, cond) {
    const el = document.getElementById(id);
    if (el && cond) el.classList.add("selectedButton");
  }

  mark("grade_" + a.fix_grade, true);
  mark("erased_" + a.real_ink_erased, true);
  mark("border_after_" + a.border_artifact_after, true);
  mark("follows_after_" + a.skeleton_follows_ink_after, true);

  document.getElementById("notes").value = a.notes || "";
}

function renderSampleList() {
  const root = document.getElementById("sampleList");
  root.innerHTML = "";

  items.forEach((item, idx) => {
    const a = annFor(item);
    const btn = document.createElement("button");
    btn.textContent = `${idx + 1}. ${item.sample_id} | ${item.audit_cell}`;
    if (idx === current) btn.classList.add("current");
    if (isDone(a)) btn.classList.add("done");
    btn.onclick = () => { current = idx; render(); };
    root.appendChild(btn);
  });
}

function render() {
  const item = items[current];
  const a = annFor(item);

  document.getElementById("meta").innerHTML = `
    <b>${current + 1}/${items.length}</b><br>
    <b>id:</b> ${escapeHtml(item.sample_id)}<br>
    <b>cell:</b> ${escapeHtml(item.audit_cell)}<br>
    <b>CER:</b> ${escapeHtml(item.cer)} | <b>risk:</b> ${escapeHtml(item.risk)}<br>
    <span class="small">${escapeHtml(item.image_path)}</span>
  `;

  document.getElementById("text").innerHTML = `
    <b>Target:</b> ${escapeHtml(item.target)}<br>
    <b>Pred:</b> ${escapeHtml(item.pred)}
  `;

  document.getElementById("original").src = item.original;

  const root = document.getElementById("variants");
  root.innerHTML = "";

  for (const v of variants) {
    const box = document.createElement("div");
    box.className = "variant";
    if (a.best_variant === v) box.classList.add("selected");

    const m = item.variants[v];

    box.innerHTML = `
      <button onclick="selectVariant('${v}')">choose ${escapeHtml(v)}</button>
      <div class="stats">
        fg=${m.fg_fraction === null ? "ERR" : Number(m.fg_fraction).toFixed(4)}
        skel=${m.skel_fraction === null ? "ERR" : Number(m.skel_fraction).toFixed(4)}
      </div>
      <img src="${m.image}">
    `;

    root.appendChild(box);
  }

  renderButtons(a);
  renderSampleList();
  updateProgress();
}

function csvEscape(v) {
  const s = String(v ?? "");
  if (/[",\n\r]/.test(s)) return '"' + s.replaceAll('"', '""') + '"';
  return s;
}

function exportCSV() {
  const cols = [
    "sample_id",
    "audit_cell",
    "dataset",
    "image_path",
    "target",
    "pred",
    "best_variant",
    "fix_grade",
    "real_ink_erased",
    "border_artifact_after",
    "skeleton_follows_ink_after",
    "notes"
  ];

  const lines = [cols.join(",")];

  for (const item of items) {
    const a = annFor(item);
    lines.push(cols.map(c => csvEscape(a[c])).join(","));
  }

  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv;charset=utf-8"});
  download(blob, data.export_csv_name);
}

function exportJSON() {
  const arr = items.map(item => annFor(item));
  const blob = new Blob([JSON.stringify(arr, null, 2)], {type: "application/json;charset=utf-8"});
  download(blob, data.export_json_name);
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight") nextItem();
  if (e.key === "ArrowLeft") prevItem();
  if (e.key.toLowerCase() === "g") presetGood();
  if (e.key.toLowerCase() === "p") presetPartial();
  if (e.key.toLowerCase() === "b") presetBad();
});

render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_html", required=True)
    parser.add_argument("--dataset", default="school_notebooks_clean")
    parser.add_argument("--panel_w", type=int, default=420)
    parser.add_argument("--panel_h", type=int, default=280)
    parser.add_argument(
        "--raw_root",
        default=("data/experiments/htr_baseline_v1/school_notebooks_clean"),
    )
    args = parser.parse_args()

    global RAW_BY_ID

    RAW_BY_ID = load_raw_index(Path(args.raw_root))

    rows = read_csv(Path(args.annotations_csv))
    rows = [r for r in rows if r.get("dataset") == args.dataset]

    items = []
    for i, row in enumerate(rows, 1):
        print(f"{i}/{len(rows)} {row.get('sample_id')}")
        items.append(prepare_item(row, args.panel_w, args.panel_h))

    data = {
        "items": items,
        "variants": VARIANTS,
        "storage_id": Path(args.out_html).stem,
        "export_csv_name": f"{Path(args.out_html).stem}_annotations.csv",
        "export_json_name": f"{Path(args.out_html).stem}_annotations.json",
    }

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    html_text = HTML.replace(
        "__DATA__",
        json.dumps(data, ensure_ascii=False).replace("</", "<\\/"),
    )

    out_html.write_text(html_text, encoding="utf-8")

    print("wrote:", out_html)
    print("items:", len(items))


if __name__ == "__main__":
    main()
