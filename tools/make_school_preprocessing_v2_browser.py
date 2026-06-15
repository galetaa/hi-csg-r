from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize


VARIANTS = [
    "baseline",
    "mask_2px",
    "mask_5px",
    "mask_8px",
    "adaptive_edge_025",
    "adaptive_edge_015",
    "whiten_8px_then_sauvola",
    "whiten_12px_then_sauvola",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sauvola_fg(arr_u8: np.ndarray, window: int = 25) -> np.ndarray:
    ink = 255 - arr_u8
    h, w = ink.shape

    win = min(window, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    thr = threshold_sauvola(ink, window_size=win)
    return ink > thr


def mask_border(fg: np.ndarray, px: int) -> np.ndarray:
    out = fg.copy()
    if px <= 0:
        return out

    h, w = out.shape
    px = min(px, h // 2, w // 2)

    out[:px, :] = False
    out[-px:, :] = False
    out[:, :px] = False
    out[:, -px:] = False
    return out


def whiten_border(arr: np.ndarray, px: int) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    px = min(px, h // 2, w // 2)

    out[:px, :] = 255
    out[-px:, :] = 255
    out[:, :px] = 255
    out[:, -px:] = 255
    return out


def adaptive_edge_strip(fg: np.ndarray, density_thr: float, max_frac: float = 0.12) -> np.ndarray:
    """
    Remove only contiguous foreground-dense strips touching image edges.

    This is intentionally conservative and only used for visual candidate generation.
    It is not automatically integrated.
    """
    out = fg.copy()
    h, w = out.shape

    max_y = max(1, int(h * max_frac))
    max_x = max(1, int(w * max_frac))

    # top
    top = 0
    for y in range(max_y):
        if out[y, :].mean() >= density_thr:
            top = y + 1
        else:
            break

    # bottom
    bottom = 0
    for y in range(h - 1, max(h - max_y - 1, -1), -1):
        if out[y, :].mean() >= density_thr:
            bottom += 1
        else:
            break

    # left
    left = 0
    for x in range(max_x):
        if out[:, x].mean() >= density_thr:
            left = x + 1
        else:
            break

    # right
    right = 0
    for x in range(w - 1, max(w - max_x - 1, -1), -1):
        if out[:, x].mean() >= density_thr:
            right += 1
        else:
            break

    if top:
        out[:top, :] = False
    if bottom:
        out[h - bottom:, :] = False
    if left:
        out[:, :left] = False
    if right:
        out[:, w - right:] = False

    return out


def make_variant(arr: np.ndarray, name: str) -> np.ndarray:
    base = sauvola_fg(arr)

    if name == "baseline":
        return base

    if name == "mask_2px":
        return mask_border(base, 2)

    if name == "mask_5px":
        return mask_border(base, 5)

    if name == "mask_8px":
        return mask_border(base, 8)

    if name == "adaptive_edge_025":
        return adaptive_edge_strip(base, density_thr=0.25, max_frac=0.12)

    if name == "adaptive_edge_015":
        return adaptive_edge_strip(base, density_thr=0.15, max_frac=0.12)

    if name == "whiten_8px_then_sauvola":
        return sauvola_fg(whiten_border(arr, 8))

    if name == "whiten_12px_then_sauvola":
        return sauvola_fg(whiten_border(arr, 12))

    raise ValueError(f"unknown variant: {name}")


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def fg_to_img(fg: np.ndarray) -> Image.Image:
    return Image.fromarray((255 - fg.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")


def overlay_skel(orig: Image.Image, skel: np.ndarray) -> Image.Image:
    arr = np.asarray(orig.convert("RGB")).copy()
    arr[skel] = np.array([255, 0, 0], dtype=np.uint8)
    return Image.fromarray(arr)


def make_variant_panel(orig: Image.Image, fg: np.ndarray, title: str, w: int, h: int) -> Image.Image:
    skel = skeletonize(fg)

    binary = fg_to_img(fg)
    overlay = overlay_skel(orig, skel)

    panel = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), title, fill="black")

    sub_h = (h - 24) // 2

    panel.paste(fit(binary, w, sub_h), (0, 24))
    panel.paste(fit(overlay, w, sub_h), (0, 24 + sub_h))

    draw.rectangle((0, 0, w - 1, h - 1), outline="black")

    return panel


def prepare_item(row: dict[str, Any], view_w: int, view_h: int) -> dict[str, Any]:
    img_path = Path(row["image_path"])
    orig_l = ImageOps.autocontrast(Image.open(img_path).convert("L"))
    orig_rgb = orig_l.convert("RGB")
    arr = np.asarray(orig_l, dtype=np.uint8)

    variants = {}

    for name in VARIANTS:
        try:
            fg = make_variant(arr, name)
            skel = skeletonize(fg)
            panel = make_variant_panel(orig_rgb, fg, name, view_w, view_h)

            variants[name] = {
                "image": data_url(panel),
                "fg_fraction": float(fg.mean()),
                "skel_fraction": float(skel.mean()),
            }
        except Exception as e:
            error_img = Image.new("RGB", (view_w, view_h), "white")
            d = ImageDraw.Draw(error_img)
            d.text((10, 10), f"{name} ERROR: {e}", fill="black")
            variants[name] = {
                "image": data_url(error_img),
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
        "target": row.get("target", ""),
        "pred": row.get("pred", ""),
        "image_path": row.get("image_path", ""),
        "original": data_url(fit(orig_rgb, view_w, 100)),
        "variants": variants,
    }


HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>School preprocessing v2 audit</title>
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
button, select {
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
  grid-template-columns: 1fr 330px;
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
  max-height: 240px;
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
      выбрать вариант, который убирает border/background artifact,
      но не стирает реальные буквы.<br><br>
      Если ни один не лучше baseline — выбирай baseline и fix_grade=bad_fix.
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

    <h3>Border artifact after best?</h3>
    <div class="row">
      <button id="border_after_0" onclick="setField('border_artifact_after','0')">no</button>
      <button id="border_after_1" onclick="setField('border_artifact_after','1')">yes</button>
    </div>

    <h3>Skeleton follows ink after best?</h3>
    <div class="row">
      <button id="follows_after_1" onclick="setField('skeleton_follows_ink_after','1')">yes</button>
      <button id="follows_after_0" onclick="setField('skeleton_follows_ink_after','0')">no</button>
    </div>

    <h3>Notes</h3>
    <textarea id="notes" oninput="setField('notes', this.value)"></textarea>

    <div class="row">
      <button class="primary" onclick="saveNext()">Save + Next</button>
    </div>

    <h3>Samples</h3>
    <div class="sample-list" id="sampleList"></div>

    <p class="small">
      Правило: не ищем идеал. Ищем: стало ли явно лучше без удаления букв.
    </p>
  </aside>
</main>

<script id="data" type="application/json">__DATA__</script>

<script>
const data = JSON.parse(document.getElementById("data").textContent);
const items = data.items;
const variants = data.variants;
const storageKey = "school_preprocessing_v2_" + data.storage_id;

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
      <div class="stats">fg=${m.fg_fraction === null ? "ERR" : Number(m.fg_fraction).toFixed(4)}
      skel=${m.skel_fraction === null ? "ERR" : Number(m.skel_fraction).toFixed(4)}</div>
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
    parser.add_argument("--view_w", type=int, default=420)
    parser.add_argument("--view_h", type=int, default=230)
    args = parser.parse_args()

    rows = read_csv(Path(args.annotations_csv))
    rows = [r for r in rows if r.get("dataset") == args.dataset]

    items = []
    for i, r in enumerate(rows, 1):
        print(f"{i}/{len(rows)} {r.get('sample_id')}")
        items.append(prepare_item(r, args.view_w, args.view_h))

    data = {
        "items": items,
        "variants": VARIANTS,
        "storage_id": Path(args.out_html).stem,
        "export_csv_name": f"{Path(args.out_html).stem}_annotations.csv",
        "export_json_name": f"{Path(args.out_html).stem}_annotations.json",
    }

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    html_text = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out_html.write_text(html_text, encoding="utf-8")

    print("wrote:", out_html)
    print("items:", len(items))


if __name__ == "__main__":
    main()