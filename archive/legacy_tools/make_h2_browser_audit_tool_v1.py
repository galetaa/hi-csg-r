from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import skeletonize


ANNOTATION_FIELDS = [
    "audit_usable",
    "exclusion_reason",
    "ink_visible_ok",
    "skeleton_follows_ink",
    "missed_visible_stroke",
    "spurious_stroke",
    "endpoint_error",
    "junction_error",
    "loop_error",
    "critical_topology_error",
    "graph_quality_0_3",
    "notes",
]


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


def fit_to_canvas(img: Image.Image, w: int, h: int) -> Image.Image:
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    x = (w - im.width) // 2
    y = (h - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def make_views(row: dict[str, Any], view_w: int, view_h: int) -> dict[str, str]:
    img_path = Path(row["image_path"])
    dataset = str(row.get("dataset") or "")

    img_l = Image.open(img_path).convert("L")
    arr = np.asarray(img_l, dtype=np.uint8)

    fg = binarize(arr, dataset)
    skel = skeletonize(fg)

    original = ImageOps.autocontrast(img_l).convert("RGB")
    binary = Image.fromarray((255 - fg.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")
    skeleton = Image.fromarray((255 - skel.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")

    overlay = original.copy()
    overlay_arr = np.asarray(overlay).copy()
    overlay_arr[skel] = np.array([255, 0, 0], dtype=np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGB")

    return {
        "original": image_to_data_url(fit_to_canvas(original, view_w, view_h)),
        "binary": image_to_data_url(fit_to_canvas(binary, view_w, view_h)),
        "skeleton": image_to_data_url(fit_to_canvas(skeleton, view_w, view_h)),
        "overlay": image_to_data_url(fit_to_canvas(overlay, view_w, view_h)),
    }


def limit_per_cell(rows: list[dict[str, Any]], per_cell_limit: int) -> list[dict[str, Any]]:
    if per_cell_limit <= 0:
        return rows

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("audit_cell") or "unknown")].append(r)

    ordered_cells = [
        "A_highCER_highRisk",
        "B_highCER_lowRisk",
        "C_lowCER_highRisk",
        "D_lowCER_lowRisk",
    ]

    out = []
    for cell in ordered_cells:
        out.extend(groups.get(cell, [])[:per_cell_limit])

    for cell, group in sorted(groups.items()):
        if cell not in ordered_cells:
            out.extend(group[:per_cell_limit])

    return out


def prepare_items(rows: list[dict[str, Any]], view_w: int, view_h: int) -> list[dict[str, Any]]:
    items = []

    for i, row in enumerate(rows, 1):
        print(f"{i}/{len(rows)} {row.get('sample_id')}")

        item = {
            "sample_id": row.get("sample_id", ""),
            "dataset": row.get("dataset", ""),
            "level": row.get("level", ""),
            "category": row.get("category", ""),
            "audit_cell": row.get("audit_cell", ""),
            "cer": safe_float(row.get("cer")),
            "risk": safe_float(row.get("structural_risk_score")),
            "image_path": row.get("image_path", ""),
            "target": row.get("target", ""),
            "pred": row.get("pred", ""),
            "views": make_views(row, view_w, view_h),
        }

        items.append(item)

    return items


HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>H2 browser audit</title>
<style>
body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
  background: #f6f6f6;
  color: #111;
}
header {
  padding: 12px 18px;
  background: #111;
  color: white;
  display: flex;
  gap: 16px;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 10;
}
button, select {
  font-size: 14px;
  padding: 8px 10px;
  border: 1px solid #aaa;
  border-radius: 6px;
  background: white;
  cursor: pointer;
}
button.primary { background: #111; color: white; border-color: #111; }
button.good { background: #dff5df; }
button.warn { background: #fff0c2; }
button.bad { background: #ffdada; }
button.selected {
  outline: 3px solid #111;
}
main {
  display: grid;
  grid-template-columns: 1fr 360px;
  gap: 16px;
  padding: 16px;
}
.panel {
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 14px;
}
.views {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.view h3 {
  margin: 0 0 6px 0;
  font-size: 14px;
}
.view img {
  width: 100%;
  border: 1px solid #ccc;
  background: white;
}
.meta {
  font-size: 14px;
  line-height: 1.4;
}
.meta b {
  display: inline-block;
  min-width: 70px;
}
.textbox {
  font-size: 18px;
  line-height: 1.4;
  background: #fafafa;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #ddd;
  margin-top: 10px;
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
textarea {
  width: 100%;
  min-height: 80px;
  font-size: 14px;
}
.progress {
  font-size: 13px;
  opacity: 0.9;
}
.help {
  font-size: 13px;
  line-height: 1.35;
  background: #fafafa;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px;
}
.sample-list {
  max-height: 170px;
  overflow: auto;
  border: 1px solid #ddd;
  border-radius: 8px;
}
.sample-list button {
  display: block;
  width: 100%;
  text-align: left;
  border: 0;
  border-bottom: 1px solid #eee;
  border-radius: 0;
  background: white;
}
.sample-list button.done { background: #e8f5e9; }
.sample-list button.current { background: #e3f2fd; }
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
  <select id="cellFilter" onchange="setFilter(this.value)">
    <option value="ALL">ALL</option>
  </select>
  <button class="primary" onclick="exportCSV()">Export CSV</button>
  <button onclick="exportJSON()">Export JSON</button>
  <span class="progress" id="progress"></span>
</header>

<main>
  <section class="panel">
    <div class="meta" id="meta"></div>
    <div class="textbox" id="text"></div>

    <div class="views" style="margin-top:14px;">
      <div class="view"><h3>1. Original</h3><img id="imgOriginal"></div>
      <div class="view"><h3>2. Binary</h3><img id="imgBinary"></div>
      <div class="view"><h3>3. Skeleton</h3><img id="imgSkeleton"></div>
      <div class="view"><h3>4. Overlay, red=skeleton</h3><img id="imgOverlay"></div>
    </div>
  </section>

  <aside class="panel controls">
    <div class="help">
      <b>Смотри быстро:</b><br>
      1. Можно ли вообще audit? Если crop мусорный — Bad crop.<br>
      2. Skeleton повторяет чернила?<br>
      3. Есть critical topology error?<br>
      4. Поставь quality 0–3.<br><br>
      Не угадывай текст. Target/pred только для контекста.
    </div>

    <h3>Quick presets</h3>
    <div class="row">
      <button class="good" onclick="presetGood()">Good graph</button>
      <button class="warn" onclick="presetMinor()">Usable / minor</button>
      <button class="bad" onclick="presetBadSkeleton()">Bad skeleton</button>
      <button class="bad" onclick="presetBadCrop()">Bad crop</button>
    </div>

    <h3>Usable?</h3>
    <div class="row">
      <button data-field="audit_usable" data-value="1" onclick="setField('audit_usable','1')">usable=1</button>
      <button data-field="audit_usable" data-value="0" onclick="setField('audit_usable','0')">usable=0</button>
    </div>

    <h3>Exclusion reason</h3>
    <div class="row">
      <button data-field="exclusion_reason" data-value="ok" onclick="setField('exclusion_reason','ok')">ok</button>
      <button data-field="exclusion_reason" data-value="illegible_or_bad_crop" onclick="setField('exclusion_reason','illegible_or_bad_crop')">bad crop</button>
      <button data-field="exclusion_reason" data-value="background_dominates" onclick="setField('exclusion_reason','background_dominates')">background</button>
      <button data-field="exclusion_reason" data-value="target_ambiguous" onclick="setField('exclusion_reason','target_ambiguous')">target?</button>
      <button data-field="exclusion_reason" data-value="too_short" onclick="setField('exclusion_reason','too_short')">too short</button>
      <button data-field="exclusion_reason" data-value="non_text_fragment" onclick="setField('exclusion_reason','non_text_fragment')">non-text</button>
    </div>

    <h3>Basic checks</h3>
    <div class="row">
      <button data-field="ink_visible_ok" data-value="1" onclick="setField('ink_visible_ok','1')">ink visible yes</button>
      <button data-field="ink_visible_ok" data-value="0" onclick="setField('ink_visible_ok','0')">ink visible no</button>
    </div>
    <div class="row">
      <button data-field="skeleton_follows_ink" data-value="1" onclick="setField('skeleton_follows_ink','1')">skeleton follows yes</button>
      <button data-field="skeleton_follows_ink" data-value="0" onclick="setField('skeleton_follows_ink','0')">skeleton follows no</button>
    </div>

    <h3>Errors</h3>
    <div class="row">
      <button data-toggle-field="missed_visible_stroke" onclick="toggle01('missed_visible_stroke')">missed stroke</button>
      <button data-toggle-field="spurious_stroke" onclick="toggle01('spurious_stroke')">spurious stroke</button>
      <button data-toggle-field="endpoint_error" onclick="toggle01('endpoint_error')">endpoint</button>
      <button data-toggle-field="junction_error" onclick="toggle01('junction_error')">junction</button>
      <button data-toggle-field="loop_error" onclick="toggle01('loop_error')">loop</button>
    </div>
    <div class="row">
      <button class="bad" data-field="critical_topology_error" data-value="1" onclick="setField('critical_topology_error','1')">critical=1</button>
      <button class="good" data-field="critical_topology_error" data-value="0" onclick="setField('critical_topology_error','0')">critical=0</button>
    </div>

    <h3>Graph quality</h3>
    <div class="row">
      <button class="bad" data-field="graph_quality_0_3" data-value="0" onclick="setField('graph_quality_0_3','0')">0 unusable</button>
      <button class="warn" data-field="graph_quality_0_3" data-value="1" onclick="setField('graph_quality_0_3','1')">1 weak</button>
      <button class="warn" data-field="graph_quality_0_3" data-value="2" onclick="setField('graph_quality_0_3','2')">2 usable</button>
      <button class="good" data-field="graph_quality_0_3" data-value="3" onclick="setField('graph_quality_0_3','3')">3 good</button>
    </div>

    <h3>Notes</h3>
    <textarea id="notes" oninput="setField('notes', this.value)" placeholder="short note"></textarea>

    <div class="row">
      <button class="primary" onclick="markDoneNext()">Save + Next</button>
    </div>

    <h3>Samples</h3>
    <div class="sample-list" id="sampleList"></div>

    <p class="small">
      Keyboard: ←/→ navigation, 0/1/2/3 quality, G good, M minor, B bad skeleton, X bad crop.
    </p>
  </aside>
</main>

<script id="audit-data" type="application/json">
__DATA_JSON__
</script>

<script>
const data = JSON.parse(document.getElementById("audit-data").textContent);
const items = data.items;
const fields = data.annotation_fields;
const storageKey = "h2_audit_browser_" + data.storage_id;

let annotations = JSON.parse(localStorage.getItem(storageKey) || "{}");
let filter = "ALL";
let visible = items.map((_, i) => i);
let currentVisibleIndex = 0;

function defaultAnn(item) {
  const a = {
    sample_id: item.sample_id,
    dataset: item.dataset,
    level: item.level,
    category: item.category,
    audit_cell: item.audit_cell,
    cer: item.cer,
    structural_risk_score: item.risk,
    image_path: item.image_path,
    target: item.target,
    pred: item.pred
  };
  for (const f of fields) a[f] = "";
  return a;
}

function annFor(item) {
  if (!annotations[item.sample_id]) annotations[item.sample_id] = defaultAnn(item);
  return annotations[item.sample_id];
}

function save() {
  localStorage.setItem(storageKey, JSON.stringify(annotations));
  updateProgress();
}

function currentIndex() {
  return visible[currentVisibleIndex] ?? 0;
}

function currentItem() {
  return items[currentIndex()];
}

function currentAnn() {
  return annFor(currentItem());
}

function setField(k, v) {
  currentAnn()[k] = String(v);
  save();
  render();
}

function toggle01(k) {
  const a = currentAnn();
  a[k] = a[k] === "1" ? "0" : "1";
  save();
  render();
}

function presetGood() {
  const a = currentAnn();
  Object.assign(a, {
    audit_usable: "1",
    exclusion_reason: "ok",
    ink_visible_ok: "1",
    skeleton_follows_ink: "1",
    missed_visible_stroke: "0",
    spurious_stroke: "0",
    endpoint_error: "0",
    junction_error: "0",
    loop_error: "0",
    critical_topology_error: "0",
    graph_quality_0_3: "3"
  });
  save(); render();
}

function presetMinor() {
  const a = currentAnn();
  Object.assign(a, {
    audit_usable: "1",
    exclusion_reason: "ok",
    ink_visible_ok: "1",
    skeleton_follows_ink: "1",
    critical_topology_error: "0",
    graph_quality_0_3: "2"
  });
  save(); render();
}

function presetBadSkeleton() {
  const a = currentAnn();
  Object.assign(a, {
    audit_usable: "1",
    exclusion_reason: "ok",
    ink_visible_ok: "1",
    skeleton_follows_ink: "0",
    critical_topology_error: "1",
    graph_quality_0_3: "1"
  });
  save(); render();
}

function presetBadCrop() {
  const a = currentAnn();
  Object.assign(a, {
    audit_usable: "0",
    exclusion_reason: "illegible_or_bad_crop",
    ink_visible_ok: "0",
    skeleton_follows_ink: "0",
    missed_visible_stroke: "0",
    spurious_stroke: "1",
    endpoint_error: "0",
    junction_error: "0",
    loop_error: "0",
    critical_topology_error: "1",
    graph_quality_0_3: "0"
  });
  save(); render();
}

function isDone(a) {
  return a.audit_usable !== "" && a.graph_quality_0_3 !== "" && a.critical_topology_error !== "";
}

function updateProgress() {
  let done = 0;
  for (const item of items) {
    if (annotations[item.sample_id] && isDone(annotations[item.sample_id])) done += 1;
  }
  document.getElementById("progress").textContent = `${done}/${items.length} annotated`;
}

function setFilter(v) {
  filter = v;
  visible = items.map((_, i) => i).filter(i => filter === "ALL" || items[i].audit_cell === filter);
  currentVisibleIndex = 0;
  render();
}

function prevItem() {
  currentVisibleIndex = Math.max(0, currentVisibleIndex - 1);
  render();
}

function nextItem() {
  currentVisibleIndex = Math.min(visible.length - 1, currentVisibleIndex + 1);
  render();
}

function markDoneNext() {
  save();
  nextItem();
}

function renderButtons() {
  const a = currentAnn();
  document.querySelectorAll("button").forEach(b => b.classList.remove("selected"));

  document.querySelectorAll("button[data-field][data-value]").forEach((b) => {
    if (a[b.dataset.field] === b.dataset.value) b.classList.add("selected");
  });
  document.querySelectorAll("button[data-toggle-field]").forEach((b) => {
    if (a[b.dataset.toggleField] === "1") b.classList.add("selected");
  });
  document.getElementById("notes").value = a.notes || "";
}

function renderSampleList() {
  const root = document.getElementById("sampleList");
  root.innerHTML = "";

  for (let vi = 0; vi < visible.length; vi++) {
    const idx = visible[vi];
    const item = items[idx];
    const a = annFor(item);
    const btn = document.createElement("button");
    btn.textContent = `${vi + 1}. ${item.audit_cell} | ${item.sample_id}`;
    if (isDone(a)) btn.classList.add("done");
    if (vi === currentVisibleIndex) btn.classList.add("current");
    btn.onclick = () => { currentVisibleIndex = vi; render(); };
    root.appendChild(btn);
  }
}

function render() {
  const item = currentItem();
  const a = currentAnn();

  document.getElementById("meta").innerHTML = `
    <b>cell:</b> ${escapeHtml(item.audit_cell)}<br>
    <b>id:</b> ${escapeHtml(item.sample_id)}<br>
    <b>dataset:</b> ${escapeHtml(item.dataset)} | ${escapeHtml(item.level)} | ${escapeHtml(item.category)}<br>
    <b>CER:</b> ${Number(item.cer).toFixed(3)} &nbsp; <b>risk:</b> ${Number(item.risk).toFixed(3)}<br>
    <b>image:</b> <span class="small">${escapeHtml(item.image_path)}</span>
  `;

  document.getElementById("text").innerHTML = `
    <b>Target:</b> ${escapeHtml(item.target)}<br>
    <b>Pred:</b> ${escapeHtml(item.pred)}
  `;

  document.getElementById("imgOriginal").src = item.views.original;
  document.getElementById("imgBinary").src = item.views.binary;
  document.getElementById("imgSkeleton").src = item.views.skeleton;
  document.getElementById("imgOverlay").src = item.views.overlay;

  renderButtons();
  renderSampleList();
  updateProgress();
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function csvEscape(v) {
  const s = String(v ?? "");
  if (/[",\n\r]/.test(s)) return '"' + s.replaceAll('"', '""') + '"';
  return s;
}

function exportCSV() {
  const baseFields = [
    "sample_id","dataset","level","category","audit_cell",
    "cer","structural_risk_score","image_path","target","pred"
  ];
  const cols = baseFields.concat(fields);
  const lines = [cols.join(",")];

  for (const item of items) {
    const a = annFor(item);
    lines.push(cols.map(c => csvEscape(a[c])).join(","));
  }

  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv;charset=utf-8"});
  downloadBlob(blob, data.export_csv_name);
}

function exportJSON() {
  const arr = items.map(item => annFor(item));
  const blob = new Blob([JSON.stringify(arr, null, 2)], {type: "application/json;charset=utf-8"});
  downloadBlob(blob, data.export_json_name);
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function setupFilters() {
  const sel = document.getElementById("cellFilter");
  const cells = [...new Set(items.map(x => x.audit_cell))].sort();
  for (const c of cells) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  }
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;

  if (e.key === "ArrowRight") nextItem();
  if (e.key === "ArrowLeft") prevItem();

  if (e.key === "0") setField("graph_quality_0_3", "0");
  if (e.key === "1") setField("graph_quality_0_3", "1");
  if (e.key === "2") setField("graph_quality_0_3", "2");
  if (e.key === "3") setField("graph_quality_0_3", "3");

  if (e.key.toLowerCase() === "g") presetGood();
  if (e.key.toLowerCase() === "m") presetMinor();
  if (e.key.toLowerCase() === "b") presetBadSkeleton();
  if (e.key.toLowerCase() === "x") presetBadCrop();
});

setupFilters();
render();
</script>

</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", required=True)
    parser.add_argument("--out_html", required=True)
    parser.add_argument("--per_cell_limit", type=int, default=10)
    parser.add_argument("--view_w", type=int, default=620)
    parser.add_argument("--view_h", type=int, default=180)
    args = parser.parse_args()

    rows = read_csv(Path(args.candidates_csv))
    rows = limit_per_cell(rows, args.per_cell_limit)

    items = prepare_items(rows, view_w=args.view_w, view_h=args.view_h)

    data = {
        "items": items,
        "annotation_fields": ANNOTATION_FIELDS,
        "storage_id": Path(args.out_html).stem,
        "export_csv_name": f"{Path(args.out_html).stem}_annotations.csv",
        "export_json_name": f"{Path(args.out_html).stem}_annotations.json",
    }

    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    html_text = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")

    print("wrote:", out_html)
    print("items:", len(items))


if __name__ == "__main__":
    main()
