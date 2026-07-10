from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


SOURCE = Path("outputs/h2_gold_audit_v1/annotations/annotation_100_filled.csv")
OUT_DIR = Path("outputs/htr_publication_v3/independent_annotation_v1")
BROWSER_SOURCE = Path("outputs/h2_gold_audit_v1/browser_audit_100.html")

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

BLIND_COLUMNS = [
    "annotation_id",
    "sample_id",
    "dataset",
    "level",
    "category",
    "image_path",
    "annotator",
    *ANNOTATION_FIELDS,
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def blind_rows(rows: list[dict[str, Any]], *, seed: int, n: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    if n > 0:
        selected = selected[:n]

    out = []
    for idx, row in enumerate(selected, start=1):
        blinded = {
            "annotation_id": f"iaa2_{idx:03d}",
            "sample_id": row.get("sample_id", ""),
            "dataset": row.get("dataset", ""),
            "level": row.get("level", ""),
            "category": row.get("category", ""),
            "image_path": row.get("image_path", ""),
            "annotator": "",
        }
        for field in ANNOTATION_FIELDS:
            blinded[field] = ""
        out.append(blinded)
    return out


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def fit_to_canvas(img: Image.Image, w: int, h: int) -> Image.Image:
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    x = (w - im.width) // 2
    y = (h - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def binarize(arr: np.ndarray, dataset: str) -> np.ndarray:
    ink = 255 - arr
    if ink.max() == ink.min():
        return np.zeros_like(ink, dtype=np.uint8)

    if dataset == "school_notebooks_clean":
        h, w = ink.shape
        block = max(3, min(25, h, w))
        if block % 2 == 0:
            block -= 1
        return cv2.adaptiveThreshold(
            ink,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            -2,
        ) > 0

    _, out = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out > 0


def skeletonize_cv(mask: np.ndarray) -> np.ndarray:
    binary = (mask.astype(np.uint8) * 255)
    thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
    if thinning is not None:
        return thinning(binary) > 0

    skel = np.zeros(binary.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary.copy()
    while cv2.countNonZero(current) > 0:
        eroded = cv2.erode(current, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(current, opened)
        skel = cv2.bitwise_or(skel, temp)
        current = eroded
    return skel > 0


def make_views(row: dict[str, Any], view_w: int, view_h: int) -> dict[str, str]:
    img_path = Path(str(row["image_path"]))
    dataset = str(row.get("dataset") or "")

    img_l = Image.open(img_path).convert("L")
    arr = np.asarray(img_l, dtype=np.uint8)

    fg = binarize(arr, dataset)
    skel = skeletonize_cv(fg)

    original = ImageOps.autocontrast(img_l).convert("RGB")
    binary = Image.fromarray((255 - fg.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")
    skeleton = Image.fromarray((255 - skel.astype(np.uint8) * 255).astype(np.uint8), mode="L").convert("RGB")

    overlay_arr = np.asarray(original).copy()
    overlay_arr[skel] = np.array([255, 0, 0], dtype=np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGB")

    return {
        "original": image_to_data_url(fit_to_canvas(original, view_w, view_h)),
        "binary": image_to_data_url(fit_to_canvas(binary, view_w, view_h)),
        "skeleton": image_to_data_url(fit_to_canvas(skeleton, view_w, view_h)),
        "overlay": image_to_data_url(fit_to_canvas(overlay, view_w, view_h)),
    }


def load_precomputed_views(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="audit-data" type="application/json">\s*(.*?)\s*</script>',
        text,
        flags=re.DOTALL,
    )
    if not match:
        return {}

    data = json.loads(match.group(1))
    views_by_sample = {}
    for item in data.get("items", []):
        sample_id = str(item.get("sample_id", ""))
        views = item.get("views")
        if sample_id and isinstance(views, dict):
            views_by_sample[sample_id] = views
    return views_by_sample


def browser_items(
    rows: list[dict[str, Any]],
    *,
    view_w: int,
    view_h: int,
    precomputed_views: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    items = []
    for idx, row in enumerate(rows, start=1):
        print(f"{idx}/{len(rows)} {row['annotation_id']} {row['sample_id']}", flush=True)
        views = precomputed_views.get(row["sample_id"])
        if views is None:
            views = make_views(row, view_w, view_h)
        items.append({
            "annotation_id": row["annotation_id"],
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "level": row["level"],
            "category": row["category"],
            "image_path": row["image_path"],
            "views": views,
        })
    return items


HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Independent Blind Annotation v1</title>
<style>
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f6f6; color: #111; }
header { padding: 12px 18px; background: #111; color: white; display: flex; gap: 12px; align-items: center; position: sticky; top: 0; z-index: 10; }
button { font-size: 14px; padding: 8px 10px; border: 1px solid #aaa; border-radius: 6px; background: white; cursor: pointer; }
button.primary { background: #111; color: white; border-color: white; }
button.good { background: #dff5df; }
button.warn { background: #fff0c2; }
button.bad { background: #ffdada; }
button.selected { outline: 3px solid #111; }
main { display: grid; grid-template-columns: 1fr 360px; gap: 16px; padding: 16px; }
.panel { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
.views { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.view h3 { margin: 0 0 6px 0; font-size: 14px; }
.view img { width: 100%; border: 1px solid #ccc; background: white; }
.meta { font-size: 14px; line-height: 1.45; }
.meta b { display: inline-block; min-width: 92px; }
.row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.controls h3 { margin: 14px 0 6px; }
textarea, input { width: 100%; box-sizing: border-box; font-size: 14px; }
textarea { min-height: 80px; }
.sample-list { max-height: 180px; overflow: auto; border: 1px solid #ddd; border-radius: 8px; }
.sample-list button { display: block; width: 100%; text-align: left; border: 0; border-bottom: 1px solid #eee; border-radius: 0; background: white; }
.sample-list button.done { background: #e8f5e9; }
.sample-list button.current { background: #e3f2fd; }
.progress, .small { font-size: 12px; color: #555; }
.help { font-size: 13px; line-height: 1.35; background: #fafafa; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
</style>
</head>
<body>
<header>
  <button onclick="prevItem()">Prev</button>
  <button onclick="nextItem()">Next</button>
  <button class="primary" onclick="exportCSV()">Export CSV</button>
  <span id="progress" class="progress"></span>
</header>

<main>
  <section class="panel">
    <div class="meta" id="meta"></div>
    <div class="views" style="margin-top:14px;">
      <div class="view"><h3>Original</h3><img id="imgOriginal"></div>
      <div class="view"><h3>Binary</h3><img id="imgBinary"></div>
      <div class="view"><h3>Skeleton</h3><img id="imgSkeleton"></div>
      <div class="view"><h3>Overlay, red=skeleton</h3><img id="imgOverlay"></div>
    </div>
  </section>

  <aside class="panel controls">
    <div class="help">
      Blind second-pass annotation. Do not use previous annotations, CER, model prediction, or risk score.
      Assess only visual structural quality.
    </div>

    <h3>Annotator</h3>
    <input id="annotator" oninput="setField('annotator', this.value)" placeholder="annotator id">

    <h3>Quick Presets</h3>
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

    <h3>Exclusion Reason</h3>
    <div class="row">
      <button data-field="exclusion_reason" data-value="ok" onclick="setField('exclusion_reason','ok')">ok</button>
      <button data-field="exclusion_reason" data-value="illegible_or_bad_crop" onclick="setField('exclusion_reason','illegible_or_bad_crop')">bad crop</button>
      <button data-field="exclusion_reason" data-value="background_dominates" onclick="setField('exclusion_reason','background_dominates')">background</button>
      <button data-field="exclusion_reason" data-value="target_ambiguous" onclick="setField('exclusion_reason','target_ambiguous')">ambiguous</button>
      <button data-field="exclusion_reason" data-value="too_short" onclick="setField('exclusion_reason','too_short')">too short</button>
      <button data-field="exclusion_reason" data-value="non_text_fragment" onclick="setField('exclusion_reason','non_text_fragment')">non-text</button>
    </div>

    <h3>Basic Checks</h3>
    <div class="row">
      <button data-field="ink_visible_ok" data-value="1" onclick="setField('ink_visible_ok','1')">ink yes</button>
      <button data-field="ink_visible_ok" data-value="0" onclick="setField('ink_visible_ok','0')">ink no</button>
    </div>
    <div class="row">
      <button data-field="skeleton_follows_ink" data-value="1" onclick="setField('skeleton_follows_ink','1')">skeleton yes</button>
      <button data-field="skeleton_follows_ink" data-value="0" onclick="setField('skeleton_follows_ink','0')">skeleton no</button>
    </div>

    <h3>Error Flags</h3>
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

    <h3>Graph Quality</h3>
    <div class="row">
      <button class="bad" data-field="graph_quality_0_3" data-value="0" onclick="setField('graph_quality_0_3','0')">0</button>
      <button class="warn" data-field="graph_quality_0_3" data-value="1" onclick="setField('graph_quality_0_3','1')">1</button>
      <button class="warn" data-field="graph_quality_0_3" data-value="2" onclick="setField('graph_quality_0_3','2')">2</button>
      <button class="good" data-field="graph_quality_0_3" data-value="3" onclick="setField('graph_quality_0_3','3')">3</button>
    </div>

    <h3>Notes</h3>
    <textarea id="notes" oninput="setField('notes', this.value)" placeholder="short note"></textarea>
    <div class="row"><button class="primary" onclick="markDoneNext()">Save + Next</button></div>

    <h3>Samples</h3>
    <div class="sample-list" id="sampleList"></div>
    <p class="small">Keyboard: arrows, 0/1/2/3 quality, G good, M minor, B bad skeleton, X bad crop.</p>
  </aside>
</main>

<script id="audit-data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("audit-data").textContent);
const items = data.items;
const fields = data.annotation_fields;
const storageKey = "independent_annotation_" + data.storage_id;
let annotations = JSON.parse(localStorage.getItem(storageKey) || "{}");
let currentIndexValue = 0;

function defaultAnn(item) {
  const a = {
    annotation_id: item.annotation_id,
    sample_id: item.sample_id,
    dataset: item.dataset,
    level: item.level,
    category: item.category,
    image_path: item.image_path,
    annotator: ""
  };
  for (const f of fields) a[f] = "";
  return a;
}
function annFor(item) {
  if (!annotations[item.annotation_id]) annotations[item.annotation_id] = defaultAnn(item);
  return annotations[item.annotation_id];
}
function save() {
  localStorage.setItem(storageKey, JSON.stringify(annotations));
  updateProgress();
}
function currentItem() { return items[currentIndexValue]; }
function currentAnn() { return annFor(currentItem()); }
function setField(k, v) { currentAnn()[k] = String(v); save(); render(); }
function toggle01(k) {
  const a = currentAnn();
  a[k] = a[k] === "1" ? "0" : "1";
  save(); render();
}
function presetGood() {
  Object.assign(currentAnn(), {
    audit_usable: "1", exclusion_reason: "ok", ink_visible_ok: "1", skeleton_follows_ink: "1",
    missed_visible_stroke: "0", spurious_stroke: "0", endpoint_error: "0", junction_error: "0",
    loop_error: "0", critical_topology_error: "0", graph_quality_0_3: "3"
  });
  save(); render();
}
function presetMinor() {
  Object.assign(currentAnn(), {
    audit_usable: "1", exclusion_reason: "ok", ink_visible_ok: "1", skeleton_follows_ink: "1",
    critical_topology_error: "0", graph_quality_0_3: "2"
  });
  save(); render();
}
function presetBadSkeleton() {
  Object.assign(currentAnn(), {
    audit_usable: "1", exclusion_reason: "ok", ink_visible_ok: "1", skeleton_follows_ink: "0",
    critical_topology_error: "1", graph_quality_0_3: "1"
  });
  save(); render();
}
function presetBadCrop() {
  Object.assign(currentAnn(), {
    audit_usable: "0", exclusion_reason: "illegible_or_bad_crop", ink_visible_ok: "0", skeleton_follows_ink: "0",
    missed_visible_stroke: "0", spurious_stroke: "1", endpoint_error: "0", junction_error: "0",
    loop_error: "0", critical_topology_error: "1", graph_quality_0_3: "0"
  });
  save(); render();
}
function isDone(a) {
  return a.audit_usable !== "" && a.graph_quality_0_3 !== "" && a.critical_topology_error !== "";
}
function updateProgress() {
  let done = 0;
  for (const item of items) if (annotations[item.annotation_id] && isDone(annotations[item.annotation_id])) done += 1;
  document.getElementById("progress").textContent = `${done}/${items.length} annotated`;
}
function prevItem() { currentIndexValue = Math.max(0, currentIndexValue - 1); render(); }
function nextItem() { currentIndexValue = Math.min(items.length - 1, currentIndexValue + 1); render(); }
function markDoneNext() { save(); nextItem(); }
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
  document.getElementById("annotator").value = a.annotator || "";
}
function renderSampleList() {
  const root = document.getElementById("sampleList");
  root.innerHTML = "";
  items.forEach((item, idx) => {
    const b = document.createElement("button");
    b.textContent = `${idx + 1}. ${item.annotation_id}`;
    if (idx === currentIndexValue) b.classList.add("current");
    if (annotations[item.annotation_id] && isDone(annotations[item.annotation_id])) b.classList.add("done");
    b.onclick = () => { currentIndexValue = idx; render(); };
    root.appendChild(b);
  });
}
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function render() {
  const item = currentItem();
  annFor(item);
  document.getElementById("meta").innerHTML = `
    <b>annotation:</b> ${escapeHtml(item.annotation_id)}<br>
    <b>sample:</b> ${escapeHtml(item.sample_id)}<br>
    <b>dataset:</b> ${escapeHtml(item.dataset)}<br>
    <b>level:</b> ${escapeHtml(item.level)}<br>
    <b>category:</b> ${escapeHtml(item.category)}<br>
    <b>image:</b> <span class="small">${escapeHtml(item.image_path)}</span>`;
  document.getElementById("imgOriginal").src = item.views.original;
  document.getElementById("imgBinary").src = item.views.binary;
  document.getElementById("imgSkeleton").src = item.views.skeleton;
  document.getElementById("imgOverlay").src = item.views.overlay;
  renderButtons();
  renderSampleList();
  updateProgress();
}
function csvEscape(v) {
  const s = String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function exportCSV() {
  const cols = data.export_columns;
  const rows = items.map(item => annFor(item));
  const csv = [cols.join(",")].concat(rows.map(row => cols.map(c => csvEscape(row[c])).join(","))).join("\n");
  const blob = new Blob([csv + "\n"], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = data.export_csv_name;
  a.click();
  URL.revokeObjectURL(url);
}
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") prevItem();
  if (e.key === "ArrowRight") nextItem();
  if (["0","1","2","3"].includes(e.key)) setField("graph_quality_0_3", e.key);
  if (e.key.toLowerCase() === "g") presetGood();
  if (e.key.toLowerCase() === "m") presetMinor();
  if (e.key.toLowerCase() === "b") presetBadSkeleton();
  if (e.key.toLowerCase() === "x") presetBadCrop();
});
render();
</script>
</body>
</html>
"""


def build_protocol(rows_n: int) -> str:
    return f"""# Independent Annotation v1 Protocol

Purpose: formal second-pass annotation for inter-annotator agreement on structural HTR preprocessing quality.

Samples: {rows_n} rows from `outputs/h2_gold_audit_v1/annotations/annotation_100_filled.csv`, shuffled with a fixed seed.

Blinding:
- Previous labels are not shown.
- CER, model prediction, target text, structural risk score, and audit-cell stratum are not shown.
- The annotator sees only sample identity, dataset/level/category, image path, and four visual views.

Fields:
- `audit_usable`: 1 if the sample can be audited, 0 otherwise.
- `exclusion_reason`: `ok`, `illegible_or_bad_crop`, `background_dominates`, `target_ambiguous`, `too_short`, or `non_text_fragment`.
- `ink_visible_ok`: 1 if main ink is visible/preserved.
- `skeleton_follows_ink`: 1 if skeleton follows the main ink strokes.
- `missed_visible_stroke`, `spurious_stroke`, `endpoint_error`, `junction_error`, `loop_error`: binary error flags.
- `critical_topology_error`: 1 if the structural representation has a major topology defect.
- `graph_quality_0_3`: 0 unusable, 1 weak, 2 usable, 3 good.

Scoring:
- Fill/export the browser CSV as `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_second_filled.csv`.
- Run `python tools/score_independent_annotation_v1.py`.
- The scorer writes agreement, Cohen kappa, and weighted kappa for `graph_quality_0_3`.

Publication boundary:
- This supports formal IAA only if the second annotator is genuinely independent from the first annotation pass.
- AI-generated or same-person repeated annotation must be reported as repeated consistency, not independent IAA.
"""


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(Path(args.source))
    blinded = blind_rows(rows, seed=args.seed, n=args.n)
    precomputed_views = load_precomputed_views(Path(args.browser_source))
    items = browser_items(
        blinded,
        view_w=args.view_w,
        view_h=args.view_h,
        precomputed_views=precomputed_views,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_csv = out_dir / "blind_annotation_second_template.csv"
    write_csv(blinded, template_csv, BLIND_COLUMNS)

    data = {
        "items": items,
        "annotation_fields": ANNOTATION_FIELDS,
        "export_columns": BLIND_COLUMNS,
        "storage_id": "independent_annotation_v1",
        "export_csv_name": "blind_annotation_second_filled.csv",
    }
    browser = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    browser_path = out_dir / "blind_annotation_browser.html"
    browser_path.write_text(browser, encoding="utf-8")

    protocol_path = out_dir / "protocol.md"
    protocol_path.write_text(build_protocol(len(blinded)), encoding="utf-8")

    manifest_path = out_dir / "blind_annotation_manifest.json"
    manifest_path.write_text(json.dumps({
        "source": str(args.source),
        "browser_source": str(args.browser_source),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "n": len(blinded),
        "precomputed_views_used": sum(1 for row in blinded if row["sample_id"] in precomputed_views),
        "template_csv": str(template_csv),
        "browser": str(browser_path),
        "protocol": str(protocol_path),
        "expected_filled_csv": str(out_dir / "blind_annotation_second_filled.csv"),
        "blinding": {
            "hidden_from_browser": [
                "previous labels",
                "cer",
                "pred",
                "target",
                "structural_risk_score",
                "audit_cell",
            ],
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "template_csv": str(template_csv),
        "browser": str(browser_path),
        "protocol": str(protocol_path),
        "manifest": str(manifest_path),
        "n": len(blinded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--browser_source", default=str(BROWSER_SOURCE))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--view_w", type=int, default=620)
    parser.add_argument("--view_h", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(build_pack(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
