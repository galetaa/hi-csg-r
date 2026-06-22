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
from PIL import Image, ImageDraw, ImageOps
from skimage.morphology import skeletonize

from src.preprocessing.school_rectangular_v2 import (
    SchoolCocoSource,
    extract_school_lineaware_v3,
)


ANNOTATION_FIELDS = [
    "usable",
    "ink_loss",
    "line_residual",
    "neighbor_text_removed",
    "skeleton_follows_ink",
    "notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_manifest_index(root: Path) -> dict[str, dict[str, Any]]:
    by_id = {}

    for split in ["train", "val", "test"]:
        path = root / f"{split}.jsonl"

        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))

            if sample_id:
                by_id[sample_id] = row

    return by_id


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fg_to_img(fg: np.ndarray) -> Image.Image:
    return Image.fromarray(
        (255 - fg.astype(np.uint8) * 255).astype(np.uint8),
        mode="L",
    ).convert("RGB")


def overlay_skeleton(gray: np.ndarray, skel: np.ndarray) -> Image.Image:
    base = Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")
    rgb = np.asarray(base).copy()
    rgb[skel] = np.array([255, 0, 0], dtype=np.uint8)
    return Image.fromarray(rgb)


def ruling_to_img(ruling_response: np.ndarray) -> Image.Image:
    arr = np.asarray(ruling_response, dtype=np.float32)

    if arr.size == 0 or float(arr.max()) <= 0.0:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        high = max(float(np.quantile(arr, 0.98)), 1.0)
        norm = np.clip(arr * 255.0 / high, 0, 255).astype(np.uint8)

    return Image.fromarray(norm, mode="L").convert("RGB")


def make_panel(
    *,
    title: str,
    gray_rectangular: np.ndarray,
    normalized_gray: np.ndarray,
    foreground: np.ndarray,
    ruling_response: np.ndarray,
    w: int,
    h: int,
) -> Image.Image:
    skel = skeletonize(foreground.astype(bool))

    panels = [
        ("rect", ImageOps.autocontrast(Image.fromarray(gray_rectangular).convert("RGB"))),
        ("norm", Image.fromarray(normalized_gray).convert("RGB")),
        ("fg", fg_to_img(foreground)),
        ("skel", overlay_skeleton(normalized_gray, skel)),
        ("ruling", ruling_to_img(ruling_response)),
    ]

    header_h = 42
    cell_w = max(1, w // len(panels))
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((8, 5), title[:96], fill="black")
    draw.text(
        (8, 23),
        f"fg={foreground.mean():.3f} skel={skel.mean():.3f}",
        fill="black",
    )

    for i, (label, img) in enumerate(panels):
        x = i * cell_w
        draw.text((x + 6, header_h), label, fill="black")
        canvas.paste(
            fit(img, cell_w, h - header_h - 18),
            (x, header_h + 18),
        )
        draw.rectangle(
            (x, header_h, min(w - 1, x + cell_w - 1), h - 1),
            outline="#cccccc",
        )

    draw.rectangle((0, 0, w - 1, h - 1), outline="black")
    return canvas


def prepare_items(
    *,
    candidates: list[dict[str, str]],
    manifest_index: dict[str, dict[str, Any]],
    coco: SchoolCocoSource,
    panel_w: int,
    panel_h: int,
) -> list[dict[str, Any]]:
    items = []

    for index, candidate in enumerate(candidates):
        sample_id = str(candidate["sample_id"])
        raw_row = manifest_index.get(sample_id)

        if raw_row is None:
            raise KeyError(f"No manifest row for {sample_id}")

        extracted = extract_school_lineaware_v3(
            raw_row,
            coco,
        )

        title = (
            f"{index + 1:03d} {candidate['validation_group']} "
            f"{sample_id} target={candidate.get('target', '')}"
        )

        panel = make_panel(
            title=title,
            gray_rectangular=extracted["gray_rectangular"],
            normalized_gray=extracted["normalized_gray"],
            foreground=extracted["foreground"],
            ruling_response=extracted["ruling_response"],
            w=panel_w,
            h=panel_h,
        )

        items.append({
            "index": index,
            "sample_id": sample_id,
            "validation_group": candidate.get("validation_group", ""),
            "target": candidate.get("target", ""),
            "split": candidate.get("split", ""),
            "metrics": {
                key: candidate.get(key, "")
                for key in [
                    "old_fg_fraction",
                    "new_fg_fraction",
                    "delta_fg_fraction",
                    "old_dir_h_frac",
                    "new_dir_h_frac",
                    "delta_dir_h_frac",
                    "ruling_response_mean",
                    "ruling_response_p95",
                    "feature_change_score",
                    "selection_reason",
                ]
            },
            "panel": data_url(panel),
        })

    return items


def make_html(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, ensure_ascii=False)
    fields = json.dumps(ANNOTATION_FIELDS)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>School lineaware v3 quality gate</title>
<style>
body {{
  margin: 0;
  font-family: system-ui, sans-serif;
  background: #f6f6f6;
  color: #111;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: #111;
  color: #fff;
}}
button {{
  border: 1px solid #999;
  background: #fff;
  color: #111;
  border-radius: 6px;
  padding: 7px 10px;
  cursor: pointer;
}}
button.active {{
  background: #111;
  color: #fff;
  border-color: #111;
}}
button.primary {{
  border-color: #fff;
}}
main {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 14px;
  padding: 14px;
}}
.panel {{
  background: #fff;
  border: 1px solid #ccc;
}}
.panel img {{
  width: 100%;
  display: block;
}}
.side {{
  position: sticky;
  top: 58px;
  align-self: start;
  background: #fff;
  border: 1px solid #ccc;
  padding: 12px;
}}
.field {{
  margin: 12px 0;
}}
.field label {{
  display: block;
  font-weight: 650;
  margin-bottom: 6px;
}}
.field .row {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}}
textarea {{
  width: 100%;
  min-height: 76px;
  box-sizing: border-box;
}}
pre {{
  white-space: pre-wrap;
  font-size: 12px;
  background: #f0f0f0;
  padding: 8px;
}}
</style>
</head>
<body>
<header>
  <button class="primary" onclick="prevItem()">Prev</button>
  <button class="primary" onclick="nextItem()">Next</button>
  <span id="counter"></span>
  <button class="primary" onclick="exportCsv()">Export CSV</button>
  <span id="progress"></span>
</header>
<main>
  <section class="panel"><img id="panel" alt=""></section>
  <aside class="side">
    <h2 id="title"></h2>
    <pre id="meta"></pre>
    <div id="fields"></div>
  </aside>
</main>
<script>
const items = {payload};
const fields = {fields};
const storageKey = "school_lineaware_quality_gate_iter2";
let index = 0;
let annotations = JSON.parse(localStorage.getItem(storageKey) || "{{}}");

function ensure(sampleId) {{
  if (!annotations[sampleId]) {{
    annotations[sampleId] = {{}};
    for (const f of fields) annotations[sampleId][f] = "";
  }}
  return annotations[sampleId];
}}

function save() {{
  localStorage.setItem(storageKey, JSON.stringify(annotations));
  updateProgress();
}}

function setField(field, value) {{
  const item = items[index];
  const a = ensure(item.sample_id);
  a[field] = value;
  save();
  render();
}}

function setNotes(value) {{
  const item = items[index];
  const a = ensure(item.sample_id);
  a.notes = value;
  save();
}}

function fieldButtons(field) {{
  const item = items[index];
  const a = ensure(item.sample_id);
  return `<div class="field">
    <label>${{field}}</label>
    <div class="row">
      <button class="${{a[field] === "1" ? "active" : ""}}" onclick="setField('${{field}}','1')">yes</button>
      <button class="${{a[field] === "0" ? "active" : ""}}" onclick="setField('${{field}}','0')">no</button>
      <button class="${{a[field] === "na" ? "active" : ""}}" onclick="setField('${{field}}','na')">n/a</button>
    </div>
  </div>`;
}}

function render() {{
  const item = items[index];
  const a = ensure(item.sample_id);
  document.getElementById("panel").src = item.panel;
  document.getElementById("title").textContent = `${{index + 1}} / ${{items.length}}  ${{item.sample_id}}`;
  document.getElementById("counter").textContent = `${{item.validation_group}}`;
  document.getElementById("meta").textContent = JSON.stringify({{
    split: item.split,
    target: item.target,
    metrics: item.metrics
  }}, null, 2);
  document.getElementById("fields").innerHTML =
    fields.filter(f => f !== "notes").map(fieldButtons).join("") +
    `<div class="field"><label>notes</label><textarea oninput="setNotes(this.value)">${{a.notes || ""}}</textarea></div>`;
  updateProgress();
}}

function complete(a) {{
  return fields
    .filter(f => f !== "notes")
    .every(f => a && a[f] !== "");
}}

function updateProgress() {{
  let n = 0;
  for (const item of items) {{
    if (complete(annotations[item.sample_id])) n++;
  }}
  document.getElementById("progress").textContent = `${{n}}/${{items.length}} complete`;
}}

function prevItem() {{
  index = Math.max(0, index - 1);
  render();
}}

function nextItem() {{
  index = Math.min(items.length - 1, index + 1);
  render();
}}

function csvEscape(value) {{
  const s = String(value ?? "");
  return '"' + s.replaceAll('"', '""') + '"';
}}

function exportCsv() {{
  const columns = [
    "sample_id",
    "validation_group",
    "split",
    "target",
    ...Object.keys(items[0].metrics),
    ...fields
  ];
  const lines = [columns.join(",")];
  for (const item of items) {{
    const a = ensure(item.sample_id);
    const row = {{
      sample_id: item.sample_id,
      validation_group: item.validation_group,
      split: item.split,
      target: item.target,
      ...item.metrics,
      ...a
    }};
    lines.push(columns.map(c => csvEscape(row[c])).join(","));
  }}
  const blob = new Blob([lines.join("\\n") + "\\n"], {{type: "text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "lineaware_quality_gate_annotations.csv";
  link.click();
  URL.revokeObjectURL(url);
}}

document.addEventListener("keydown", event => {{
  if (event.key === "ArrowLeft") prevItem();
  if (event.key === "ArrowRight") nextItem();
}});

render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", required=True)
    parser.add_argument("--manifest_root", required=True)
    parser.add_argument("--school_raw_dir", required=True)
    parser.add_argument("--out_html", required=True)
    parser.add_argument("--panel_w", type=int, default=1200)
    parser.add_argument("--panel_h", type=int, default=290)
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidates_csv))
    manifest_index = load_manifest_index(Path(args.manifest_root))
    coco = SchoolCocoSource(args.school_raw_dir)

    items = prepare_items(
        candidates=candidates,
        manifest_index=manifest_index,
        coco=coco,
        panel_w=args.panel_w,
        panel_h=args.panel_h,
    )

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        make_html(items),
        encoding="utf-8",
    )

    print(json.dumps({
        "candidates": args.candidates_csv,
        "manifest_root": args.manifest_root,
        "school_raw_dir": args.school_raw_dir,
        "out_html": str(out_html),
        "n": len(items),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
