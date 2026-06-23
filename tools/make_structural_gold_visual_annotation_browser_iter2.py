from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import remove_small_objects, skeletonize

from src.preprocessing.school_rectangular_v2 import (
    SchoolCocoSource,
    extract_school_lineaware_v3,
)
from tools.extract_htr_graph_features import binarize


FIELDS = [
    "sample_id",
    "stratum",
    "dataset",
    "token_type",
    "structural_usable",
    "foreground_ok",
    "skeleton_ok",
    "graph_ok",
    "line_residual",
    "neighbor_noise",
    "missed_ink",
    "false_ink",
    "false_branches",
    "broken_strokes",
    "overconnected",
    "segmentation_issue",
    "htr_error_explained_by_structure",
    "notes",
]

BOOL_FIELDS = [
    "structural_usable",
    "foreground_ok",
    "skeleton_ok",
    "graph_ok",
]

SEVERITY_FIELDS = [
    "line_residual",
    "neighbor_noise",
    "missed_ink",
    "false_ink",
    "false_branches",
    "broken_strokes",
    "overconnected",
    "segmentation_issue",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def image_uri_from_array(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        image = Image.fromarray(arr, mode="L")
    else:
        image = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def skeleton_degrees(skel: np.ndarray) -> np.ndarray:
    work = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(
        work,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return neighbors.astype(np.int16) - work.astype(np.int16)


def overlay_graph(gray: np.ndarray, fg: np.ndarray, skel: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(np.asarray(gray, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    overlay = base.astype(np.float32)

    fg = np.asarray(fg, dtype=bool)
    skel = np.asarray(skel, dtype=bool)
    degrees = skeleton_degrees(skel)
    endpoints = skel & (degrees == 1)
    branchpoints = skel & (degrees >= 3)
    body = skel & ~(endpoints | branchpoints)

    overlay[fg] = 0.65 * overlay[fg] + 0.35 * np.array([255, 0, 0], dtype=np.float32)
    overlay[body] = np.array([0, 160, 255], dtype=np.float32)
    overlay[endpoints] = np.array([0, 220, 80], dtype=np.float32)
    overlay[branchpoints] = np.array([255, 210, 0], dtype=np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def foreground_panel(fg: np.ndarray, polygon_mask: np.ndarray | None = None) -> np.ndarray:
    panel = np.full(fg.shape, 255, dtype=np.uint8)
    panel[np.asarray(fg, dtype=bool)] = 0
    if polygon_mask is not None:
        rgb = cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB)
        edges = cv2.morphologyEx(
            np.asarray(polygon_mask, dtype=np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        ) > 0
        rgb[edges] = np.array([0, 180, 0], dtype=np.uint8)
        return rgb
    return panel


def heatmap_panel(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or float(arr.max()) <= 0.0:
        norm = np.zeros(arr.shape, dtype=np.uint8)
    else:
        p95 = max(float(np.quantile(arr, 0.95)), 1.0)
        norm = np.clip(arr / p95 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)[:, :, ::-1]


def load_gray_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def render_structural_panels(
    row: dict[str, Any],
    full_row: dict[str, Any] | None,
    *,
    coco: SchoolCocoSource,
) -> dict[str, str]:
    dataset = str(row.get("dataset", ""))
    if dataset == "school_notebooks_clean" and full_row is not None:
        extracted = extract_school_lineaware_v3(full_row, coco)
        gray = extracted["gray_rectangular"]
        normalized = extracted["normalized_gray"]
        fg = extracted["foreground"].astype(bool)
        polygon_mask = extracted["polygon_mask"].astype(bool)
        ruling = extracted["ruling_response"]
        skel = skeletonize(fg)
        return {
            "raw": image_uri_from_array(gray),
            "normalized": image_uri_from_array(normalized),
            "foreground": image_uri_from_array(foreground_panel(fg, polygon_mask)),
            "skeleton_graph": image_uri_from_array(overlay_graph(normalized, fg, skel)),
            "ruling": image_uri_from_array(heatmap_panel(ruling)),
        }

    image_path = Path(str(row["image_path"]))
    gray = load_gray_image(image_path)
    method = "otsu"
    fg = binarize(gray, method=method, sauvola_window=31)
    if fg.any():
        fg = remove_small_objects(fg, min_size=3)
    skel = skeletonize(fg)
    blank_ruling = np.zeros_like(gray, dtype=np.uint8)
    return {
        "raw": image_uri_from_array(gray),
        "normalized": image_uri_from_array(gray),
        "foreground": image_uri_from_array(foreground_panel(fg)),
        "skeleton_graph": image_uri_from_array(overlay_graph(gray, fg, skel)),
        "ruling": image_uri_from_array(blank_ruling),
    }


def select_html(sample_id: str, field: str, options: list[tuple[str, str]], default: str = "") -> str:
    opts = ['<option value=""></option>']
    for value, label in options:
        selected = " selected" if value == default else ""
        opts.append(f'<option value="{value}"{selected}>{label}</option>')
    return (
        f'<label>{field}'
        f'<select data-sample="{sample_id}" data-field="{field}">'
        f'{"".join(opts)}</select></label>'
    )


def textarea_html(sample_id: str) -> str:
    return (
        f'<label class="notes">notes'
        f'<textarea data-sample="{sample_id}" data-field="notes"></textarea>'
        f'</label>'
    )


def favorable_defaults(row: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "structural_usable": "1",
        "foreground_ok": "1",
        "skeleton_ok": "1",
        "graph_ok": "1",
        "line_residual": "0",
        "neighbor_noise": "0",
        "missed_ink": "0",
        "false_ink": "0",
        "false_branches": "0",
        "broken_strokes": "0",
        "overconnected": "0",
        "segmentation_issue": "0",
        "htr_error_explained_by_structure": (
            "not_applicable"
            if float(row.get("exact", 0.0)) >= 1.0
            else "no"
        ),
        "notes": "",
    }
    return defaults


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--full_manifest", required=True)
    parser.add_argument("--school_raw_dir", required=True)
    parser.add_argument("--out_html", required=True)
    args = parser.parse_args()

    sample_rows = read_jsonl(Path(args.sample_manifest))
    full_rows = {
        str(row["sample_id"]): row
        for row in read_jsonl(Path(args.full_manifest))
    }
    coco = SchoolCocoSource(args.school_raw_dir)

    cards = []
    for index, row in enumerate(sample_rows, start=1):
        sample_id = str(row["sample_id"])
        panels = render_structural_panels(
            row,
            full_rows.get(sample_id),
            coco=coco,
        )
        htr_default = "not_applicable" if float(row.get("exact", 0.0)) >= 1.0 else ""

        controls = []
        for field in BOOL_FIELDS:
            controls.append(select_html(sample_id, field, [("1", "1 yes"), ("0", "0 no")]))
        for field in SEVERITY_FIELDS:
            controls.append(select_html(
                sample_id,
                field,
                [("0", "0 none"), ("1", "1 minor"), ("2", "2 severe")],
            ))
        controls.append(select_html(
            sample_id,
            "htr_error_explained_by_structure",
            [
                ("yes", "yes"),
                ("partial", "partial"),
                ("no", "no"),
                ("not_applicable", "not_applicable"),
            ],
            default=htr_default,
        ))
        controls.append(textarea_html(sample_id))

        meta = {
            "sample_id": sample_id,
            "stratum": row.get("gold_stratum", ""),
            "dataset": row.get("dataset", ""),
            "token_type": row.get("token_type", ""),
            "defaults": favorable_defaults(row),
        }

        panel_html = "".join([
            f'<figure><figcaption>{name}</figcaption><img src="{uri}" alt="{name}"></figure>'
            for name, uri in panels.items()
        ])

        cards.append(f"""
<section class="card" data-sample="{sample_id}">
  <div class="idx">#{index}</div>
  <div class="meta">
    <b>{sample_id}</b>
    <span>{row.get("gold_stratum", "")}</span><br>
    dataset={row.get("dataset", "")} |
    token={row.get("token_type", "")} |
    quality={row.get("school_quality_bucket", "")} |
    risk={float(row.get("risk", 0.0)):.4f} |
    CER={float(row.get("cer", 0.0)):.3f} |
    exact={float(row.get("exact", 0.0)):.0f}
  </div>
  <div class="textline">
    target: <b>{row.get("target", "")}</b><br>
    pred: <b>{row.get("pred", "")}</b>
  </div>
  <div class="panels">{panel_html}</div>
  <div class="legend">
    overlay: red=foreground, blue=skeleton, green=endpoints, yellow=branchpoints. Foreground panel green outline=School polygon.
  </div>
  <div class="features">
    fg={row.get("fg_fraction")} |
    skel={row.get("skel_fraction")} |
    cc={row.get("cc_count")} |
    dir_h={row.get("dir_h_frac")} |
    stroke={row.get("stroke_width_mean")} |
    ruling_mean={row.get("ruling_response_mean")} |
    reasons={";".join(row.get("school_quality_reasons") or [])}
  </div>
  <div class="controls" data-meta='{json.dumps(meta, ensure_ascii=False)}'>
    {"".join(controls)}
  </div>
</section>
""")

    fields_json = json.dumps(FIELDS, ensure_ascii=False)

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Structural Gold Visual Annotation Browser</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  background: #f4f4f4;
  color: #222;
}}
.topbar {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: #fff;
  border-bottom: 1px solid #ccc;
  padding: 12px 18px;
}}
.topbar button {{
  margin-right: 8px;
  padding: 7px 10px;
}}
.wrap {{
  margin: 18px 24px;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 16px;
}}
.idx {{
  float: right;
  color: #777;
  font-size: 13px;
}}
.meta {{
  font-size: 13px;
  line-height: 1.45;
}}
.meta span {{
  display: inline-block;
  margin-left: 8px;
  padding: 2px 6px;
  border-radius: 4px;
  background: #eceff3;
}}
.textline {{
  margin: 10px 0;
  font-size: 16px;
}}
.panels {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  align-items: start;
}}
figure {{
  margin: 0;
}}
figcaption {{
  font-size: 12px;
  color: #444;
  margin-bottom: 4px;
}}
img {{
  max-width: 100%;
  background: white;
  border: 1px solid #ccc;
}}
.legend, .features {{
  margin-top: 8px;
  font-size: 12px;
  color: #555;
}}
.controls {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px;
  margin-top: 12px;
}}
label {{
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
}}
select, textarea {{
  font: inherit;
  padding: 6px;
}}
.notes {{
  grid-column: 1 / -1;
}}
textarea {{
  min-height: 54px;
}}
#csvOutput {{
  width: 100%;
  min-height: 160px;
  margin-top: 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
.status {{
  color: #555;
  font-size: 13px;
  margin-left: 8px;
}}
</style>
</head>
<body>
<div class="topbar">
  <button id="fillFavorableBtn">Fill favorable defaults</button>
  <button id="exportBtn">Export CSV</button>
  <button id="downloadBtn">Download CSV</button>
  <button id="clearBtn">Clear saved annotations</button>
  <span class="status" id="status">Autosaves in this browser.</span>
  <textarea id="csvOutput" placeholder="CSV export appears here"></textarea>
</div>
<div class="wrap">
<h1>Structural Gold Visual Annotation Browser</h1>
<p>Use the structural panels to annotate foreground, skeleton, graph, and ruling/noise issues.</p>
{''.join(cards)}
</div>
<script>
const FIELDS = {fields_json};
const STORAGE_KEY = "structural_gold_v1_visual_annotations";

function loadState() {{
  try {{
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
  }} catch (err) {{
    return {{}};
  }}
}}

function saveState(state) {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}}

function csvEscape(value) {{
  value = value == null ? "" : String(value);
  if (/[",\\n\\r]/.test(value)) {{
    return '"' + value.replace(/"/g, '""') + '"';
  }}
  return value;
}}

function collectRows() {{
  const state = loadState();
  const rows = [];
  document.querySelectorAll(".card").forEach(card => {{
    const controls = card.querySelector(".controls");
    const meta = JSON.parse(controls.dataset.meta);
    const sample = meta.sample_id;
    const row = {{
      sample_id: sample,
      stratum: meta.stratum,
      dataset: meta.dataset,
      token_type: meta.token_type
    }};
    FIELDS.forEach(field => {{
      if (!(field in row)) row[field] = "";
    }});
    const saved = state[sample] || {{}};
    Object.keys(saved).forEach(field => {{
      row[field] = saved[field];
    }});
    rows.push(row);
  }});
  return rows;
}}

function fillFavorableDefaults() {{
  const state = loadState();
  document.querySelectorAll(".card").forEach(card => {{
    const controls = card.querySelector(".controls");
    const meta = JSON.parse(controls.dataset.meta);
    const sample = meta.sample_id;
    if (!state[sample]) state[sample] = {{}};
    Object.keys(meta.defaults || {{}}).forEach(field => {{
      if (!state[sample][field]) {{
        state[sample][field] = meta.defaults[field];
      }}
    }});
  }});
  saveState(state);
  updateControlFromState();
  document.getElementById("status").textContent = "Filled empty fields with favorable defaults";
}}

function toCsv(rows) {{
  const lines = [FIELDS.map(csvEscape).join(",")];
  rows.forEach(row => {{
    lines.push(FIELDS.map(field => csvEscape(row[field] || "")).join(","));
  }});
  return lines.join("\\n") + "\\n";
}}

function updateControlFromState() {{
  const state = loadState();
  document.querySelectorAll("[data-sample][data-field]").forEach(el => {{
    const sample = el.dataset.sample;
    const field = el.dataset.field;
    if (state[sample] && state[sample][field] != null) {{
      el.value = state[sample][field];
    }}
  }});
}}

function persistInput(el) {{
  const state = loadState();
  const sample = el.dataset.sample;
  const field = el.dataset.field;
  if (!state[sample]) state[sample] = {{}};
  state[sample][field] = el.value;
  saveState(state);
  document.getElementById("status").textContent = "Saved " + sample + " / " + field;
}}

document.querySelectorAll("[data-sample][data-field]").forEach(el => {{
  el.addEventListener("change", () => persistInput(el));
  el.addEventListener("input", () => persistInput(el));
}});

document.getElementById("exportBtn").addEventListener("click", () => {{
  document.getElementById("csvOutput").value = toCsv(collectRows());
}});

document.getElementById("fillFavorableBtn").addEventListener("click", () => {{
  fillFavorableDefaults();
}});

document.getElementById("downloadBtn").addEventListener("click", () => {{
  const csv = toCsv(collectRows());
  const blob = new Blob([csv], {{type: "text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "annotations_structural_filled.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}});

document.getElementById("clearBtn").addEventListener("click", () => {{
  if (confirm("Clear saved annotations in this browser?")) {{
    localStorage.removeItem(STORAGE_KEY);
    location.reload();
  }}
}});

updateControlFromState();
</script>
</body>
</html>
"""

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    print(json.dumps({
        "out_html": str(out_html),
        "n": len(sample_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
