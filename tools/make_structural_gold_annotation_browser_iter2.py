from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any


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


def image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{encoded}"


def option(value: str, label: str, selected: bool = False) -> str:
    attr = " selected" if selected else ""
    return f'<option value="{html.escape(value)}"{attr}>{html.escape(label)}</option>'


def select_html(sample_id: str, field: str, options: list[tuple[str, str]], default: str = "") -> str:
    opts = [option("", "")]
    for value, label in options:
        opts.append(option(value, label, value == default))
    return (
        f'<label>{html.escape(field)}'
        f'<select data-sample="{html.escape(sample_id)}" data-field="{html.escape(field)}">'
        f'{"".join(opts)}</select></label>'
    )


def textarea_html(sample_id: str, field: str) -> str:
    return (
        f'<label class="notes">{html.escape(field)}'
        f'<textarea data-sample="{html.escape(sample_id)}" data-field="{html.escape(field)}"></textarea>'
        f'</label>'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--out_html", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.sample_manifest))
    cards = []

    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        image_path = Path(str(row["image_path"]))
        uri = image_data_uri(image_path) if image_path.exists() else ""
        htr_default = "not_applicable" if float(row.get("exact", 0.0)) >= 1.0 else ""

        controls = []
        for field in BOOL_FIELDS:
            controls.append(select_html(
                sample_id,
                field,
                [("1", "1 yes"), ("0", "0 no")],
            ))
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
        controls.append(textarea_html(sample_id, "notes"))

        meta = {
            "sample_id": sample_id,
            "stratum": row.get("gold_stratum", ""),
            "dataset": row.get("dataset", ""),
            "token_type": row.get("token_type", ""),
        }

        cards.append(f"""
<section class="card" id="card-{html.escape(sample_id)}" data-sample="{html.escape(sample_id)}">
  <div class="idx">#{index}</div>
  <div class="meta">
    <b>{html.escape(sample_id)}</b>
    <span>{html.escape(str(row.get("gold_stratum", "")))}</span><br>
    dataset={html.escape(str(row.get("dataset", "")))} |
    token={html.escape(str(row.get("token_type", "")))} |
    quality={html.escape(str(row.get("school_quality_bucket", "")))} |
    risk={float(row.get("risk", 0.0)):.4f} |
    CER={float(row.get("cer", 0.0)):.3f} |
    exact={float(row.get("exact", 0.0)):.0f}
  </div>
  <div class="textline">
    target: <b>{html.escape(str(row.get("target", "")))}</b><br>
    pred: <b>{html.escape(str(row.get("pred", "")))}</b>
  </div>
  <img src="{uri}" alt="{html.escape(sample_id)}">
  <div class="features">
    fg={row.get("fg_fraction")} |
    skel={row.get("skel_fraction")} |
    cc={row.get("cc_count")} |
    dir_h={row.get("dir_h_frac")} |
    stroke={row.get("stroke_width_mean")} |
    ruling_mean={row.get("ruling_response_mean")} |
    reasons={html.escape(";".join(row.get("school_quality_reasons") or []))}
  </div>
  <div class="controls" data-meta='{html.escape(json.dumps(meta, ensure_ascii=False))}'>
    {"".join(controls)}
  </div>
</section>
""")

    fields_json = json.dumps(FIELDS, ensure_ascii=False)

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Structural Gold Annotation Browser</title>
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
img {{
  max-width: 100%;
  background: white;
  border: 1px solid #ccc;
}}
.features {{
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
  min-height: 180px;
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
  <button id="exportBtn">Export CSV</button>
  <button id="downloadBtn">Download CSV</button>
  <button id="clearBtn">Clear saved annotations</button>
  <span class="status" id="status">Autosaves in this browser.</span>
  <textarea id="csvOutput" placeholder="CSV export appears here"></textarea>
</div>
<div class="wrap">
<h1>Structural Gold Annotation Browser</h1>
<p>Fill controls directly in the cards. Use Export CSV, then download or copy from the textarea.</p>
{''.join(cards)}
</div>
<script>
const FIELDS = {fields_json};
const STORAGE_KEY = "structural_gold_v1_annotations";

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
        "n": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
