from __future__ import annotations

import argparse
import base64
import html
import json
import random
from pathlib import Path
from typing import Any


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
    return f"data:image/png;base64,{encoded}"


def stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    buckets: dict[str, list[dict[str, Any]]] = {
        "2_words": [],
        "3_words": [],
        "4_7_words": [],
        "8plus_words": [],
        "hard_line": [],
    }

    for row in rows:
        n_words = int(row.get("n_words", 0))
        corpus_bucket = str(row.get("line_corpus_bucket", ""))

        if corpus_bucket == "hard_line":
            buckets["hard_line"].append(row)

        if n_words == 2:
            buckets["2_words"].append(row)
        elif n_words == 3:
            buckets["3_words"].append(row)
        elif 4 <= n_words <= 7:
            buckets["4_7_words"].append(row)
        elif n_words >= 8:
            buckets["8plus_words"].append(row)

    plan = {
        "2_words": 20,
        "3_words": 20,
        "4_7_words": 25,
        "8plus_words": 15,
        "hard_line": 20,
    }

    selected: list[dict[str, Any]] = []
    seen = set()

    for bucket, target in plan.items():
        candidates = [
            row for row in buckets[bucket]
            if row["sample_id"] not in seen
        ]

        rng.shuffle(candidates)

        for row in candidates[:target]:
            selected.append(row)
            seen.add(row["sample_id"])

    if len(selected) < n:
        rest = [
            row for row in rows
            if row["sample_id"] not in seen
        ]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])

    return selected[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out_html", required=True)
    parser.add_argument("--out_csv_template", required=True)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    selected = stratified_sample(rows, args.n, args.seed)

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv_template)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_lines = [
        "sample_id,readable,correct_crop,good_for_htr,notes"
    ]

    cards = []

    for row in selected:
        image_path = Path(row["image_path"])
        uri = image_data_uri(image_path)

        sample_id = html.escape(str(row["sample_id"]))
        text = html.escape(str(row.get("text", "")))
        bucket = html.escape(str(row.get("line_corpus_bucket", "")))
        n_words = html.escape(str(row.get("n_words", "")))
        xgap = html.escape(str(row.get("x_gap_max_norm_by_height", "")))

        cards.append(f"""
        <div class="card" data-sample-id="{sample_id}">
          <div class="meta">
            <b>{sample_id}</b><br>
            bucket: {bucket} |
            n_words: {n_words} |
            x_gap_h: {xgap}
          </div>
          <div class="text">{text}</div>
          <img src="{uri}">
          <div class="controls">
            <label><input type="checkbox" data-field="readable"> readable</label>
            <label><input type="checkbox" data-field="correct_crop"> correct_crop</label>
            <label><input type="checkbox" data-field="good_for_htr"> good_for_htr</label>
          </div>
          <textarea data-field="notes" rows="2" placeholder="notes"></textarea>
        </div>
        """)

        csv_lines.append(f"{sample_id},,,,")

    html_doc = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Rendered line sanity browser</title>
<style>
body {{
  font-family: sans-serif;
  margin: 0;
  background: #f7f7f7;
}}
main {{
  margin: 24px;
}}
.topbar {{
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 14px 24px;
  background: rgba(247, 247, 247, 0.96);
  border-bottom: 1px solid #ddd;
}}
h1 {{
  margin: 0;
  font-size: 22px;
}}
.actions {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
button {{
  border: 1px solid #bbb;
  border-radius: 6px;
  background: white;
  padding: 8px 11px;
  font-weight: 600;
  cursor: pointer;
}}
button.primary {{
  background: #1f6feb;
  border-color: #1f6feb;
  color: white;
}}
#progress {{
  color: #555;
  font-size: 13px;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 18px;
}}
.card.done {{
  border-color: #2e7d32;
  box-shadow: 0 0 0 1px #2e7d32 inset;
}}
.meta {{
  font-size: 13px;
  color: #333;
  margin-bottom: 8px;
}}
.text {{
  font-size: 16px;
  margin-bottom: 10px;
}}
img {{
  max-width: 100%;
  border: 1px solid #ccc;
  background: white;
}}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
}}
.controls label {{
  display: flex;
  align-items: center;
  gap: 7px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #f8fafc;
  padding: 7px 9px;
  font-size: 13px;
}}
input[type="checkbox"] {{
  width: 18px;
  height: 18px;
}}
textarea {{
  box-sizing: border-box;
  width: 100%;
  margin-top: 10px;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 6px;
  font: inherit;
  font-size: 13px;
}}
</style>
</head>
<body>
<div class="topbar">
  <h1>Rendered School line sanity browser</h1>
  <div class="actions">
    <span id="progress"></span>
    <button type="button" id="mark-defaults">Mark Defaults</button>
    <button type="button" id="clear-all">Clear Saved</button>
    <button type="button" class="primary" id="download-csv">Download CSV</button>
  </div>
</div>
<main>
  <p>n={len(selected)}</p>
  {''.join(cards)}
</main>
<script>
const fields = ["readable", "correct_crop", "good_for_htr"];
const sampleIds = {json.dumps([str(row["sample_id"]) for row in selected], ensure_ascii=False)};
const storageKey = "rendered_line_sanity_v1";

function loadState() {{
  try {{
    return JSON.parse(localStorage.getItem(storageKey) || "{{}}");
  }} catch (error) {{
    return {{}};
  }}
}}

function saveState(state) {{
  localStorage.setItem(storageKey, JSON.stringify(state));
}}

function csvEscape(value) {{
  const text = String(value ?? "");
  if (/[",\\n\\r]/.test(text)) {{
    return '"' + text.replaceAll('"', '""') + '"';
  }}
  return text;
}}

function cardState(card) {{
  const item = {{}};
  for (const field of fields) {{
    item[field] = card.querySelector(`[data-field="${{field}}"]`).checked ? "1" : "0";
  }}
  item.notes = card.querySelector('[data-field="notes"]').value || "";
  return item;
}}

function isDone(item) {{
  return fields.some((field) => item[field] === "1") || Boolean(item.notes);
}}

function updateProgress() {{
  const state = loadState();
  let done = 0;
  for (const sampleId of sampleIds) {{
    if (isDone(state[sampleId] || {{}})) {{
      done += 1;
    }}
  }}
  document.getElementById("progress").textContent = `${{done}} / ${{sampleIds.length}} annotated`;
  for (const card of document.querySelectorAll(".card")) {{
    const id = card.dataset.sampleId;
    card.classList.toggle("done", isDone(state[id] || {{}}));
  }}
}}

function restore() {{
  const state = loadState();
  for (const card of document.querySelectorAll(".card")) {{
    const id = card.dataset.sampleId;
    const item = state[id] || {{}};
    for (const field of fields) {{
      card.querySelector(`[data-field="${{field}}"]`).checked = item[field] === "1";
    }}
    card.querySelector('[data-field="notes"]').value = item.notes || "";
  }}
  updateProgress();
}}

function attachAutosave() {{
  for (const card of document.querySelectorAll(".card")) {{
    const id = card.dataset.sampleId;
    const save = () => {{
      const state = loadState();
      state[id] = cardState(card);
      saveState(state);
      updateProgress();
    }};
    for (const input of card.querySelectorAll("input, textarea")) {{
      input.addEventListener("change", save);
      input.addEventListener("input", save);
    }}
  }}
}}

function markDefaults() {{
  const state = loadState();
  for (const sampleId of sampleIds) {{
    const existing = state[sampleId] || {{}};
    state[sampleId] = {{
      readable: existing.readable || "1",
      correct_crop: existing.correct_crop || "1",
      good_for_htr: existing.good_for_htr || "1",
      notes: existing.notes || "",
    }};
  }}
  saveState(state);
  restore();
}}

function downloadCsv() {{
  const state = loadState();
  const header = ["sample_id", "readable", "correct_crop", "good_for_htr", "notes"];
  const lines = [header.join(",")];
  for (const sampleId of sampleIds) {{
    const item = state[sampleId] || {{}};
    lines.push([
      sampleId,
      item.readable || "0",
      item.correct_crop || "0",
      item.good_for_htr || "0",
      item.notes || "",
    ].map(csvEscape).join(","));
  }}
  const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/csv;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "annotations_filled.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}}

document.getElementById("mark-defaults").addEventListener("click", markDefaults);
document.getElementById("download-csv").addEventListener("click", downloadCsv);
document.getElementById("clear-all").addEventListener("click", () => {{
  if (confirm("Clear saved annotations in this browser?")) {{
    localStorage.removeItem(storageKey);
    restore();
  }}
}});

attachAutosave();
restore();
</script>
</body>
</html>
"""

    out_html.write_text(html_doc, encoding="utf-8")
    out_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print("wrote:", out_html)
    print("wrote:", out_csv)


if __name__ == "__main__":
    main()
