from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def score_candidate(row: dict[str, Any]) -> tuple[int, int, int, str]:
    flags = set(row.get("flags") or [])
    n_words = int(row.get("n_words", 0))

    priority = 0

    if "has_invalid_or_review" in flags:
        priority += 100

    if "has_hard_real" in flags:
        priority += 50

    if "short_group" not in flags:
        priority += 20

    if "contains_single_char_or_mark" in flags:
        priority += 10

    return (
        -priority,
        -n_words,
        int(row.get("bbox_xyxy", [0, 0, 0, 0])[1]),
        str(row.get("line_group_id", "")),
    )


def selected_candidates(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows

    return sorted(rows, key=score_candidate)[:limit]


def crop_line_image(
    row: dict[str, Any],
    *,
    images_root: Path,
    assets_dir: Path,
    padding: int,
) -> str | None:
    image_path = images_root / str(row["source_image_file"])

    if not image_path.exists():
        return None

    x0, y0, x1, y1 = [
        int(v)
        for v in row["bbox_xyxy"]
    ]

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size

        crop_x0 = max(0, x0 - padding)
        crop_y0 = max(0, y0 - padding)
        crop_x1 = min(width, x1 + padding)
        crop_y1 = min(height, y1 + padding)

        crop = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        draw = ImageDraw.Draw(crop)

        for index, bbox in enumerate(row["word_bboxes_xyxy"]):
            bx0, by0, bx1, by1 = [
                int(v)
                for v in bbox
            ]
            local = [
                bx0 - crop_x0,
                by0 - crop_y0,
                bx1 - crop_x0,
                by1 - crop_y0,
            ]
            quality_buckets = row.get("quality_buckets") or []
            quality_bucket = (
                quality_buckets[index]
                if index < len(quality_buckets)
                else "full_coco"
            )
            color = (
                (46, 125, 50)
                if quality_bucket == "clean_core"
                else (25, 118, 210)
                if quality_bucket == "full_coco"
                else (239, 108, 0)
            )
            draw.rectangle(local, outline=color, width=3)
            draw.text((local[0] + 2, max(0, local[1] - 14)), str(index + 1), fill=color)

        draw.rectangle(
            [
                x0 - crop_x0,
                y0 - crop_y0,
                x1 - crop_x0,
                y1 - crop_y0,
            ],
            outline=(25, 118, 210),
            width=2,
        )

    asset_name = f"{row['line_group_id']}.jpg".replace("/", "_")
    asset_path = assets_dir / asset_name
    crop.save(asset_path, quality=92)

    return f"assets/{asset_name}"


def render_html(
    rows: list[dict[str, Any]],
    *,
    image_refs: dict[str, str | None],
    summary: dict[str, Any] | None,
) -> str:
    cards = []

    for row in rows:
        image_ref = image_refs.get(str(row["line_group_id"]))
        flags = ", ".join(row.get("flags") or [])
        words = " | ".join(row.get("texts") or [])
        buckets = " | ".join(row.get("quality_buckets") or [])
        joined_text = str(
            row.get("joined_text")
            or row.get("joined_text_space")
            or ""
        )

        image_html = (
            f'<img src="{html.escape(image_ref)}" alt="">'
            if image_ref
            else '<div class="missing">missing source image</div>'
        )

        cards.append(
            f"""
            <article class="card">
              <header>
                <div>
                  <h2>{html.escape(str(row["line_group_id"]))}</h2>
                  <p>{html.escape(str(row["split"]))} · {html.escape(str(row["source_image_file"]))} · line {html.escape(str(row["line_id"]))}</p>
                </div>
                <strong>{int(row["n_words"])} words</strong>
              </header>
              {image_html}
              <dl>
                <dt>joined</dt><dd>{html.escape(joined_text)}</dd>
                <dt>words</dt><dd>{html.escape(words)}</dd>
                <dt>quality</dt><dd>{html.escape(buckets)}</dd>
                <dt>flags</dt><dd>{html.escape(flags)}</dd>
                <dt>samples</dt><dd>{html.escape(", ".join(row["sample_ids"]))}</dd>
              </dl>
            </article>
            """
        )

    summary_html = ""

    if summary:
        summary_html = (
            "<pre>"
            + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
            + "</pre>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>School natural line candidates v1</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #16191f;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 24px;
    }}
    pre {{
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      padding: 14px;
      border-radius: 6px;
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .card {{
      background: white;
      border: 1px solid #d7dce3;
      border-radius: 8px;
      padding: 14px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }}
    h2 {{
      margin: 0;
      font-size: 15px;
      line-height: 1.25;
    }}
    header p {{
      margin: 3px 0 0;
      color: #667085;
      font-size: 12px;
    }}
    strong {{
      white-space: nowrap;
      font-size: 13px;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 220px;
      object-fit: contain;
      background: #eceff3;
      border: 1px solid #e2e6eb;
    }}
    .missing {{
      padding: 32px;
      background: #fee2e2;
      color: #991b1b;
      border-radius: 4px;
    }}
    dl {{
      display: grid;
      grid-template-columns: 78px 1fr;
      gap: 5px 10px;
      margin: 12px 0 0;
      font-size: 13px;
    }}
    dt {{
      color: #667085;
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <main>
    <h1>School natural line candidates v1</h1>
    {summary_html}
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""


ANNOTATION_FIELDS = [
    "valid_line",
    "correct_order",
    "missing_words",
    "neighbor_noise",
    "good_for_train_aug",
]


def render_annotation_html(
    rows: list[dict[str, Any]],
    *,
    image_refs: dict[str, str | None],
    summary: dict[str, Any] | None,
    annotation_fields: list[str] | None = None,
    download_name: str = "natural_line_validation_annotations.csv",
) -> str:
    annotation_fields = annotation_fields or ANNOTATION_FIELDS
    cards = []

    for index, row in enumerate(rows, start=1):
        line_group_id = str(row["line_group_id"])
        image_ref = image_refs.get(line_group_id)
        flags = ", ".join(row.get("flags") or [])
        words = " | ".join(row.get("texts") or [])
        buckets = " | ".join(row.get("quality_buckets") or [])
        stratum = str(row.get("validation_stratum", ""))
        joined_text = str(
            row.get("joined_text")
            or row.get("joined_text_space")
            or ""
        )

        image_html = (
            f'<img src="{html.escape(image_ref)}" alt="">'
            if image_ref
            else '<div class="missing">missing source image</div>'
        )

        controls = []

        for field in annotation_fields:
            controls.append(
                f"""
                <label class="check">
                  <input type="checkbox" data-field="{field}">
                  <span>{field}</span>
                </label>
                """
            )

        cards.append(
            f"""
            <article class="card" data-line-group-id="{html.escape(line_group_id)}">
              <header>
                <div>
                  <h2>{index}. {html.escape(line_group_id)}</h2>
                  <p>{html.escape(str(row["split"]))} · {html.escape(str(row["source_image_file"]))} · line {html.escape(str(row["line_id"]))} · {html.escape(stratum)}</p>
                </div>
                <strong>{int(row["n_words"])} words</strong>
              </header>
              {image_html}
              <dl>
                <dt>joined</dt><dd>{html.escape(joined_text)}</dd>
                <dt>words</dt><dd>{html.escape(words)}</dd>
                <dt>quality</dt><dd>{html.escape(buckets)}</dd>
                <dt>flags</dt><dd>{html.escape(flags)}</dd>
                <dt>samples</dt><dd>{html.escape(", ".join(row["sample_ids"]))}</dd>
              </dl>
              <div class="controls">
                {''.join(controls)}
              </div>
              <textarea data-field="notes" rows="2" placeholder="notes"></textarea>
            </article>
            """
        )

    summary_html = ""

    if summary:
        summary_html = (
            "<pre>"
            + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
            + "</pre>"
        )

    rows_json = json.dumps(
        [
            {
                "line_group_id": row["line_group_id"],
            }
            for row in rows
        ],
        ensure_ascii=False,
    )

    fields_json = json.dumps(annotation_fields, ensure_ascii=False)
    download_name_json = json.dumps(download_name, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>School natural line validation</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #16191f;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin: -24px -24px 18px;
      padding: 14px 24px;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid #d7dce3;
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
    }}
    .actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    button {{
      border: 1px solid #b8c0cc;
      background: white;
      color: #172033;
      border-radius: 6px;
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
      font-size: 13px;
      color: #4b5563;
    }}
    pre {{
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      padding: 14px;
      border-radius: 6px;
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .card {{
      background: white;
      border: 1px solid #d7dce3;
      border-radius: 8px;
      padding: 14px;
    }}
    .card.done {{
      border-color: #2e7d32;
      box-shadow: 0 0 0 1px #2e7d32 inset;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }}
    h2 {{
      margin: 0;
      font-size: 15px;
      line-height: 1.25;
    }}
    header p {{
      margin: 3px 0 0;
      color: #667085;
      font-size: 12px;
    }}
    strong {{
      white-space: nowrap;
      font-size: 13px;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 220px;
      object-fit: contain;
      background: #eceff3;
      border: 1px solid #e2e6eb;
    }}
    .missing {{
      padding: 32px;
      background: #fee2e2;
      color: #991b1b;
      border-radius: 4px;
    }}
    dl {{
      display: grid;
      grid-template-columns: 78px 1fr;
      gap: 5px 10px;
      margin: 12px 0 0;
      font-size: 13px;
    }}
    dt {{
      color: #667085;
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .check {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 6px 8px;
      background: #f8fafc;
      border: 1px solid #d7dce3;
      border-radius: 6px;
      font-size: 13px;
    }}
    input[type="checkbox"] {{
      width: 18px;
      height: 18px;
      margin: 0;
    }}
    textarea {{
      box-sizing: border-box;
      width: 100%;
      margin-top: 10px;
      padding: 8px;
      border: 1px solid #c8d0dc;
      border-radius: 6px;
      font: inherit;
      font-size: 13px;
      resize: vertical;
    }}
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <h1>School natural line validation</h1>
      <div class="actions">
        <span id="progress"></span>
        <button type="button" id="mark-defaults">Mark Defaults</button>
        <button type="button" id="clear-all">Clear Saved</button>
        <button type="button" class="primary" id="download-csv">Download CSV</button>
      </div>
    </div>
    {summary_html}
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
  <script>
    const rows = {rows_json};
    const fields = {fields_json};
    const storageKey = "school_natural_line_validation_v1";

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
      for (const row of rows) {{
        if (isDone(state[row.line_group_id] || {{}})) {{
          done += 1;
        }}
      }}
      document.getElementById("progress").textContent = `${{done}} / ${{rows.length}} annotated`;
      for (const card of document.querySelectorAll(".card")) {{
        const id = card.dataset.lineGroupId;
        card.classList.toggle("done", isDone(state[id] || {{}}));
      }}
    }}

    function restore() {{
      const state = loadState();
      for (const card of document.querySelectorAll(".card")) {{
        const id = card.dataset.lineGroupId;
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
        const id = card.dataset.lineGroupId;
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

    function downloadCsv() {{
      const state = loadState();
      const header = ["line_group_id", ...fields, "notes"];
      const lines = [header.join(",")];
      for (const row of rows) {{
        const item = state[row.line_group_id] || {{}};
        lines.push([
          row.line_group_id,
          ...fields.map((field) => item[field] || "0"),
          item.notes || "",
        ].map(csvEscape).join(","));
      }}
      const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = {download_name_json};
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }}

    function markDefaults() {{
      const state = loadState();
      for (const row of rows) {{
        const existing = state[row.line_group_id] || {{}};
        const item = {{ notes: existing.notes || "" }};
        for (const field of fields) {{
          if (field === "neighbor_noise" || field === "missing_words") {{
            item[field] = existing[field] || "0";
          }} else {{
            item[field] = existing[field] || "1";
          }}
        }}
        state[row.line_group_id] = item;
      }}
      saveState(state);
      restore();
    }}

    document.getElementById("download-csv").addEventListener("click", downloadCsv);
    document.getElementById("mark-defaults").addEventListener("click", markDefaults);
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--school_raw_dir", default="data/interim/school_notebooks")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--padding", type=int, default=30)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument(
        "--annotation_fields",
        default=",".join(ANNOTATION_FIELDS),
        help="Comma-separated annotation fields before notes.",
    )
    parser.add_argument(
        "--download_name",
        default="natural_line_validation_annotations.csv",
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.candidates))
    rows = selected_candidates(rows, args.limit)

    out_dir = Path(args.out_dir)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    images_root = Path(args.school_raw_dir) / "images" / "images"

    image_refs = {
        str(row["line_group_id"]): crop_line_image(
            row,
            images_root=images_root,
            assets_dir=assets_dir,
            padding=args.padding,
        )
        for row in rows
    }

    summary_path = Path(args.summary_json)
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else None
    )

    if args.annotate:
        annotation_fields = [
            field.strip()
            for field in args.annotation_fields.split(",")
            if field.strip()
        ]
        html_text = render_annotation_html(
            rows,
            image_refs=image_refs,
            summary=summary,
            annotation_fields=annotation_fields,
            download_name=args.download_name,
        )
        out_path = out_dir / "natural_line_annotation_browser.html"
    else:
        html_text = render_html(
            rows,
            image_refs=image_refs,
            summary=summary,
        )
        out_path = out_dir / "natural_line_candidates_browser.html"
    out_path.write_text(html_text, encoding="utf-8")

    print("wrote:", out_path)
    print("rendered groups:", len(rows))


if __name__ == "__main__":
    main()
