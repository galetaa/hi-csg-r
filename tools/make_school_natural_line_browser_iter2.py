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
            color = (
                (46, 125, 50)
                if row["quality_buckets"][index] == "clean_core"
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
                <dt>joined</dt><dd>{html.escape(str(row["joined_text"]))}</dd>
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--school_raw_dir", default="data/interim/school_notebooks")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--padding", type=int, default=30)
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
