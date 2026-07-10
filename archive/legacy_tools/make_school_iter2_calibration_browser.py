from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.morphology import remove_small_objects, skeletonize


SPLIT_QUOTA = {
    "train": 24,
    "val": 8,
    "test": 8,
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def as_int(row: dict[str, Any], key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def stratum(row: dict[str, Any]) -> str | None:
    line_fraction = as_float(
        row,
        "horizontal_line_fraction",
    )
    fg_fraction = as_float(row, "fg_fraction")
    text_len = as_int(row, "text_len")

    if fg_fraction >= 0.35:
        return "foreground_heavy"

    if text_len == 2:
        if line_fraction >= 0.12:
            return "short_text_line_candidate"
        return "short_text_no_line"

    if line_fraction == 0:
        return "line_none"

    if line_fraction < 0.12:
        return "line_low"

    if line_fraction < 0.30:
        return "line_medium"

    return "line_high"


def stratified_sample(
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        group = stratum(row)

        if group is None:
            continue

        split = str(row.get("split", ""))
        groups[(group, split)].append(row)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    all_strata = sorted({
        key[0] for key in groups
    })

    for group in all_strata:
        for split, quota in SPLIT_QUOTA.items():
            candidates = list(
                groups.get((group, split), [])
            )
            rng.shuffle(candidates)

            taken = 0

            for row in candidates:
                sample_id = str(row["sample_id"])

                if sample_id in seen:
                    continue

                copied = dict(row)
                copied["calibration_stratum"] = group

                selected.append(copied)
                seen.add(sample_id)
                taken += 1

                if taken >= quota:
                    break

    rng.shuffle(selected)
    return selected


def school_dark_auto(
    arr: np.ndarray,
) -> tuple[np.ndarray, int]:
    fg145 = remove_small_objects(
        (arr < 145).astype(bool),
        min_size=4,
    )

    if float(fg145.mean()) <= 0.35:
        return fg145.astype(bool), 145

    fg120 = remove_small_objects(
        (arr < 120).astype(bool),
        min_size=4,
    )

    return fg120.astype(bool), 120


def detect_horizontal_lines(
    fg: np.ndarray,
) -> np.ndarray:
    h, w = fg.shape

    kernel_width = max(
        9,
        min(w, int(round(w * 0.35))),
    )

    binary = fg.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1),
    )

    detected = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    ) > 0

    return detected


def naive_remove_lines(
    fg: np.ndarray,
    line_mask: np.ndarray,
) -> np.ndarray:
    cleaned = np.logical_and(
        fg,
        np.logical_not(line_mask),
    )

    if cleaned.any():
        cleaned = remove_small_objects(
            cleaned,
            min_size=4,
        )

    return cleaned.astype(bool)


def fit(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    image = image.copy()
    image.thumbnail(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    canvas.paste(
        image,
        (
            (width - image.width) // 2,
            (height - image.height) // 2,
        ),
    )

    return canvas


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def mask_image(mask: np.ndarray) -> Image.Image:
    arr = (
        255
        - mask.astype(np.uint8) * 255
    )

    return Image.fromarray(
        arr,
        mode="L",
    ).convert("RGB")


def overlay(
    gray: np.ndarray,
    fg: np.ndarray,
    line_mask: np.ndarray,
) -> Image.Image:
    rgb = np.repeat(
        gray[:, :, None],
        3,
        axis=2,
    ).copy()

    skeleton = skeletonize(fg)

    rgb[skeleton] = np.asarray(
        [0, 180, 0],
        dtype=np.uint8,
    )

    rgb[line_mask] = np.asarray(
        [255, 0, 0],
        dtype=np.uint8,
    )

    return Image.fromarray(rgb)


def make_panel(
    title: str,
    images: list[tuple[str, Image.Image]],
    width: int = 900,
    item_height: int = 150,
) -> Image.Image:
    header = 42
    height = header + item_height * len(images)

    panel = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    draw = ImageDraw.Draw(panel)
    draw.text(
        (8, 8),
        title,
        fill="black",
    )

    y = header

    for label, image in images:
        draw.text(
            (8, y + 4),
            label,
            fill="black",
        )

        fitted = fit(
            image,
            width - 140,
            item_height,
        )

        panel.paste(
            fitted,
            (135, y),
        )

        y += item_height

    draw.rectangle(
        (0, 0, width - 1, height - 1),
        outline="black",
    )

    return panel


def prepare_item(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["image_path"])

    image = Image.open(path).convert("L")
    gray = np.asarray(
        image,
        dtype=np.uint8,
    )

    fg, threshold = school_dark_auto(gray)
    line_mask = detect_horizontal_lines(fg)
    cleaned = naive_remove_lines(
        fg,
        line_mask,
    )

    original = ImageOps.autocontrast(
        image,
        cutoff=1,
    ).convert("RGB")

    panel = make_panel(
        title=(
            f"{row['sample_id']} | "
            f"{row['calibration_stratum']}"
        ),
        images=[
            ("original", original),
            (
                "foreground",
                mask_image(fg),
            ),
            (
                "green=skeleton red=line detector",
                overlay(
                    gray,
                    fg,
                    line_mask,
                ),
            ),
            (
                "naive line removal",
                mask_image(cleaned),
            ),
            (
                "skeleton after naive removal",
                overlay(
                    gray,
                    cleaned,
                    np.zeros_like(
                        cleaned,
                        dtype=bool,
                    ),
                ),
            ),
        ],
    )

    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "stratum": row["calibration_stratum"],
        "text": row["text"],
        "image_path": row["image_path"],
        "width": row["width"],
        "text_len": row["text_len"],
        "fg_fraction": row["fg_fraction"],
        "horizontal_line_fraction": (
            row["horizontal_line_fraction"]
        ),
        "horizontal_line_width_fraction": (
            row["horizontal_line_width_fraction"]
        ),
        "threshold_used": threshold,
        "panel": data_url(panel),
    }


HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>School Iteration 2 calibration</title>
<style>
body {
  margin: 0;
  font-family: system-ui, sans-serif;
  background: #f4f4f4;
}
header {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 10px 14px;
  background: #111;
  color: white;
}
button {
  padding: 7px 10px;
  cursor: pointer;
}
main {
  display: grid;
  grid-template-columns: minmax(700px, 1fr) 390px;
  gap: 12px;
  padding: 12px;
}
.card {
  background: white;
  border: 1px solid #ddd;
  border-radius: 9px;
  padding: 12px;
}
#panel {
  width: 100%;
  border: 1px solid #bbb;
}
.meta {
  line-height: 1.5;
  margin-bottom: 10px;
}
.group {
  margin: 13px 0;
}
.group h3 {
  margin: 0 0 6px;
  font-size: 15px;
}
.choice {
  margin: 3px;
  border: 1px solid #aaa;
  background: white;
  border-radius: 5px;
}
.choice.selected {
  outline: 3px solid #111;
}
textarea {
  width: 100%;
  min-height: 70px;
}
#list {
  max-height: 260px;
  overflow: auto;
  border: 1px solid #ddd;
}
#list button {
  display: block;
  width: 100%;
  text-align: left;
  border: 0;
  border-bottom: 1px solid #eee;
  background: white;
}
#list button.done {
  background: #dff2df;
}
#list button.current {
  background: #dcecff;
}
.help {
  font-size: 13px;
  background: #fafafa;
  border: 1px solid #ddd;
  padding: 8px;
}
</style>
</head>
<body>
<header>
<button onclick="move(-1)">← Prev</button>
<button onclick="move(1)">Next →</button>
<button onclick="exportCSV()">Export CSV</button>
<span id="progress"></span>
</header>

<main>
<section class="card">
<div class="meta" id="meta"></div>
<img id="panel">
</section>

<aside class="card">
<div class="help">
Красный — найденная горизонтальная структура.<br>
Зелёный — skeleton.<br><br>
Оцениваем сам sample, наличие линии и эффект простого удаления.
Ничего пока автоматически не удаляем.
</div>

<div id="controls"></div>

<div class="group">
<h3>Notes</h3>
<textarea id="notes"></textarea>
</div>

<div id="list"></div>
</aside>
</main>

<script>
const items = __ITEMS__;
let index = 0;
const annotations = JSON.parse(
  localStorage.getItem("school_iter2_annotations") || "{}"
);

const fields = [
  {
    key: "sample_validity",
    title: "Sample validity",
    options: [
      "valid",
      "valid_but_short",
      "partial_or_clipped",
      "multiple_fragments",
      "background_or_empty",
      "unreadable",
      "other_invalid"
    ]
  },
  {
    key: "transcription",
    title: "Transcription",
    options: [
      "match",
      "mismatch",
      "uncertain"
    ]
  },
  {
    key: "ruling_line",
    title: "Ruling / grid line",
    options: [
      "none",
      "present_not_intersecting",
      "intersects_handwriting",
      "dominant",
      "detector_false_positive"
    ]
  },
  {
    key: "naive_removal",
    title: "Naive line removal",
    options: [
      "improves",
      "partial_improvement",
      "no_effect",
      "damages_handwriting",
      "not_applicable"
    ]
  },
  {
    key: "merge_context",
    title: "Needs larger context / merging",
    options: [
      "no",
      "yes_probably_same_line",
      "yes_context_but_unknown",
      "invalid_not_mergeable"
    ]
  }
];

function current() {
  return items[index];
}

function ann() {
  const id = current().sample_id;
  if (!annotations[id]) annotations[id] = {};
  return annotations[id];
}

function save() {
  localStorage.setItem(
    "school_iter2_annotations",
    JSON.stringify(annotations)
  );
}

function choose(key, value) {
  ann()[key] = value;
  save();
  render();
}

function isDone(item) {
  const a = annotations[item.sample_id] || {};
  return fields.every(f => a[f.key]);
}

function renderControls() {
  const a = ann();
  const root = document.getElementById("controls");
  root.innerHTML = "";

  for (const field of fields) {
    const div = document.createElement("div");
    div.className = "group";

    const title = document.createElement("h3");
    title.textContent = field.title;
    div.appendChild(title);

    for (const option of field.options) {
      const button = document.createElement("button");
      button.className =
        "choice" + (a[field.key] === option ? " selected" : "");
      button.textContent = option;
      button.onclick = () => choose(field.key, option);
      div.appendChild(button);
    }

    root.appendChild(div);
  }
}

function renderList() {
  const root = document.getElementById("list");
  root.innerHTML = "";

  items.forEach((item, i) => {
    const button = document.createElement("button");
    button.textContent =
      `${i + 1}. ${item.sample_id} | ${item.stratum} | ${item.text}`;

    if (isDone(item)) button.classList.add("done");
    if (i === index) button.classList.add("current");

    button.onclick = () => {
      storeNotes();
      index = i;
      render();
    };

    root.appendChild(button);
  });
}

function storeNotes() {
  if (!items.length) return;
  ann().notes = document.getElementById("notes").value;
  save();
}

function render() {
  const item = current();
  const a = ann();

  document.getElementById("progress").textContent =
    `${index + 1}/${items.length} | done ${items.filter(isDone).length}`;

  document.getElementById("meta").innerHTML =
    `<b>${item.sample_id}</b><br>` +
    `split=${item.split}, stratum=${item.stratum}<br>` +
    `text=<b>${item.text}</b><br>` +
    `width=${item.width}, text_len=${item.text_len}, ` +
    `fg=${item.fg_fraction}, line=${item.horizontal_line_fraction}, ` +
    `line_width=${item.horizontal_line_width_fraction}, ` +
    `threshold=${item.threshold_used}`;

  document.getElementById("panel").src = item.panel;
  document.getElementById("notes").value = a.notes || "";

  renderControls();
  renderList();
}

function move(delta) {
  storeNotes();
  index = Math.max(
    0,
    Math.min(items.length - 1, index + delta)
  );
  render();
}

function csvEscape(value) {
  const text = String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function exportCSV() {
  storeNotes();

  const header = [
    "sample_id",
    "split",
    "stratum",
    "text",
    "image_path",
    "fg_fraction",
    "horizontal_line_fraction",
    "horizontal_line_width_fraction",
    ...fields.map(f => f.key),
    "notes"
  ];

  const rows = [header];

  for (const item of items) {
    const a = annotations[item.sample_id] || {};

    rows.push([
      item.sample_id,
      item.split,
      item.stratum,
      item.text,
      item.image_path,
      item.fg_fraction,
      item.horizontal_line_fraction,
      item.horizontal_line_width_fraction,
      ...fields.map(f => a[f.key] || ""),
      a.notes || ""
    ]);
  }

  const csv = rows
    .map(row => row.map(csvEscape).join(","))
    .join("\n");

  const blob = new Blob(
    [csv],
    {type: "text/csv;charset=utf-8"}
  );

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download =
    "school_iter2_calibration_annotations.csv";

  link.click();
  URL.revokeObjectURL(url);
}

render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics_csv",
        required=True,
    )
    parser.add_argument(
        "--out_html",
        required=True,
    )
    parser.add_argument(
        "--out_sample_csv",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260616,
    )

    args = parser.parse_args()

    rows = read_csv(Path(args.metrics_csv))
    sampled = stratified_sample(
        rows,
        seed=args.seed,
    )

    counts = Counter(
        row["calibration_stratum"]
        for row in sampled
    )

    print(
        json.dumps(
            {
                "sample_n": len(sampled),
                "by_stratum": dict(counts),
                "by_split": dict(Counter(
                    row["split"]
                    for row in sampled
                )),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    write_csv(
        sampled,
        Path(args.out_sample_csv),
    )

    items = [
        prepare_item(row)
        for row in sampled
    ]

    html = HTML.replace(
        "__ITEMS__",
        json.dumps(
            items,
            ensure_ascii=False,
        ),
    )

    out_html = Path(args.out_html)
    out_html.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_html.write_text(
        html,
        encoding="utf-8",
    )

    print("wrote:", out_html)


if __name__ == "__main__":
    main()