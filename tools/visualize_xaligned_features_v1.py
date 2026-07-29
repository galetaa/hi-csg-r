from __future__ import annotations

import argparse
import html
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hi_csg_r_matplotlib"))

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.xaligned_hi_csg_r import (  # noqa: E402
    FEATURE_NAMES,
    load_feature_record,
    load_or_extract_graph,
    locate_graph_path,
    read_jsonl,
    resolve_path,
)


def domain_key(row: dict[str, Any]) -> str:
    value = str(row.get("dataset") or row.get("source_dataset") or "unknown").lower()
    if "school" in value:
        return "school"
    if "hkr" in value:
        return "hkr"
    if "cyr" in value:
        return "cyrillic"
    return value


def select_rows(rows: list[dict[str, Any]], manifest: Path, field: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for domain in ("cyrillic", "hkr", "school"):
        candidates = []
        for row in rows:
            if domain_key(row) != domain:
                continue
            record = load_feature_record(resolve_path(str(row[field]), manifest))
            features = np.asarray(record["features"])
            score = float(features[:, 19].mean() + (1.0 - features[:, 18].mean()))
            candidates.append((score, row))
        candidates.sort(key=lambda item: (item[0], str(item[1]["sample_id"])))
        if not candidates:
            continue
        indices = np.linspace(0, len(candidates) - 1, min(10, len(candidates))).astype(int)
        for index in indices:
            picked = dict(candidates[index][1])
            fraction = index / max(len(candidates) - 1, 1)
            picked["_audit_stratum"] = (
                "clean" if fraction < 1 / 3 else "medium" if fraction < 2 / 3 else "hard"
            )
            selected.append(picked)
    return selected


def overlay_graph(axis: Any, graph: dict[str, Any], time_steps: int, width: int) -> None:
    for edge in graph.get("edges", []):
        points = np.asarray(edge.get("polyline") or edge.get("points") or [], dtype=float)
        if points.ndim == 2 and len(points) >= 2:
            axis.plot(points[:, 0], points[:, 1], color="#d62728", linewidth=0.7, alpha=0.8)
    colors = {"endpoint": "#1f77b4", "junction": "#ffbf00", "junction_cluster": "#ffbf00"}
    for node in graph.get("nodes", []):
        axis.scatter(
            [float(node.get("x", 0))],
            [float(node.get("y", 0))],
            s=10,
            color=colors.get(str(node.get("type")), "#2ca02c"),
        )
    for index in range(1, time_steps):
        axis.axvline(index * width / time_steps, color="white", linewidth=0.35, alpha=0.7)


def render(
    row: dict[str, Any],
    manifest: Path,
    args: argparse.Namespace,
    output: Path,
    m0_predictions: dict[str, dict[str, Any]],
    m3_predictions: dict[str, dict[str, Any]],
) -> None:
    image_path = resolve_path(str(row["image_path"]), manifest)
    feature_path = resolve_path(str(row[args.feature_field]), manifest)
    record = load_feature_record(feature_path)
    graph_path = locate_graph_path(
        row,
        graph_field=args.graph_field,
        graph_root=args.graph_root,
        manifest_path=manifest,
    )
    foreground, skeleton, graph, _ = load_or_extract_graph(
        image_path,
        str(row.get("dataset") or row.get("source_dataset") or "unknown"),
        graph_path=graph_path,
    )
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"))
    features = np.asarray(record["features"])
    time_steps = int(record["time_steps"])
    x = np.arange(time_steps)

    figure, axes = plt.subplots(8, 1, figsize=(16, 15), constrained_layout=True)
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    sample_id = str(row["sample_id"])
    m0_text = m0_predictions.get(sample_id, {}).get("prediction", "n/a")
    m3_text = m3_predictions.get(sample_id, {}).get("prediction", "n/a")
    axes[0].set_title(
        f"{sample_id} | {domain_key(row)} | target={row.get('text', '')} | "
        f"stratum={row.get('_audit_stratum')} | M0-FT={m0_text} | M3={m3_text}"
    )
    axes[1].imshow(foreground, cmap="gray")
    axes[1].set_title("foreground")
    axes[2].imshow(skeleton, cmap="gray")
    overlay_graph(axes[2], graph, time_steps, gray.shape[1])
    axes[2].set_title("HI-CSG-R and x bins")
    groups = [
        range(0, 5),
        range(5, 10),
        range(10, 17),
        range(17, 20),
    ]
    for axis, indices in zip(axes[3:7], groups, strict=True):
        for index in indices:
            axis.plot(x, features[:, index], label=FEATURE_NAMES[index], linewidth=1)
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=7, loc="upper right")
    gate_curve = m3_predictions.get(sample_id, {}).get("gate_curve")
    if gate_curve:
        axes[7].plot(np.arange(len(gate_curve)), gate_curve, color="#9467bd")
    axes[7].set_ylim(0, 1)
    axes[7].set_title("quality-aware gate")
    axes[7].grid(alpha=0.2)
    for axis in axes[:3]:
        axis.set_axis_off()
    figure.savefig(output, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature_field", default="xaligned_graph_npz")
    parser.add_argument("--graph_field")
    parser.add_argument("--graph_root")
    parser.add_argument("--m0_predictions")
    parser.add_argument("--m3_predictions")
    parser.add_argument("--out_dir", default="outputs/htr_adapter_v1/feature_audit/browser")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    rows = select_rows(read_jsonl(manifest), manifest, args.feature_field)
    output = Path(args.out_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    m0_predictions = (
        {str(row["sample_id"]): row for row in read_jsonl(args.m0_predictions)}
        if args.m0_predictions
        else {}
    )
    m3_predictions = (
        {str(row["sample_id"]): row for row in read_jsonl(args.m3_predictions)}
        if args.m3_predictions
        else {}
    )
    index_rows = []
    for index, row in enumerate(rows):
        name = f"{index:02d}_{row['sample_id']}.png".replace("/", "_")
        render(
            row,
            manifest,
            args,
            image_dir / name,
            m0_predictions,
            m3_predictions,
        )
        index_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "dataset": domain_key(row),
                "stratum": row.get("_audit_stratum"),
                "image": f"images/{name}",
            }
        )
    cards = "\n".join(
        f'<figure><img src="{html.escape(row["image"])}">'
        f'<figcaption>{html.escape(row["dataset"])} | {html.escape(str(row["stratum"]))} | '
        f'{html.escape(row["sample_id"])}</figcaption></figure>'
        for row in index_rows
    )
    (output / "browser.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>x-aligned audit</title>"
        "<style>body{font:14px sans-serif;margin:20px}figure{margin:0 0 28px}"
        "img{max-width:100%;height:auto}figcaption{font-weight:600}</style>"
        f"<h1>X-Aligned HI-CSG-R Audit</h1>{cards}",
        encoding="utf-8",
    )
    (output / "selection.json").write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"written": len(index_rows), "out_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
