from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hi_csg_r_matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from src.htr.xaligned_hi_csg_r import (
    load_or_extract_graph,
    resolve_path,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def edit_operations(source: str, target: str) -> dict[str, int]:
    rows = len(source) + 1
    columns = len(target) + 1
    costs = np.zeros((rows, columns), dtype=np.int32)
    operations: list[list[tuple[int, int, int] | None]] = [
        [None] * columns for _ in range(rows)
    ]
    costs[:, 0] = np.arange(rows)
    costs[0, :] = np.arange(columns)
    for index in range(1, rows):
        operations[index][0] = (index - 1, 0, 1)
    for index in range(1, columns):
        operations[0][index] = (0, index - 1, 2)
    for row in range(1, rows):
        for column in range(1, columns):
            substitution = costs[row - 1, column - 1] + (
                source[row - 1] != target[column - 1]
            )
            deletion = costs[row - 1, column] + 1
            insertion = costs[row, column - 1] + 1
            options = (
                (substitution, (row - 1, column - 1, 0)),
                (deletion, (row - 1, column, 1)),
                (insertion, (row, column - 1, 2)),
            )
            cost, operation = min(options, key=lambda value: value[0])
            costs[row, column] = cost
            operations[row][column] = operation
    counts = {"substitution": 0, "deletion": 0, "insertion": 0}
    row, column = len(source), len(target)
    while row or column:
        previous = operations[row][column]
        if previous is None:
            break
        previous_row, previous_column, operation = previous
        if operation == 0 and source[row - 1] != target[column - 1]:
            counts["substitution"] += 1
        elif operation == 1:
            counts["deletion"] += 1
        elif operation == 2:
            counts["insertion"] += 1
        row, column = previous_row, previous_column
    return counts


def classify(row: dict[str, Any]) -> str:
    baseline_correct = bool(row["baseline_exact"])
    final_correct = bool(row["exact"])
    if baseline_correct and not final_correct:
        return "A_baseline_correct_v2_wrong"
    if not baseline_correct and final_correct:
        return "B_baseline_wrong_v2_correct"
    if not baseline_correct:
        return "C_both_wrong"
    return "D_both_correct"


def enrich_rows(
    predictions: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    enriched: list[dict[str, Any]] = []
    for row in predictions:
        source = manifest[str(row["sample_id"])]
        final_ops = edit_operations(str(row["prediction"]), str(row["target"]))
        base_ops = edit_operations(
            str(row["baseline_prediction"]),
            str(row["target"]),
        )
        enriched.append(
            {
                **row,
                "analysis_group": classify(row),
                "alpha": alpha,
                "image_path": source["image_path"],
                "xaligned_graph_npz": source.get("xaligned_graph_npz"),
                "target_length": len(str(row["target"])),
                **{f"final_{key}": value for key, value in final_ops.items()},
                **{f"baseline_{key}": value for key, value in base_ops.items()},
            }
        )
    return enriched


def choose_cases(rows: list[dict[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    helps = sorted(
        (row for row in rows if row["edit_delta_vs_baseline"] < 0),
        key=lambda row: (
            row["edit_delta_vs_baseline"],
            -row["correction_norm_mean"],
        ),
    )[:limit]
    hurts = sorted(
        (row for row in rows if row["edit_delta_vs_baseline"] > 0),
        key=lambda row: (
            -row["edit_delta_vs_baseline"],
            -row["correction_norm_mean"],
        ),
    )[:limit]
    high_unchanged = sorted(
        (row for row in rows if not row["prediction_changed"]),
        key=lambda row: (-row["gate_mean"], -row["correction_norm_mean"]),
    )[:limit]
    errors = [row for row in rows if not row["exact"]]
    low_errors = sorted(
        errors,
        key=lambda row: (row["gate_mean"], row["correction_norm_mean"]),
    )[:limit]
    return {
        "graph_helps": helps,
        "graph_hurts": hurts,
        "high_intervention_unchanged": high_unchanged,
        "low_intervention_errors": low_errors,
    }


def draw_graph_overlay(axis: Any, row: dict[str, Any], manifest_path: Path) -> None:
    image_path = resolve_path(str(row["image_path"]), manifest_path)
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"))
    axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
    try:
        _, _, graph, _ = load_or_extract_graph(
            image_path,
            str(row.get("dataset") or row.get("source_dataset") or "unknown"),
        )
        for edge in graph.get("edges", []):
            points = edge.get("points") or edge.get("polyline") or []
            if len(points) >= 2:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                axis.plot(xs, ys, color="#00a6a6", linewidth=0.7, alpha=0.8)
        for node in graph.get("nodes", []):
            x = float(node.get("x", node.get("cx", 0.0)))
            y = float(node.get("y", node.get("cy", 0.0)))
            axis.scatter([x], [y], s=7, color="#dc2626")
    except Exception:
        pass
    axis.axis("off")


def create_case_figure(
    cases: dict[str, list[dict[str, Any]]],
    manifest_rows: dict[str, dict[str, Any]],
    manifest_path: Path,
    out: Path,
) -> None:
    selected = [
        ("helps", row) for row in cases["graph_helps"][:4]
    ] + [("hurts", row) for row in cases["graph_hurts"][:4]]
    if not selected:
        return
    figure, axes = plt.subplots(len(selected), 1, figsize=(13, 2.0 * len(selected)))
    axes = np.atleast_1d(axes)
    for axis, (label, row) in zip(axes, selected, strict=True):
        source = manifest_rows[str(row["sample_id"])]
        draw_graph_overlay(axis, source, manifest_path)
        axis.set_title(
            f"{label}: target={row['target']!r} | "
            f"B0={row['baseline_prediction']!r} | V2={row['prediction']!r} | "
            f"delta edits={row['edit_delta_vs_baseline']:+d}, "
            f"u={row['visual_uncertainty_mean']:.3f}, "
            f"gate={row['gate_mean']:.3f}",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    predictions = load_jsonl(args.predictions)
    manifest_rows = load_jsonl(args.manifest)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = enrich_rows(predictions, manifest_rows, float(summary["alpha"]))
    cases = choose_cases(rows, args.limit)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_sample_intervention.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, selected in cases.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["analysis_group"])].append(row)
    group_summary = {
        group: {
            "samples": len(values),
            "cer": sum(row["char_edits"] for row in values)
            / max(sum(row["target_chars"] for row in values), 1),
            "uncertainty_mean": float(
                np.mean([row["visual_uncertainty_mean"] for row in values])
            ),
            "gate_mean": float(np.mean([row["gate_mean"] for row in values])),
            "correction_norm_mean": float(
                np.mean([row["correction_norm_mean"] for row in values])
            ),
        }
        for group, values in sorted(grouped.items())
    }
    analysis = {
        "samples": len(rows),
        "groups": group_summary,
        "case_counts": {key: len(value) for key, value in cases.items()},
        "intervention_precision": summary["intervention"]["precision"],
        "changed_prediction_rate": summary["intervention"][
            "prediction_change_rate"
        ],
    }
    (output / "failure_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# HI-CSG-R Late Correction v2: failure/intervention analysis",
        "",
        f"Samples: `{len(rows)}`",
        "",
        "| Group | N | CER | Mean uncertainty | Mean gate | Mean correction norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, values in group_summary.items():
        lines.append(
            f"| {group} | {values['samples']} | {values['cer']:.6f} | "
            f"{values['uncertainty_mean']:.6f} | {values['gate_mean']:.6f} | "
            f"{values['correction_norm_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- intervention precision: `{analysis['intervention_precision']:.6f}`",
            f"- changed prediction rate: `{analysis['changed_prediction_rate']:.6f}`",
            "",
            "Списки `graph_helps`, `graph_hurts`, `high_intervention_unchanged` "
            "и `low_intervention_errors` сохранены по 20 примеров, если в "
            "evaluation существует достаточно случаев соответствующего типа.",
        ]
    )
    (output / "failure_analysis.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    manifest_index = {str(row["sample_id"]): row for row in manifest_rows}
    create_case_figure(
        cases,
        manifest_index,
        Path(args.manifest),
        output / "figure_d_helps_hurts.png",
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
