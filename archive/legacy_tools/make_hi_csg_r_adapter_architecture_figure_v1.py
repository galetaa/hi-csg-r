from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hi_csg_r_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def box(axis: object, x: float, y: float, width: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.11,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.2,
        edgecolor="#222222",
        facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + 0.055, text, ha="center", va="center", fontsize=9)


def arrow(axis: object, x0: float, y0: float, x1: float, y1: float) -> None:
    axis.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    figure, axis = plt.subplots(figsize=(14, 4))
    axis.set_xlim(0, 1.03)
    axis.set_ylim(0, 1)
    axis.axis("off")
    visual = [
        (0.03, "Image"),
        (0.16, "Existing CNN"),
        (0.31, "Visual projection\nV[t]"),
        (0.63, "Residual fusion"),
        (0.77, "Existing BiLSTM"),
        (0.90, "Linear + CTC"),
    ]
    for x, label in visual:
        box(axis, x, 0.66, 0.1, label, "#dbeafe")
    for left, right in zip(visual, visual[1:], strict=False):
        arrow(axis, left[0] + 0.1, 0.715, right[0], 0.715)
    graph = [
        (0.03, "HI-CSG-R"),
        (0.18, "x-aligned\n20 features"),
        (0.36, "Temporal adapter\n20→64→128→256"),
        (0.54, "Quality gate\na[t]"),
    ]
    for x, label in graph:
        box(axis, x, 0.22, 0.12, label, "#dcfce7")
    for left, right in zip(graph, graph[1:], strict=False):
        arrow(axis, left[0] + 0.12, 0.275, right[0], 0.275)
    arrow(axis, 0.60, 0.33, 0.68, 0.66)
    axis.text(0.42, 0.08, "Auxiliary graph CTC (training only)", ha="center", fontsize=9)
    arrow(axis, 0.42, 0.22, 0.42, 0.12)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
