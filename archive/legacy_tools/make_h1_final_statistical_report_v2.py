from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_scope(
    paired: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    for row in paired["results"]:
        if row.get("scope") == scope:
            return row

    raise KeyError(f"Scope not found: {scope}")


def fmt(value: Any, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def pct_ci(values: list[float]) -> str:
    return (
        f"{100.0 * float(values[0]):.2f}%–"
        f"{100.0 * float(values[1]):.2f}%"
    )


def raw_ci(values: list[float]) -> str:
    return (
        f"{float(values[0]):.5f}–"
        f"{float(values[1]):.5f}"
    )


def relative_status(row: dict[str, Any]) -> str:
    ci = row["bootstrap"]["relative_advantage_ci95"]
    p = row["permutation"]["relative_advantage_one_sided_p"]
    estimate = float(row["relative_advantage"])

    if ci[0] > 0 and p < 0.05:
        return "supported"

    if estimate <= 0 and p >= 0.05:
        return "not_supported"

    return "inconclusive"


def absolute_status(row: dict[str, Any]) -> str:
    ci = row["bootstrap"]["absolute_advantage_ci95"]

    if ci[0] > 0:
        return "graph_degrades_less"

    if ci[1] < 0:
        return "graph_degrades_more"

    return "inconclusive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_json", required=True)
    parser.add_argument("--modes_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    paired = load_json(args.paired_json)
    modes = load_json(args.modes_json)

    overall = find_scope(paired, "overall")

    datasets = [
        find_scope(paired, "dataset:cyrillic_handwriting"),
        find_scope(paired, "dataset:hkr_words"),
        find_scope(paired, "dataset:school_notebooks_clean"),
    ]

    families = [
        find_scope(paired, "family:blur"),
        find_scope(paired, "family:low_contrast"),
        find_scope(paired, "family:noise"),
        find_scope(paired, "family:thick_strokes"),
        find_scope(paired, "family:thin_strokes"),
    ]

    relative_ci = overall["bootstrap"][
        "relative_advantage_ci95"
    ]
    absolute_ci = overall["bootstrap"][
        "absolute_advantage_ci95"
    ]
    gap_ci = overall["bootstrap"][
        "distorted_cer_gap_ci95"
    ]

    relative_supported = (
        relative_ci[0] > 0
        and overall["permutation"][
            "relative_advantage_one_sided_p"
        ] < 0.05
    )

    absolute_supported = absolute_ci[0] > 0
    graph_absolute_better = gap_ci[1] < 0

    strong_h1 = bool(
        relative_supported
        and absolute_supported
        and graph_absolute_better
    )

    partial_h1 = bool(
        relative_supported
        and not strong_h1
    )

    result = {
        "verdict": {
            "strong_h1_supported": strong_h1,
            "partial_h1_supported": partial_h1,
            "relative_robustness_supported": (
                relative_supported
            ),
            "absolute_degradation_advantage_supported": (
                absolute_supported
            ),
            "absolute_distorted_htr_advantage_supported": (
                graph_absolute_better
            ),
        },
        "primary_estimand": (
            "paired cluster-bootstrap corpus-level "
            "relative CER degradation advantage"
        ),
        "overall": overall,
        "datasets": {
            row["scope"]: {
                **row,
                "relative_status": relative_status(row),
                "absolute_status": absolute_status(row),
            }
            for row in datasets
        },
        "families": {
            row["scope"]: {
                **row,
                "relative_status": relative_status(row),
                "absolute_status": absolute_status(row),
            }
            for row in families
        },
        "descriptive_condition_average": (
            modes["summaries"]
        ),
    }

    lines: list[str] = []

    lines.append("# H1 final statistical robustness report v2")
    lines.append("")

    lines.append("## 1. Strict verdict")
    lines.append("")
    lines.append("```text")
    lines.append(
        f"Strong H1 supported: "
        f"{'yes' if strong_h1 else 'no'}"
    )
    lines.append(
        f"Partial H1 supported: "
        f"{'yes' if partial_h1 else 'no'}"
    )
    lines.append(
        f"Relative robustness supported: "
        f"{'yes' if relative_supported else 'no'}"
    )
    lines.append(
        f"Absolute HTR advantage supported: "
        f"{'yes' if graph_absolute_better else 'no'}"
    )
    lines.append("```")
    lines.append("")

    lines.append("## 2. Primary paired result")
    lines.append("")
    lines.append("| metric | result |")
    lines.append("|---|---:|")
    lines.append(
        f"| image-only relative degradation | "
        f"{pct(overall['image_relative_degradation'])} |"
    )
    lines.append(
        f"| graph relative degradation | "
        f"{pct(overall['graph_relative_degradation'])} |"
    )
    lines.append(
        f"| relative robustness advantage | "
        f"{pct(overall['relative_advantage'])} |"
    )
    lines.append(
        f"| relative advantage 95% CI | "
        f"{pct_ci(relative_ci)} |"
    )
    lines.append(
        f"| one-sided permutation p | "
        f"{overall['permutation']['relative_advantage_one_sided_p']:.6f} |"
    )
    lines.append(
        f"| absolute degradation advantage | "
        f"{fmt(overall['absolute_advantage'])} |"
    )
    lines.append(
        f"| absolute advantage 95% CI | "
        f"{raw_ci(absolute_ci)} |"
    )
    lines.append(
        f"| graph − image distorted CER | "
        f"{fmt(overall['distorted_cer_gap'])} |"
    )
    lines.append(
        f"| distorted CER gap 95% CI | "
        f"{raw_ci(gap_ci)} |"
    )
    lines.append("")

    lines.append("## 3. Results by dataset")
    lines.append("")
    lines.append(
        "| dataset | image rel. | graph rel. | "
        "advantage | 95% CI | p | relative verdict |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---|"
    )

    for row in datasets:
        name = row["scope"].removeprefix("dataset:")
        ci = row["bootstrap"]["relative_advantage_ci95"]
        p = row["permutation"][
            "relative_advantage_one_sided_p"
        ]

        lines.append(
            f"| `{name}` | "
            f"{pct(row['image_relative_degradation'])} | "
            f"{pct(row['graph_relative_degradation'])} | "
            f"{pct(row['relative_advantage'])} | "
            f"{pct_ci(ci)} | "
            f"{p:.6f} | "
            f"`{relative_status(row)}` |"
        )

    lines.append("")

    lines.append("## 4. Results by distortion family")
    lines.append("")
    lines.append(
        "| family | image rel. | graph rel. | "
        "advantage | 95% CI | p | relative verdict | "
        "absolute verdict |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---|---|"
    )

    for row in families:
        name = row["scope"].removeprefix("family:")
        ci = row["bootstrap"]["relative_advantage_ci95"]
        p = row["permutation"][
            "relative_advantage_one_sided_p"
        ]

        lines.append(
            f"| `{name}` | "
            f"{pct(row['image_relative_degradation'])} | "
            f"{pct(row['graph_relative_degradation'])} | "
            f"{pct(row['relative_advantage'])} | "
            f"{pct_ci(ci)} | "
            f"{p:.6f} | "
            f"`{relative_status(row)}` | "
            f"`{absolute_status(row)}` |"
        )

    lines.append("")

    lines.append("## 5. Estimand note")
    lines.append("")
    lines.append(
        "The primary inferential result uses corpus-level CER "
        "with paired cluster resampling over the 5,563 clean "
        "source samples. Each source sample is retained together "
        "with all 15 distortion conditions."
    )
    lines.append("")
    lines.append(
        "The earlier 38.20% image-only degradation is the "
        "arithmetic mean of condition-level relative degradation. "
        "The paired corpus estimate is 33.77%. Both are valid "
        "descriptive quantities, but the paired corpus analysis "
        "is used for statistical inference."
    )
    lines.append("")

    lines.append("## 6. Final scientific conclusion")
    lines.append("")
    lines.append(
        "The graph-vector model exhibits a statistically "
        "supported reduction in relative CER degradation under "
        "the tested visual distortions. The overall relative "
        f"advantage is {pct(overall['relative_advantage'])}, "
        f"with a 95% cluster-bootstrap interval of "
        f"{pct_ci(relative_ci)} and a one-sided paired "
        f"permutation p-value of "
        f"{overall['permutation']['relative_advantage_one_sided_p']:.6f}."
    )
    lines.append("")
    lines.append(
        "This advantage is supported for low contrast, additive "
        "noise, and thinning of strokes; blur is inconclusive "
        "under the combined bootstrap-and-permutation criterion, "
        "and no advantage is found for stroke thickening."
    )
    lines.append("")
    lines.append(
        "However, the graph model does not have a positive "
        "absolute degradation advantage, and its absolute CER "
        "on distorted images remains substantially worse than "
        "the image-only baseline. Therefore strong H1 is rejected. "
        "The evidence supports only a partial claim of lower "
        "relative sensitivity to distortion."
    )

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()