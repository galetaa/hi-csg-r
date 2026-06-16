from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("metrics", obj)


def parse_condition(name: str) -> tuple[str, str]:
    if name == "clean":
        return "clean", "clean"

    level = name.rsplit("_", 1)[-1]
    if level in {"mild", "medium", "strong"}:
        return name.rsplit("_", 1)[0], level

    return name, "unknown"


def read_mode(path: Path) -> dict[str, dict[str, Any]]:
    out = {}

    for directory in sorted(path.iterdir()):
        summary_path = directory / "summary.json"

        if not directory.is_dir() or not summary_path.exists():
            continue

        out[directory.name] = load(summary_path)

    if "clean" not in out:
        raise RuntimeError(f"No clean result in {path}")

    return out


def condition_rows(
    mode: str,
    data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    clean_cer = float(data["clean"]["cer"])
    rows = []

    for condition, summary in sorted(data.items()):
        distortion, level = parse_condition(condition)
        cer = float(summary["cer"])

        rows.append({
            "mode": mode,
            "condition": condition,
            "distortion": distortion,
            "level": level,
            "cer": cer,
            "clean_cer": clean_cer,
            "absolute_delta": cer - clean_cer,
            "relative_degradation": (
                (cer - clean_cer) / max(clean_cer, 1e-12)
            ),
        })

    return rows


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = next(r for r in rows if r["condition"] == "clean")
    distorted = [
        r for r in rows
        if r["condition"] != "clean"
    ]

    by_distortion: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in distorted:
        by_distortion[row["distortion"]].append(row)

    return {
        "clean_cer": clean["cer"],
        "mean_distorted_cer": mean([
            r["cer"] for r in distorted
        ]),
        "mean_absolute_delta": mean([
            r["absolute_delta"] for r in distorted
        ]),
        "mean_relative_degradation": mean([
            r["relative_degradation"] for r in distorted
        ]),
        "by_distortion": {
            name: {
                "mean_cer": mean([r["cer"] for r in group]),
                "mean_relative_degradation": mean([
                    r["relative_degradation"]
                    for r in group
                ]),
            }
            for name, group in sorted(by_distortion.items())
        },
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return "n/a"


def pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_only_dir", required=True)
    parser.add_argument("--frozen_graph_dir", required=True)
    parser.add_argument("--recomputed_graph_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    modes = {
        "image_only": read_mode(Path(args.image_only_dir)),
        "graph_frozen_clean": read_mode(Path(args.frozen_graph_dir)),
        "graph_recomputed_v3": read_mode(
            Path(args.recomputed_graph_dir)
        ),
    }

    rows = {
        name: condition_rows(name, values)
        for name, values in modes.items()
    }

    summaries = {
        name: summarize(values)
        for name, values in rows.items()
    }

    conditions = sorted(
        set.intersection(*[
            set(values.keys())
            for values in modes.values()
        ])
    )

    condition_comparison = []

    for condition in conditions:
        condition_comparison.append({
            "condition": condition,
            "image_only_cer": float(
                modes["image_only"][condition]["cer"]
            ),
            "frozen_graph_cer": float(
                modes["graph_frozen_clean"][condition]["cer"]
            ),
            "recomputed_graph_cer": float(
                modes["graph_recomputed_v3"][condition]["cer"]
            ),
        })

    result = {
        "summaries": summaries,
        "condition_comparison": condition_comparison,
        "definitions": {
            "graph_frozen_clean": (
                "Distorted image evaluated with graph features "
                "inherited from the clean manifest."
            ),
            "graph_recomputed_v3": (
                "Distorted image evaluated with graph features "
                "recomputed from the distorted image."
            ),
        },
    }

    lines = []
    lines.append("# Robustness with recomputed graph features")
    lines.append("")
    lines.append("## 1. Overall")
    lines.append("")
    lines.append(
        "| mode | clean CER | mean distorted CER | "
        "absolute delta | relative degradation |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    for name in [
        "image_only",
        "graph_frozen_clean",
        "graph_recomputed_v3",
    ]:
        s = summaries[name]
        lines.append(
            f"| `{name}` | "
            f"{fmt(s['clean_cer'])} | "
            f"{fmt(s['mean_distorted_cer'])} | "
            f"{fmt(s['mean_absolute_delta'])} | "
            f"{pct(s['mean_relative_degradation'])} |"
        )

    lines.append("")
    lines.append("## 2. By distortion family")
    lines.append("")
    lines.append(
        "| distortion | image-only rel. degradation | "
        "frozen graph | recomputed graph |"
    )
    lines.append("|---|---:|---:|---:|")

    distortions = sorted(
        summaries["image_only"]["by_distortion"]
    )

    for distortion in distortions:
        lines.append(
            f"| `{distortion}` | "
            f"{pct(summaries['image_only']['by_distortion'][distortion]['mean_relative_degradation'])} | "
            f"{pct(summaries['graph_frozen_clean']['by_distortion'][distortion]['mean_relative_degradation'])} | "
            f"{pct(summaries['graph_recomputed_v3']['by_distortion'][distortion]['mean_relative_degradation'])} |"
        )

    lines.append("")
    lines.append("## 3. Methodological interpretation")
    lines.append("")

    frozen = summaries["graph_frozen_clean"][
        "mean_relative_degradation"
    ]
    recomputed = summaries["graph_recomputed_v3"][
        "mean_relative_degradation"
    ]
    image = summaries["image_only"][
        "mean_relative_degradation"
    ]

    if recomputed < image:
        lines.append(
            "The graph model retains a relative robustness advantage "
            "when graph features are recomputed from distorted images."
        )
    else:
        lines.append(
            "The relative robustness advantage does not survive "
            "end-to-end graph-feature recomputation."
        )

    if recomputed > frozen:
        lines.append(
            "Frozen clean graph features made the previous robustness "
            "estimate optimistic."
        )
    else:
        lines.append(
            "Recomputed graph features do not degrade robustness more "
            "than the previous frozen-feature protocol."
        )

    lines.append("")
    lines.append(
        "Absolute CER must still be compared directly: lower relative "
        "degradation alone does not establish better recognition."
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("wrote:", out_json)
    print("wrote:", out_md)


if __name__ == "__main__":
    main()