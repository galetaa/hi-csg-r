from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def fmt(x: float) -> str:
    return f"{x:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    models = summary["models"]

    image = models["image_only"]
    graph = models["graph_vector_v2"]
    gated = models["gated_v2_dist"]

    image_clean = image["clean"]["cer"]

    report: dict[str, Any] = {
        "strict_h1_supported": False,
        "partial_robustness_signal": True,
        "reason": (
            "Graph-aware models reduce relative CER degradation, but their clean CER "
            "and distorted CER remain substantially worse than image-only."
        ),
        "models": {},
    }

    for name, m in [
        ("image_only", image),
        ("graph_vector_v2", graph),
        ("gated_v2_dist", gated),
    ]:
        clean = m["clean"]["cer"]
        report["models"][name] = {
            "clean_cer": clean,
            "mean_distorted_cer": m["mean_distorted_cer"],
            "mean_absolute_cer_delta": m["mean_absolute_cer_delta"],
            "mean_relative_cer_degradation": m["mean_relative_cer_degradation"],
            "clean_gap_vs_image_only": clean - image_clean,
            "clean_relative_gap_vs_image_only": (
                (clean - image_clean) / image_clean if name != "image_only" else 0.0
            ),
        }

    lines: list[str] = []
    lines.append("# H1 robustness report — v1")
    lines.append("")
    lines.append("## 1. Strict verdict")
    lines.append("")
    lines.append("```text")
    lines.append("Strong H1 supported: no")
    lines.append("Partial robustness signal: yes")
    lines.append("```")
    lines.append("")
    lines.append(
        "Graph-aware models show lower relative CER degradation under the tested "
        "distortion families, but they substantially underperform the image-only "
        "baseline on clean data and also have worse absolute CER on distorted data. "
        "Therefore the current evidence is insufficient for strong H1."
    )
    lines.append("")
    lines.append("## 2. Main metrics")
    lines.append("")
    lines.append(
        "| model | clean CER | mean distorted CER | absolute CER delta | relative degradation | clean gap vs image-only |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    for name in ["image_only", "graph_vector_v2", "gated_v2_dist"]:
        r = report["models"][name]
        lines.append(
            f"| `{name}` | "
            f"{fmt(r['clean_cer'])} | "
            f"{fmt(r['mean_distorted_cer'])} | "
            f"{fmt(r['mean_absolute_cer_delta'])} | "
            f"{pct(r['mean_relative_cer_degradation'])} | "
            f"{fmt(r['clean_gap_vs_image_only'])} |"
        )

    lines.append("")
    lines.append("## 3. Distortion-family interpretation")
    lines.append("")
    lines.append(
        "| distortion | image-only rel. degradation | graph-vector rel. degradation | gated rel. degradation | strict interpretation |"
    )
    lines.append("|---|---:|---:|---:|---|")

    distortions = sorted(image["by_distortion"].keys())
    for d in distortions:
        image_rel = image["by_distortion"][d]["mean_relative_cer_degradation"]
        graph_rel = graph["by_distortion"][d]["mean_relative_cer_degradation"]
        gated_rel = gated["by_distortion"][d]["mean_relative_cer_degradation"]

        if graph_rel < image_rel and gated_rel < image_rel:
            interp = "relative robustness signal"
        else:
            interp = "no consistent signal"

        lines.append(
            f"| `{d}` | {pct(image_rel)} | {pct(graph_rel)} | {pct(gated_rel)} | {interp} |"
        )

    lines.append("")
    lines.append("## 4. Methodological conclusion")
    lines.append("")
    lines.append(
        "The result should not be presented as a clean HTR improvement. "
        "The correct conclusion is narrower: graph-aware variants are less sensitive "
        "in relative terms to the tested visual distortions, but their absolute "
        "recognition quality is worse. This supports continuing with graph-quality "
        "and failure-analysis experiments, not further architecture search."
    )
    lines.append("")
    lines.append("## 5. Next required work")
    lines.append("")
    lines.append("1. Add paired bootstrap or permutation tests using per-sample predictions.")
    lines.append("2. Run H3: graph quality/confidence versus per-sample CER.")
    lines.append("3. Start the gold subset for H2 structural graph quality.")
    lines.append("4. Stop adding new HTR architectures unless H2/H3 exposes a specific failure mode.")

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()