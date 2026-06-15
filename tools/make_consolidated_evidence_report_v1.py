from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1_json", required=True)
    parser.add_argument("--h2_json", required=True)
    parser.add_argument("--h3_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    h1 = load_json(args.h1_json)
    h2 = load_json(args.h2_json)
    h3 = load_json(args.h3_json)

    h1_models = h1.get("models", {})
    h2_hkr_cyr = h2.get("hkr_plus_cyrillic", {})
    h2_school = h2.get("school_notebooks_decision", {})
    h3_positive = h3.get("main_positive_result") or {}
    h3_negative = h3.get("main_negative_result") or {}

    final = {
        "overall_verdict": "mixed_partial_support",
        "h1": {
            "verdict": "partial_support_only",
            "reason": (
                "Graph-aware models showed lower relative CER degradation, "
                "but worse clean and distorted absolute CER than image-only."
            ),
            "models": h1_models,
        },
        "h2": {
            "verdict": h2.get("h2_v1_verdict", "partial_support"),
            "hkr_plus_cyrillic": h2_hkr_cyr,
            "school_notebooks_decision": h2_school,
        },
        "h3": {
            "verdict": h3.get("h3_overall_verdict", "partial_support"),
            "main_positive_result": h3_positive,
            "main_negative_result": h3_negative,
            "interpretation": h3.get("methodological_interpretation"),
        },
        "safe_claims": [
            "Canonical visible-stroke graph descriptors are diagnostically useful in some settings.",
            "Graph-aware HTR variants are relatively less sensitive to distortions, but not better recognizers in absolute CER.",
            "HKR/Cyrillic graph extraction preserves visible stroke structure reasonably well in the audited subset.",
            "School-notebooks failures are dominated by upstream crop/binarization border artifacts.",
            "The current structural risk score is a hard-sample indicator, not a direct graph-quality score.",
        ],
        "unsafe_claims": [
            "Do not claim that graph-aware recognition beats the image-only baseline.",
            "Do not claim H1 is fully confirmed.",
            "Do not claim H2 holds uniformly across all datasets.",
            "Do not claim structural risk is equivalent to graph quality.",
            "Do not present school-notebooks failures as pure graph-topology failures.",
        ],
    }

    lines: list[str] = []

    lines.append("# HI-CSG-R consolidated evidence report — v1")
    lines.append("")
    lines.append("## 1. Executive verdict")
    lines.append("")
    lines.append("```text")
    lines.append("Overall result: mixed / partial support")
    lines.append("H1 robustness: partial support only")
    lines.append("H2 structural preservation: partial support with preprocessing exception")
    lines.append("H3 graph diagnostics: partial support")
    lines.append("```")
    lines.append("")
    lines.append(
        "The current evidence supports a narrower claim than originally hoped. "
        "Canonical visible-stroke graph descriptors are useful as diagnostic and robustness-related signals, "
        "but the current graph-aware HTR models do not outperform the image-only baseline in absolute recognition quality."
    )
    lines.append("")

    lines.append("## 2. H1 — Robustness")
    lines.append("")
    lines.append("### Verdict")
    lines.append("")
    lines.append("```text")
    lines.append("H1 strong form: not confirmed")
    lines.append("H1 weak/relative form: partially supported")
    lines.append("```")
    lines.append("")
    lines.append(
        "Graph-aware models showed lower relative CER degradation under distortions, "
        "but they had much worse clean CER and worse absolute distorted CER than the image-only baseline. "
        "Therefore this is not evidence that graph-aware recognition is better overall."
    )
    lines.append("")
    lines.append("| model | clean CER | mean distorted CER | absolute CER delta | relative degradation |")
    lines.append("|---|---:|---:|---:|---:|")

    for name in ["image_only", "graph_vector_v2", "gated_v2_dist"]:
        m = h1_models.get(name, {})
        lines.append(
            f"| `{name}` | "
            f"{fmt(m.get('clean_cer'))} | "
            f"{fmt(m.get('mean_distorted_cer'))} | "
            f"{fmt(m.get('mean_absolute_cer_delta'))} | "
            f"{pct(m.get('mean_relative_cer_degradation'))} |"
        )

    lines.append("")
    lines.append("### H1 conclusion")
    lines.append("")
    lines.append(
        "The correct claim is: graph-aware variants are less sensitive in relative terms, "
        "but they are not competitive with the image-only baseline in absolute CER. "
        "This supports robustness analysis, not a better HTR system claim."
    )
    lines.append("")

    lines.append("## 3. H2 — Structural graph preservation")
    lines.append("")
    lines.append("### Verdict")
    lines.append("")
    lines.append("```text")
    lines.append("H2-v1: partial support with school-notebooks preprocessing exception")
    lines.append("```")
    lines.append("")
    lines.append(
        "Manual audit indicates that HKR and Cyrillic samples generally preserve visible stroke structure. "
        "School-notebooks samples are dominated by upstream crop/border/binarization artifacts and should be reported separately."
    )
    lines.append("")
    lines.append("| subset | n | critical topology error rate | skeleton follows ink rate | mean graph quality |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| `HKR + Cyrillic` | "
        f"{h2_hkr_cyr.get('n', 'n/a')} | "
        f"{pct(h2_hkr_cyr.get('critical_topology_error_rate'))} | "
        f"{pct(h2_hkr_cyr.get('skeleton_follows_ink_rate'))} | "
        f"{fmt(h2_hkr_cyr.get('mean_graph_quality_0_3'), 3)} |"
    )

    by_dataset = h2.get("by_dataset", {})
    school = by_dataset.get("school_notebooks_clean", {})
    if school:
        lines.append(
            f"| `school_notebooks_clean` | "
            f"{school.get('n', 'n/a')} | "
            f"{pct(school.get('critical_topology_error_rate'))} | "
            f"{pct(school.get('skeleton_follows_ink_rate'))} | "
            f"{fmt(school.get('mean_graph_quality_0_3'), 3)} |"
        )

    lines.append("")
    lines.append("### School-notebooks exception")
    lines.append("")
    lines.append(
        "The school-notebooks subset is not valid evidence against the graph abstraction itself. "
        "The observed failures occur earlier: crop/background borders are binarized as foreground, "
        "which then corrupts skeletons and graphs. A simple border-connected-component suppression check was rejected "
        "because it either failed to fix the artifact or removed handwriting."
    )
    lines.append("")

    lines.append("## 4. H3 — Graph diagnostics")
    lines.append("")
    lines.append("### Verdict")
    lines.append("")
    lines.append("```text")
    lines.append("H3: partial support")
    lines.append("```")
    lines.append("")
    lines.append(
        "Global single-feature graph metrics do not strongly explain CER. "
        "However, multifeature structural descriptors provide useful high-error detection in stratified subsets."
    )
    lines.append("")

    if h3_positive:
        lines.append("| best multifeature signal | value |")
        lines.append("|---|---:|")
        lines.append(f"| feature set | `{h3_positive.get('feature_set', 'n/a')}` |")
        lines.append(f"| group | `{h3_positive.get('group', 'n/a')}` |")
        lines.append(f"| n | {h3_positive.get('n', 'n/a')} |")
        lines.append(f"| ROC-AUC | {fmt(h3_positive.get('roc_auc'))} |")
        lines.append(f"| PR-AUC | {fmt(h3_positive.get('pr_auc'))} |")
        lines.append(f"| PR-AUC lift | {fmt(h3_positive.get('pr_auc_lift_over_base_rate'))} |")
        lines.append(f"| top20 precision | {fmt(h3_positive.get('top20_precision'))} |")
        lines.append("")

    lines.append("### H3 conclusion")
    lines.append("")
    lines.append(
        "The structural descriptor set is useful for finding hard samples, but it should not be described as graph quality. "
        "Manual H2 audit showed that high structural risk often marks sample difficulty rather than visible skeleton failure."
    )
    lines.append("")

    lines.append("## 5. Safe claims")
    lines.append("")
    for claim in final["safe_claims"]:
        lines.append(f"- {claim}")
    lines.append("")

    lines.append("## 6. Unsafe claims to avoid")
    lines.append("")
    for claim in final["unsafe_claims"]:
        lines.append(f"- {claim}")
    lines.append("")

    lines.append("## 7. Recommended thesis/research framing")
    lines.append("")
    lines.append(
        "The project should be framed as evidence that offline handwriting images can be converted into reproducible "
        "visible-stroke structural descriptors that support interpretability, robustness analysis, and failure triage. "
        "It should not be framed as a new state-of-the-art HTR model."
    )
    lines.append("")
    lines.append("A precise final claim:")
    lines.append("")
    lines.append("> Canonical visible-stroke graph descriptors provide a useful intermediate diagnostic representation for offline handwritten text recognition. They show partial robustness and high-error detection value, while exposing preprocessing limitations in noisy cropped notebook data. Current graph-aware recognition models do not outperform a strong image-only recognizer in absolute CER.")
    lines.append("")

    lines.append("## 8. Next work")
    lines.append("")
    lines.append("1. Freeze architecture experiments.")
    lines.append("2. Treat school-notebooks as a preprocessing/crop-cleaning problem, not as graph topology evidence.")
    lines.append("3. If time remains, run a small preprocessing experiment only on school-notebooks, but do not retrain HTR.")
    lines.append("4. Prepare final figures/tables for H1/H2/H3.")
    lines.append("5. Write the limitations section explicitly.")

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()