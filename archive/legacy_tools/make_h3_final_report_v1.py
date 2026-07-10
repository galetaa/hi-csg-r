from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(x: float) -> str:
    return f"{x:.4f}"


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h3_v1_summary", required=True)
    parser.add_argument("--h3_v2_summary", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    v1 = load(args.h3_v1_summary)
    v2 = load(args.h3_v2_summary)

    best_single_corr = v1.get("best_abs_spearman")
    best_single_auc = v1.get("best_high_error_auc")
    best_cv = v2.get("best_cv_by_feature_set", {})

    structural = best_cv.get("structural_core")
    non_geometry = best_cv.get("all_non_geometry")
    geometry = best_cv.get("geometry_control")
    quality = best_cv.get("quality_only")

    strict = {
        "h3_global_single_feature_supported": False,
        "h3_multifeature_supported": True,
        "h3_overall_verdict": "partial_support",
        "main_positive_result": structural,
        "main_negative_result": {
            "single_feature_global": best_single_corr,
            "single_feature_high_error": best_single_auc,
            "quality_only": quality,
        },
        "methodological_interpretation": (
            "Graph scalar features are weak as global individual correlates of CER, "
            "but structural multifeature descriptors provide useful high-error detection "
            "in stratified subsets. This supports a diagnostic interpretation of graph "
            "features, not a claim that graph features improve recognition."
        ),
    }

    lines: list[str] = []

    lines.append("# H3 final diagnostic report — v1")
    lines.append("")
    lines.append("## 1. Verdict")
    lines.append("")
    lines.append("```text")
    lines.append("Global single-feature H3: not supported")
    lines.append("Stratified multifeature H3: partially supported")
    lines.append("Overall H3: partial support")
    lines.append("```")
    lines.append("")
    lines.append(
        "The current graph features do not strongly explain CER as individual global "
        "variables. However, structural graph descriptors provide useful high-error "
        "detection in some stratified subsets, especially HKR word samples."
    )
    lines.append("")
    lines.append("## 2. Global single-feature result")
    lines.append("")

    if best_single_corr:
        lines.append(
            f"Best global Spearman feature: `{best_single_corr['feature']}` "
            f"with r = {fmt(best_single_corr['spearman_r'])}."
        )
    else:
        lines.append("No valid global correlation result.")

    if best_single_auc:
        lines.append(
            f"Best single-feature high-error detector: `{best_single_auc['feature']}` "
            f"with ROC-AUC = {fmt(best_single_auc['roc_auc_direction_invariant'])}."
        )
    else:
        lines.append("No valid single-feature high-error detector.")

    lines.append("")
    lines.append("Interpretation: this is weak. It is below the useful diagnostic threshold.")
    lines.append("")
    lines.append("## 3. Multifeature stratified result")
    lines.append("")
    lines.append("| feature set | group | n | ROC-AUC | PR-AUC | lift | top20 precision |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for name, row in [
        ("structural_core", structural),
        ("all_non_geometry", non_geometry),
        ("geometry_control", geometry),
        ("quality_only", quality),
    ]:
        if not row:
            continue

        lines.append(
            f"| `{name}` | `{row['group']}` | {row['n']} | "
            f"{fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} | "
            f"{fmt(row['pr_auc_lift_over_base_rate'])} | "
            f"{fmt(row['top20_precision'])} |"
        )

    lines.append("")
    lines.append("## 4. Methodological interpretation")
    lines.append("")
    lines.append(
        "The useful signal comes from multifeature structural descriptors, not from "
        "`warning_count`. Therefore, the current warning proxy should not be used as "
        "the main graph-confidence measure."
    )
    lines.append("")
    lines.append(
        "The result is not strong enough to claim that graph features explain recognition "
        "errors globally. It is strong enough to justify using graph structural descriptors "
        "for failure-case triage and for selecting samples for gold structural annotation."
    )
    lines.append("")
    lines.append("## 5. Consequence for the project")
    lines.append("")
    lines.append("```text")
    lines.append("Do not add new HTR architectures.")
    lines.append("Do not claim H3 is fully confirmed.")
    lines.append("Use H3 results to guide H2 gold-subset sampling and failure analysis.")
    lines.append("```")
    lines.append("")
    lines.append("## 6. Next step")
    lines.append("")
    lines.append(
        "Build an H2/H3 audit candidate pool with four types of samples: "
        "high-error/high-structural-risk, high-error/low-structural-risk, "
        "low-error/high-structural-risk, and low-error/low-structural-risk. "
        "This will expose whether graph structure actually corresponds to visible "
        "stroke failures."
    )

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(strict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()