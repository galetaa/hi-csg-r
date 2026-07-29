from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
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


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("wrote:", path)


def make_results_tables(h1: dict[str, Any], h2: dict[str, Any], h3: dict[str, Any]) -> str:
    h1_models = h1["models"]
    h2_ds = h2["by_dataset"]
    h2_hc = h2["hkr_plus_cyrillic"]
    h3_pos = h3["main_positive_result"]
    h3_neg = h3["main_negative_result"]

    lines: list[str] = []

    lines.append("# Final results tables — v1")
    lines.append("")
    lines.append("## Table 1. Hypothesis-level verdicts")
    lines.append("")
    lines.append("| hypothesis | verdict | supported claim | main caveat |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| H1 robustness | partial support | Graph-aware variants show lower relative CER degradation under distortions. | "
        "They remain worse than image-only in clean and distorted absolute CER. |"
    )
    lines.append(
        "| H2 structural preservation | partial support with preprocessing exception | HKR/Cyrillic audit samples usually preserve visible stroke structure. | "
        "School-notebooks failures are dominated by crop/border/binarization artifacts. |"
    )
    lines.append(
        "| H3 graph diagnostics | partial support | Structural descriptors can detect high-error samples in stratified subsets. | "
        "Global single-feature correlations are weak; risk is not graph quality. |"
    )
    lines.append("")

    lines.append("## Table 2. H1 robustness summary")
    lines.append("")
    lines.append("| model | clean CER | mean distorted CER | absolute CER delta | relative CER degradation | interpretation |")
    lines.append("|---|---:|---:|---:|---:|---|")

    interp = {
        "image_only": "Best absolute recognizer; primary baseline.",
        "graph_vector_v2": "Lower relative degradation but worse absolute CER.",
        "gated_v2_dist": "Lowest relative degradation but worse absolute CER and low graph gate.",
    }

    for name in ["image_only", "graph_vector_v2", "gated_v2_dist"]:
        m = h1_models[name]
        lines.append(
            f"| `{name}` | {fmt(m['clean_cer'])} | {fmt(m['mean_distorted_cer'])} | "
            f"{fmt(m['mean_absolute_cer_delta'])} | {pct(m['mean_relative_cer_degradation'])} | "
            f"{interp[name]} |"
        )

    lines.append("")
    lines.append("## Table 3. H2 manual audit summary")
    lines.append("")
    lines.append("| subset | n | critical topology error | skeleton follows ink | border artifact | mean graph quality | interpretation |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    lines.append(
        f"| `HKR + Cyrillic` | {h2_hc['n']} | "
        f"{pct(h2_hc['critical_topology_error_rate'])} | "
        f"{pct(h2_hc['skeleton_follows_ink_rate'])} | n/a | "
        f"{fmt(h2_hc['mean_graph_quality_0_3'], 3)} | "
        "Usable diagnostic evidence for visible-stroke preservation. |"
    )

    school = h2_ds["school_notebooks_clean"]
    lines.append(
        f"| `school_notebooks_clean` | {school['n']} | "
        f"{pct(school['critical_topology_error_rate'])} | "
        f"{pct(school['skeleton_follows_ink_rate'])} | "
        f"{pct(school['border_artifact_rate'])} | "
        f"{fmt(school['mean_graph_quality_0_3'], 3)} | "
        "Preprocessing/binarization failure mode; report separately. |"
    )

    lines.append("")
    lines.append("## Table 4. H2 by dataset")
    lines.append("")
    lines.append("| dataset | n | usable | critical topology error | skeleton follows ink | border artifact | mean graph quality | failure stage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")

    for dataset, s in h2_ds.items():
        stages = ", ".join(f"{k}:{v}" for k, v in sorted(s["failure_stage_counts"].items()))
        lines.append(
            f"| `{dataset}` | {s['n']} | {pct(s['usable_rate'])} | "
            f"{pct(s['critical_topology_error_rate'])} | "
            f"{pct(s['skeleton_follows_ink_rate'])} | "
            f"{pct(s['border_artifact_rate'])} | "
            f"{fmt(s['mean_graph_quality_0_3'], 3)} | {stages} |"
        )

    lines.append("")
    lines.append("## Table 5. H3 diagnostic signal")
    lines.append("")
    lines.append("| analysis | best feature/set | group | n | metric | value | interpretation |")
    lines.append("|---|---|---|---:|---|---:|---|")

    single_corr = h3_neg["single_feature_global"]
    single_auc = h3_neg["single_feature_high_error"]
    quality_only = h3_neg["quality_only"]

    lines.append(
        f"| global single-feature correlation | `{single_corr['feature']}` | global | "
        f"{single_corr['n']} | Spearman r | {fmt(single_corr['spearman_r'])} | Weak. |"
    )
    lines.append(
        f"| single-feature high-error detection | `{single_auc['feature']}` | global | "
        f"{single_auc['n']} | ROC-AUC | {fmt(single_auc['roc_auc_direction_invariant'])} | Weak. |"
    )
    lines.append(
        f"| quality proxy only | `{quality_only['features']}` | `{quality_only['group']}` | "
        f"{quality_only['n']} | ROC-AUC | {fmt(quality_only['roc_auc'])} | Not useful. |"
    )
    lines.append(
        f"| multifeature structural descriptors | `{h3_pos['feature_set']}` | `{h3_pos['group']}` | "
        f"{h3_pos['n']} | ROC-AUC | {fmt(h3_pos['roc_auc'])} | Useful but localized. |"
    )

    lines.append("")
    lines.append("## Table 6. Final safe claim matrix")
    lines.append("")
    lines.append("| claim | status |")
    lines.append("|---|---|")
    lines.append("| Graph descriptors are useful diagnostic signals. | supported, with stratification caveat |")
    lines.append("| Graph-aware models outperform image-only HTR. | not supported |")
    lines.append("| Graph-aware models degrade less relatively under distortions. | partially supported |")
    lines.append("| Current graph pipeline preserves visible structure on all datasets. | not supported |")
    lines.append("| Current graph pipeline preserves visible structure on audited HKR/Cyrillic samples. | partially supported |")
    lines.append("| School-notebooks failures are graph-topology failures. | not supported; they are preprocessing artifacts |")
    lines.append("| Structural risk is graph quality. | not supported |")
    lines.append("| Structural risk can help find hard samples. | partially supported |")

    return "\n".join(lines) + "\n"


def make_limitations(h1: dict[str, Any], h2: dict[str, Any], h3: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Final limitations — v1")
    lines.append("")
    lines.append("## 1. Recognition performance limitation")
    lines.append("")
    lines.append(
        "The graph-aware recognition models do not outperform the image-only baseline in absolute CER. "
        "Their lower relative degradation under distortions should not be interpreted as better recognition performance. "
        "Because their clean CER is substantially worse, relative degradation alone is insufficient evidence of superior robustness as an HTR system."
    )
    lines.append("")

    lines.append("## 2. H1 limitation")
    lines.append("")
    lines.append(
        "H1 is supported only in a weak relative sense. The evidence shows reduced relative degradation, "
        "not improved absolute recognition. The robust interpretation is therefore limited to sensitivity analysis, "
        "not to a claim of a better recognizer."
    )
    lines.append("")

    lines.append("## 3. H2 audit limitation")
    lines.append("")
    lines.append(
        "The H2 manual audit used a diagnostic candidate pool selected across CER/risk quadrants. "
        "It is not a random population sample. Therefore its rates characterize failure modes and provide audit evidence, "
        "but they should not be reported as dataset-level graph-quality estimates."
    )
    lines.append("")

    lines.append("## 4. School-notebooks preprocessing limitation")
    lines.append("")
    lines.append(
        "The school-notebooks subset is dominated by crop/border/binarization artifacts. "
        "These failures occur before canonical graph construction. They should be reported as upstream preprocessing failures, "
        "not as direct failures of the graph abstraction. A simple border-suppression rule was tested and rejected because it was unreliable."
    )
    lines.append("")

    lines.append("## 5. H3 limitation")
    lines.append("")
    lines.append(
        "H3 is only partially supported. Global single-feature graph metrics are weak correlates of CER. "
        "Multifeature structural descriptors can detect high-error samples in some stratified subsets, but the effect is localized. "
        "The current structural risk score is a hard-sample indicator, not a direct graph-quality score."
    )
    lines.append("")

    lines.append("## 6. Graph quality limitation")
    lines.append("")
    lines.append(
        "Most graph-quality values used in automated experiments are proxy structural descriptors rather than gold graph accuracy measurements. "
        "A larger manually annotated gold subset would be required to estimate graph preservation quantitatively at population level."
    )
    lines.append("")

    lines.append("## 7. Generalization limitation")
    lines.append("")
    lines.append(
        "The current evidence is strongest for the audited HKR and Cyrillic subsets. "
        "It does not establish uniform behavior across all handwriting sources, crop styles, page backgrounds, or scanning conditions."
    )
    lines.append("")

    lines.append("## 8. Recommended wording")
    lines.append("")
    lines.append(
        "Use cautious language: partial support, diagnostic utility, failure triage, robustness analysis, and preprocessing limitation. "
        "Avoid language suggesting state-of-the-art recognition, full confirmation, or uniform graph quality across datasets."
    )

    return "\n".join(lines) + "\n"


def make_claims() -> str:
    lines: list[str] = []

    lines.append("# Final claims — v1")
    lines.append("")
    lines.append("## 1. Main claim")
    lines.append("")
    lines.append(
        "Canonical visible-stroke graph descriptors provide a useful intermediate diagnostic representation for offline handwritten text recognition. "
        "They support robustness analysis and failure triage, but the current graph-aware recognizers do not outperform a strong image-only baseline in absolute CER."
    )
    lines.append("")

    lines.append("## 2. Short thesis claim")
    lines.append("")
    lines.append(
        "The project demonstrates that visible-stroke structural descriptors can expose robustness and preprocessing failure modes in offline HTR, "
        "while also showing that naive graph fusion is not sufficient to improve recognition accuracy."
    )
    lines.append("")

    lines.append("## 3. H1 claim")
    lines.append("")
    lines.append(
        "Graph-aware variants show lower relative CER degradation under synthetic visual distortions, "
        "but because they have substantially worse clean and distorted absolute CER, H1 is only partially supported."
    )
    lines.append("")

    lines.append("## 4. H2 claim")
    lines.append("")
    lines.append(
        "Manual diagnostic audit suggests that the graph pipeline preserves visible stroke structure reasonably well on HKR and Cyrillic samples. "
        "School-notebooks samples are excluded from the graph-topology preservation claim because their failures are dominated by crop/border/binarization artifacts."
    )
    lines.append("")

    lines.append("## 5. H3 claim")
    lines.append("")
    lines.append(
        "Graph-derived structural descriptors provide useful but localized high-error detection in stratified subsets. "
        "However, individual global features are weak, and structural risk should not be interpreted as graph quality."
    )
    lines.append("")

    lines.append("## 6. Claims to avoid")
    lines.append("")
    lines.append("- Graph-aware HTR beats image-only HTR.")
    lines.append("- H1 is confirmed.")
    lines.append("- H2 is confirmed uniformly across all datasets.")
    lines.append("- School-notebooks failures prove the graph abstraction fails.")
    lines.append("- Structural risk is the same as graph quality.")
    lines.append("- Current graph features reconstruct real pen trajectory.")
    lines.append("")

    lines.append("## 7. Abstract-style paragraph")
    lines.append("")
    lines.append(
        "We investigate canonical visible-stroke graph descriptors as an intermediate representation for offline Russian-English handwritten text recognition. "
        "The representation is not intended to reconstruct real pen trajectories, but to capture reproducible visible stroke structure from static images. "
        "Across robustness, diagnostic, and manual audit experiments, graph-derived descriptors show partial value for relative robustness analysis and high-error sample triage. "
        "However, graph-aware recognition models do not outperform a strong image-only baseline in absolute CER, and manual audit reveals that failures in school-notebook samples are dominated by upstream crop and binarization artifacts. "
        "These results support graph descriptors as an interpretability and failure-analysis tool rather than as a standalone path to improved recognition accuracy."
    )

    return "\n".join(lines) + "\n"


def make_status() -> str:
    lines: list[str] = []

    lines.append("# Final experiment status — v1")
    lines.append("")
    lines.append("## Completed")
    lines.append("")
    lines.append("- Image-only baseline identified and evaluated.")
    lines.append("- Graph-vector model evaluated.")
    lines.append("- Gated local dist-only model evaluated.")
    lines.append("- Robustness manifests and summaries generated.")
    lines.append("- H1 robustness aggregation completed.")
    lines.append("- H3 graph diagnostic analysis completed.")
    lines.append("- H2 diagnostic audit candidate pool selected.")
    lines.append("- Browser-based H2 annotation completed for 100 samples.")
    lines.append("- School-notebooks binarization artifact identified.")
    lines.append("- Simple border suppression sanity check attempted and rejected.")
    lines.append("- Consolidated H1/H2/H3 evidence report generated.")
    lines.append("")

    lines.append("## Do not continue")
    lines.append("")
    lines.append("- Do not add new HTR architectures.")
    lines.append("- Do not tune graph fusion to chase CER.")
    lines.append("- Do not present school-notebooks as pure graph topology failure.")
    lines.append("- Do not integrate rejected border suppression v1.")
    lines.append("")

    lines.append("## Remaining writing tasks")
    lines.append("")
    lines.append("- Write methods section for graph extraction and audit.")
    lines.append("- Write results section using final tables.")
    lines.append("- Write limitations section explicitly.")
    lines.append("- Prepare figures from existing outputs.")
    lines.append("- Freeze experimental claims.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1_json", required=True)
    parser.add_argument("--h2_json", required=True)
    parser.add_argument("--h3_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    h1 = load(args.h1_json)
    h2 = load(args.h2_json)
    h3 = load(args.h3_json)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write(out_dir / "final_results_tables.md", make_results_tables(h1, h2, h3))
    write(out_dir / "final_limitations.md", make_limitations(h1, h2, h3))
    write(out_dir / "final_claims.md", make_claims())
    write(out_dir / "final_experiment_status.md", make_status())


if __name__ == "__main__":
    main()