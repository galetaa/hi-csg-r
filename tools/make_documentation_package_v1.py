from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_text_if_exists(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


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


def make_readme() -> str:
    return """# HI-CSG-R documentation package — v1

This directory contains the finalized documentation layer for the current experimental state.

## Contents

| file | purpose |
|---|---|
| `01_research_claims.md` | final safe/unsafe claims |
| `02_methods_summary.md` | method summary for graph extraction, HTR, robustness, diagnostics, and audit |
| `03_results_narrative.md` | result narrative for H1/H2/H3 |
| `04_limitations_and_threats.md` | limitations and validity threats |
| `05_figures_and_tables_checklist.md` | final figure/table checklist |
| `06_reproducibility_inventory.md` | outputs, scripts, and reproducibility inventory |
| `07_improvement_roadmap.md` | post-documentation improvement plan |
| `08_defense_qa.md` | likely reviewer/defense questions and safe answers |

## Status

The current evidence supports a cautious framing:

```text
Overall: mixed / partial support
H1: partial support only
H2: partial support with school-notebooks preprocessing exception
H3: partial support
The project should be presented as an interpretable graph-based diagnostic framework for offline handwritten text recognition, not as a state-of-the-art HTR recognizer.
"""

def make_claims(consolidated: dict[str, Any]) -> str:
    safe = consolidated.get("safe_claims", [])
    unsafe = consolidated.get("unsafe_claims", [])

    lines: list[str] = []
    lines.append("# 01 — Research claims")
    lines.append("")
    lines.append("## Main claim")
    lines.append("")
    lines.append(
        "Canonical visible-stroke graph descriptors provide a useful intermediate diagnostic representation "
        "for offline handwritten text recognition. They support robustness analysis, high-error sample triage, "
        "and manual failure analysis. Current graph-aware recognition models do not outperform a strong image-only "
        "baseline in absolute CER."
    )
    lines.append("")
    lines.append("## Safe claims")
    lines.append("")

    for claim in safe:
        lines.append(f"- {claim}")

    lines.append("")
    lines.append("## Unsafe claims")
    lines.append("")

    for claim in unsafe:
        lines.append(f"- {claim}")

    lines.append("")
    lines.append("## Exact recommended wording")
    lines.append("")
    lines.append(
        "> Canonical visible-stroke graph descriptors provide a reproducible structural representation for offline "
        "handwriting images. In the current experiments, these descriptors are useful for robustness analysis and "
        "failure triage, but graph-aware recognition models remain worse than the image-only baseline in absolute CER. "
        "Manual audit further shows that some severe graph failures arise from upstream crop and binarization artifacts, "
        "especially in school-notebook samples."
    )
    lines.append("")
    lines.append("## One-sentence contribution")
    lines.append("")
    lines.append(
        "The contribution is an interpretable visible-stroke graph diagnostic framework for offline handwriting, "
        "not a superior recognizer."
    )

    return "\n".join(lines) + "\n"

def make_methods_summary() -> str:
    return """# 02 — Methods summary
    
    1. Representation
    
    The project uses canonical visible-stroke graphs as an intermediate representation for offline handwriting images.
    
    Important distinction:
    
    The graph is not a reconstruction of real pen trajectory.
    It is a canonical graph of visible stroke structure extracted from a static image.
    
    The intended role is interpretability, structural diagnostics, robustness analysis, and failure triage.
    
    2. Recognition models
    
    The experiments compare:
    
    image-only CRNN/CTC baseline;
    graph-vector fusion model;
    gated local graph/dist-map fusion model.
    
    The image-only model is the main absolute-CER baseline. Graph-aware models are evaluated as structural/robustness variants, not as guaranteed accuracy improvements.
    
    3. Robustness evaluation
    
    Robustness is evaluated by comparing clean CER with CER under visual/structural distortions.
    
    Key interpretation rule:
    
    Lower relative degradation is not enough to claim better HTR if clean and distorted absolute CER are worse.
    
    Therefore H1 is evaluated in both absolute and relative terms.
    
    4. Graph diagnostics
    
    H3 evaluates whether graph-derived structural descriptors help identify high-error samples.
    
    Two levels are distinguished:
    
    global single-feature correlations;
    stratified multifeature high-error detection.
    
    The second is more meaningful in the current results.
    
    5. Manual H2 audit
    
    The H2 audit uses a diagnostic candidate pool selected across CER/risk quadrants:
    
    A: high CER + high structural risk
    B: high CER + low structural risk
    C: low CER + high structural risk
    D: low CER + low structural risk
    
    This is not a random population sample. It is used for failure-mode characterization and structural sanity checking.
    
    6. Failure staging
    
    Manual audit distinguishes:
    
    ok
    input_crop
    binarization
    skeletonization
    graph_topology
    illegible
    
    This prevents upstream preprocessing artifacts from being misreported as graph-topology failures.
    """

def make_results_narrative(h1: dict[str, Any], h2: dict[str, Any], h3: dict[str, Any]) -> str:
    h1_models = h1.get("models", {})
    h2_hc = h2.get("hkr_plus_cyrillic", {})
    h2_ds = h2.get("by_dataset", {})
    school = h2_ds.get("school_notebooks_clean", {})
    h3_pos = h3.get("main_positive_result", {})

    image = h1_models.get("image_only", {})
    graph = h1_models.get("graph_vector_v2", {})
    gated = h1_models.get("gated_v2_dist", {})

    lines: list[str] = []
    lines.append("# 03 — Results narrative")
    lines.append("")
    lines.append("## H1 — Robustness")
    lines.append("")
    lines.append(
        "The strong form of H1 is not confirmed. The image-only baseline remains the best absolute recognizer. "
        "Graph-aware variants show lower relative CER degradation under distortions, but they start from much worse "
        "clean CER and also have worse mean distorted CER."
    )
    lines.append("")
    lines.append("| model | clean CER | mean distorted CER | relative degradation |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| image-only | {fmt(image.get('clean_cer'))} | {fmt(image.get('mean_distorted_cer'))} | {pct(image.get('mean_relative_cer_degradation'))} |")
    lines.append(f"| graph-vector | {fmt(graph.get('clean_cer'))} | {fmt(graph.get('mean_distorted_cer'))} | {pct(graph.get('mean_relative_cer_degradation'))} |")
    lines.append(f"| gated dist | {fmt(gated.get('clean_cer'))} | {fmt(gated.get('mean_distorted_cer'))} | {pct(gated.get('mean_relative_cer_degradation'))} |")
    lines.append("")
    lines.append(
        "Interpretation: H1 receives weak/relative support only. It supports robustness analysis, not a claim of improved HTR."
    )
    lines.append("")

    lines.append("## H2 — Structural preservation")
    lines.append("")
    lines.append(
        "H2 is partially supported with an important preprocessing exception. In the diagnostic audit subset, HKR and Cyrillic "
        "samples generally preserve visible stroke structure. School-notebooks samples are dominated by crop/border/binarization artifacts."
    )
    lines.append("")
    lines.append("| subset | n | critical topology error | skeleton follows ink | mean quality |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| HKR + Cyrillic | {h2_hc.get('n', 'n/a')} | "
        f"{pct(h2_hc.get('critical_topology_error_rate'))} | "
        f"{pct(h2_hc.get('skeleton_follows_ink_rate'))} | "
        f"{fmt(h2_hc.get('mean_graph_quality_0_3'), 3)} |"
    )
    lines.append(
        f"| school-notebooks | {school.get('n', 'n/a')} | "
        f"{pct(school.get('critical_topology_error_rate'))} | "
        f"{pct(school.get('skeleton_follows_ink_rate'))} | "
        f"{fmt(school.get('mean_graph_quality_0_3'), 3)} |"
    )
    lines.append("")
    lines.append(
        "Interpretation: school-notebooks should not be aggregated into a pure graph-topology failure claim. "
        "They expose an upstream preprocessing limitation."
    )
    lines.append("")

    lines.append("## H3 — Graph diagnostics")
    lines.append("")
    lines.append(
        "H3 is partially supported. Global single-feature correlations are weak, but multifeature structural descriptors "
        "provide useful high-error detection in some stratified subsets."
    )
    lines.append("")
    lines.append("| best signal | value |")
    lines.append("|---|---:|")
    lines.append(f"| feature set | `{h3_pos.get('feature_set', 'n/a')}` |")
    lines.append(f"| group | `{h3_pos.get('group', 'n/a')}` |")
    lines.append(f"| n | {h3_pos.get('n', 'n/a')} |")
    lines.append(f"| ROC-AUC | {fmt(h3_pos.get('roc_auc'))} |")
    lines.append(f"| PR-AUC | {fmt(h3_pos.get('pr_auc'))} |")
    lines.append(f"| PR-AUC lift | {fmt(h3_pos.get('pr_auc_lift_over_base_rate'))} |")
    lines.append("")
    lines.append(
        "Interpretation: graph-derived structural descriptors can help identify hard samples, but structural risk is not graph quality."
    )

    return "\n".join(lines) + "\n"

def make_limitations() -> str:
    return """# 04 — Limitations and threats to validity
    
    1. Recognition performance
    
    Graph-aware models do not outperform the image-only baseline in absolute CER. Any claim of improved recognition accuracy would be unsupported.
    
    2. Relative robustness
    
    Lower relative degradation under distortions is meaningful only as a sensitivity signal. Because graph-aware models start with worse clean CER, relative degradation cannot be interpreted as superior HTR robustness by itself.
    
    3. Diagnostic audit sampling
    
    The H2 audit subset is deliberately selected across CER/risk quadrants. It is not a random sample and should not be used to estimate population-level graph quality.
    
    4. School-notebooks preprocessing
    
    School-notebooks failures are dominated by crop/border/binarization artifacts. These failures occur before graph construction. They should be reported separately from graph-topology failures.
    
    5. Structural risk
    
    The structural risk score is useful as a hard-sample indicator, but manual audit shows it is not equivalent to visible graph quality.
    
    6. Gold graph annotation
    
    The project does not yet include a large independent gold graph annotation set. Therefore H2 is supported through diagnostic manual audit rather than full population-level graph accuracy estimation.
    
    7. Dataset specificity
    
    Evidence is strongest for the audited HKR and Cyrillic samples. Generalization to other handwriting sources, scanning conditions, crop procedures, and background types remains limited.
    
    8. Model search
    
    Architecture experiments should be frozen. Additional model chasing risks obscuring the main methodological contribution.
    """

def make_figures_checklist() -> str:
    return """# 05 — Figures and tables checklist
    
    Required figures
    Figure 1 — Pipeline diagram
    
    Show:
    
    image → foreground mask → skeleton → canonical graph descriptors → HTR / diagnostics
    
    Caption must say:
    
    The graph is a canonical visible-stroke structure, not a real pen trajectory reconstruction.
    Figure 2 — H1 robustness comparison
    
    Use a bar/table plot comparing:
    
    clean CER
    mean distorted CER
    relative degradation
    
    Models:
    
    image-only
    graph-vector
    gated dist
    
    Main message:
    
    graph-aware models have lower relative degradation but worse absolute CER.
    Figure 3 — H2 good examples
    
    Show HKR/Cyrillic examples with:
    
    original
    binary
    skeleton
    overlay
    
    Main message:
    
    visible stroke structure is mostly preserved in audited HKR/Cyrillic samples.
    Figure 4 — H2 school-notebooks failure
    
    Show school-notebooks crop/border artifact:
    
    original
    binary with border artifact
    bad skeleton
    overlay
    
    Main message:
    
    failure is upstream crop/binarization, not pure graph topology.
    Figure 5 — H3 diagnostic signal
    
    Show the best multifeature high-error detection result:
    
    structural_core
    hkr_words|word
    ROC-AUC / PR-AUC / top20 precision
    
    Main message:
    
    localized diagnostic value, not global graph-quality scoring.
    Figure 6 — Failure taxonomy
    
    Show staged failure types:
    
    input_crop
    binarization
    skeletonization
    graph_topology
    recognition difficulty
    Required tables
    Hypothesis verdict table.
    H1 robustness table.
    H2 audit summary table.
    H3 diagnostic result table.
    Safe/unsafe claims table.
    Figures to avoid
    
    Avoid figures that imply:
    
    graph-aware model is the best recognizer;
    structural risk is graph quality;
    school-notebooks failures are pure graph-topology failures;
    offline graph equals real pen trajectory.
    """

def make_repro_inventory(paths: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append("# 06 — Reproducibility inventory")
    lines.append("")
    lines.append("## Primary evidence files")
    lines.append("")
    for name, path in paths.items():
        lines.append(f"- {name}: {path}")
        lines.append("")
        lines.append("## Key generated reports")
        lines.append("")
        lines.append("- outputs/robustness_v1/h1_robustness_report_v1.md")
        lines.append("- outputs/h3_graph_quality_v1/h3_final_diagnostic_report_v1.md")
        lines.append("- outputs/h2_gold_audit_v1/h2_final_report_v1.md")
        lines.append("- outputs/final_evidence_v1/hi_csg_r_consolidated_evidence_report_v1.md")
        lines.append("- outputs/final_evidence_v1/text_assets/")
        lines.append("")
        lines.append("## Key scripts")
        lines.append("")
        lines.append("- tools/aggregate_robustness_v1.py")
        lines.append("- tools/make_h1_robustness_report_v1.py")
        lines.append("- tools/analyze_graph_quality_vs_cer_v1.py")
        lines.append("- tools/analyze_graph_quality_vs_cer_v2.py")
        lines.append("- tools/make_h3_final_report_v1.py")
        lines.append("- tools/select_h2_gold_audit_candidates_v1.py")
        lines.append("- tools/make_h2_browser_audit_tool_v1.py")
        lines.append("- tools/summarize_h2_manual_audit_v2.py")
        lines.append("- tools/make_h2_final_report_v1.py")
        lines.append("- tools/make_consolidated_evidence_report_v1.py")
        lines.append("- tools/make_final_text_assets_v1.py")
        lines.append("- tools/make_documentation_package_v1.py")
        lines.append("")
        lines.append("## Reproducibility rule")
        lines.append("")
        lines.append(
        "Do not overwrite final evidence files without committing a new versioned report. "
        "Use versioned output directories for new improvement experiments."
        )
    return "\n".join(lines) + "\n"

def make_roadmap() -> str:
    return """# 07 — Improvement roadmap
    
    Principle
    
    Do not return to architecture chasing. Improvements should target failure modes revealed by evidence.
    
    Priority 1 — School-notebooks preprocessing
    
    Problem:
    
    crop/border/background artifacts are binarized as foreground
    
    Goal:
    
    improve foreground extraction before skeletonization
    
    Allowed experiments:
    
    crop-border masking before binarization;
    background normalization;
    border-line detection;
    notebook-specific foreground cleanup;
    regenerated skeleton/graph quality audit on the same 23 samples.
    
    Do not retrain HTR for this step.
    
    Priority 2 — H2 gold subset expansion
    
    Goal:
    
    move from diagnostic audit to a more reliable gold subset
    
    Possible work:
    
    sample random HKR/Cyrillic cases;
    manually annotate graph quality;
    estimate population-level graph preservation with confidence intervals.
    Priority 3 — Better graph-quality score
    
    Problem:
    
    structural risk is not graph quality
    
    Goal:
    
    train/calibrate a graph-quality predictor using manual labels
    
    Use manual H2 labels as supervision.
    
    Priority 4 — Robustness follow-up
    
    Only after preprocessing is fixed:
    
    regenerate clean/distorted graph features;
    rerun H1 aggregation;
    check whether graph-aware robustness remains.
    Priority 5 — Model improvement
    
    Only if previous steps succeed:
    
    freeze image encoder baseline;
    test graph input as auxiliary diagnostic head;
    avoid claiming accuracy improvement unless absolute CER improves.
    """

def make_defense_qa() -> str:
    return """# 08 — Defense / reviewer Q&A
    
    Q1. Did graph-aware recognition beat the image-only baseline?
    
    No. The image-only baseline has better absolute CER. Graph-aware variants show lower relative degradation under distortions, but they are worse in clean and distorted absolute CER.
    
    Q2. Is H1 confirmed?
    
    Only weakly. The strong form is not confirmed. The supported claim is lower relative degradation, not better HTR performance.
    
    Q3. Is the graph a pen-trajectory reconstruction?
    
    No. It is a canonical visible-stroke graph extracted from offline images. It represents visible stroke structure, not true writing dynamics.
    
    Q4. Does high structural risk mean bad graph quality?
    
    No. Manual audit shows that structural risk often marks difficult samples rather than visible skeleton failure. It is a hard-sample indicator, not a graph-quality score.
    
    Q5. Why are school-notebooks so bad?
    
    Because crop/border/background artifacts are binarized as foreground. This corrupts skeletons and graphs upstream of graph construction.
    
    Q6. Does school-notebooks invalidate the graph representation?
    
    No. It reveals a preprocessing limitation. HKR/Cyrillic audited samples show much better structural preservation.
    
    Q7. Why not fix school-notebooks immediately?
    
    A simple border-suppression rule was tested and rejected because it either did nothing or removed handwriting. A proper fix requires dataset-specific preprocessing, not quick graph tuning.
    
    Q8. What is the main contribution?
    
    The main contribution is a reproducible visible-stroke graph diagnostic framework for offline handwriting recognition, with evidence for robustness analysis, failure triage, and preprocessing failure detection.
    
    Q9. What remains future work?
    
    Better preprocessing for notebook data, larger gold graph annotation, calibrated graph-quality prediction, and only then renewed graph-aware recognition experiments.
    """

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1_json", required=True)
    parser.add_argument("--h2_json", required=True)
    parser.add_argument("--h3_json", required=True)
    parser.add_argument("--consolidated_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    h1 = load_json(args.h1_json)
    h2 = load_json(args.h2_json)
    h3 = load_json(args.h3_json)
    consolidated = load_json(args.consolidated_json)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_paths = {
        "h1_json": args.h1_json,
        "h2_json": args.h2_json,
        "h3_json": args.h3_json,
        "consolidated_json": args.consolidated_json,
    }

    write(out_dir / "00_README.md", make_readme())
    write(out_dir / "01_research_claims.md", make_claims(consolidated))
    write(out_dir / "02_methods_summary.md", make_methods_summary())
    write(out_dir / "03_results_narrative.md", make_results_narrative(h1, h2, h3))
    write(out_dir / "04_limitations_and_threats.md", make_limitations())
    write(out_dir / "05_figures_and_tables_checklist.md", make_figures_checklist())
    write(out_dir / "06_reproducibility_inventory.md", make_repro_inventory(evidence_paths))
    write(out_dir / "07_improvement_roadmap.md", make_roadmap())
    write(out_dir / "08_defense_qa.md", make_defense_qa())

if __name__ == "__main__":
    main()