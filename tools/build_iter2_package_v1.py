from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("outputs/iter2_package_v1")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def build_result() -> dict[str, Any]:
    dose = read_json("outputs/htr_graph_v1/line_aug_dose_response_context_v1/summary.json")
    selective = read_json("outputs/htr_graph_v1/selective_iter2_confidence_v1/selective_summary.json")
    operating = read_json("outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_points.json")
    gold = read_json("outputs/iter2_structural_gold_v1/annotation_summary.json")
    graph_fusion = read_json("outputs/htr_graph_v1/graph_fusion_iter2_context10k_v1/summary.json")

    return {
        "title": "Iteration 2 package: data-centric HTR improvement with structural diagnostics",
        "accepted_preprocessing": {
            "school_foreground_method": "rectangular_whitebalance_lineaware_postpoly_v3",
            "quality_gate": {
                "usable": 0.958,
                "skeleton_follows_ink": 1.0,
                "neighbor_text_removed": 1.0,
                "ink_loss": 0.050,
                "line_residual": 0.075,
            },
        },
        "school_quality_manifests": {
            "train": {"clean_core": 8783, "hard_real": 1217, "invalid_or_review": 0, "n": 10000},
            "val": {"clean_core": 1796, "hard_real": 204, "invalid_or_review": 0, "n": 2000},
            "test": {"clean_core": 1764, "hard_real": 236, "invalid_or_review": 0, "n": 2000},
        },
        "full_natural_line_corpus": {
            "line_groups": 60074,
            "covered_word_instances": 262301,
            "total_word_instances": 283670,
            "coverage": 0.9247,
            "mean_words_per_group": 4.37,
            "groups_4plus_words": 38125,
            "geometry_outliers_excluded": 101,
            "usable_line_groups": 59973,
        },
        "line_augmentation": dose["runs"],
        "line_augmentation_paired_bootstrap": dose["paired_bootstrap"],
        "canonical_checkpoints": {
            "cer_canonical": "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1",
            "balanced_canonical": "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_5k_context_v1",
        },
        "selective_prediction": {
            "canonical_htr_model": "plus_10k_context",
            "canonical_risk_model": "confidence_graph",
            "risk_quality": {
                model: {
                    method: {
                        "auc_all": values["risk_auc_exact_error_all"],
                        "auc_school": values["risk_auc_exact_error_school"],
                        "ece_all": values["ece_all"],
                        "ece_school": values["ece_school"],
                    }
                    for method, values in model_values["risk_methods"].items()
                }
                for model, model_values in selective["models"].items()
            },
            "operating_points": operating["operating_points"],
            "limitations": [
                "global thresholds are not coverage-fair across datasets/token types",
                "numeric/mixed and HKR/Cyrillic are rejected more often",
                "short 1-3 samples are accepted more readily, so short-token errors are overconfident ambiguity errors",
            ],
        },
        "structural_gold": {
            "n": gold["overall"]["n"],
            "completed": gold["overall"]["completed"],
            "rates": gold["overall"]["rates"],
            "htr_error_explained_by_structure": gold["overall"]["htr_error_explained_by_structure"],
            "all_acceptance_passed": gold["overall"]["all_acceptance_passed"],
        },
        "graph_fusion_pilot": graph_fusion,
        "main_conclusion": [
            "corrected foreground extraction",
            "natural-line context augmentation gives statistically supported CER gains",
            "confidence-aware selective prediction works strongly",
            "structural graph features are useful for diagnostics and confidence calibration",
            "simple graph-fusion provides School-specific gains, especially on hard_real, but naive global fusion is not stable across the mixed dataset",
        ],
        "main_limitations": [
            "training comparison is still mostly single-seed",
            "contextual line crops are not clean isolated line crops",
            "structural gold is a diagnostic usability check, not a pixel-level topology benchmark",
            "naive global graph fusion hurts non-School datasets and should not be treated as the universal canonical recognizer",
            "global selective thresholds are not group-fair",
        ],
    }


def build_md(result: dict[str, Any]) -> str:
    runs = result["line_augmentation"]
    bootstrap = result["line_augmentation_paired_bootstrap"]
    selective = result["selective_prediction"]
    gold = result["structural_gold"]
    graph = result["graph_fusion_pilot"]
    graph_vs_image = graph["paired_vs_image10k"]
    graph_models = graph["models"]

    lines = [
        f"# {result['title']}",
        "",
        "## 1. Summary",
        "",
        "Iteration 2 shifted the project from graph-only HTR experiments to a data-centric and structurally controlled HTR pipeline.",
        "",
        "Main result:",
        "- corrected School foreground extraction;",
        "- built natural-line contextual augmentation;",
        "- obtained statistically supported image-only HTR gains;",
        "- validated confidence-aware selective prediction;",
        "- confirmed that the current structural extraction is not the main bottleneck on the diagnostic gold subset;",
        "- tested a single graph-fusion pilot and found targeted School benefit but mixed-dataset instability.",
        "",
        "## 2. Accepted preprocessing",
        "",
        "School foreground method:",
        "- rectangular raw COCO crop;",
        "- whitebalance + line-aware foreground;",
        "- post-binarization polygon filtering;",
        f"- method: `{result['accepted_preprocessing']['school_foreground_method']}`.",
        "",
        "Quality gate:",
        "- usable: 95.8%",
        "- skeleton_follows_ink: 100%",
        "- neighbor_text_removed: 100%",
        "- ink_loss: 5.0%",
        "- line_residual: 7.5%",
        "",
        "## 3. School quality manifests",
        "",
        "School lineaware_v3 quality buckets:",
        "",
    ]

    for split in ["train", "val", "test"]:
        item = result["school_quality_manifests"][split]
        lines.extend([
            f"{split}:",
            f"- clean_core: {item['clean_core']} / {item['n']}",
            f"- hard_real: {item['hard_real']} / {item['n']}",
            f"- invalid_or_review: {item['invalid_or_review']}",
            "",
        ])

    corpus = result["full_natural_line_corpus"]
    lines.extend([
        "## 4. Full natural-line corpus",
        "",
        "Full School COCO line candidates:",
        f"- line groups: {corpus['line_groups']}",
        f"- covered word instances: {corpus['covered_word_instances']} / {corpus['total_word_instances']}",
        f"- coverage: {corpus['coverage'] * 100:.2f}%",
        f"- mean words/group: {corpus['mean_words_per_group']:.2f}",
        f"- groups with 4+ words: {corpus['groups_4plus_words']}",
        f"- geometry outliers excluded: {corpus['geometry_outliers_excluded']}",
        "",
        "Full line corpus v1:",
        f"- usable line groups: {corpus['usable_line_groups']}",
        "",
        "## 5. Line augmentation",
        "",
        "Image-only word-level test results:",
        "",
        "| model | train_n | line_n | overall CER | HKR CER | Cyrillic CER | School CER | School WER | School exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    run_order = [
        ("baseline", "baseline"),
        ("+2k context", "+2k lines"),
        ("+5k context", "+5k lines"),
        ("+10k context", "+10k lines"),
    ]
    for label, key in run_order:
        row = runs[key]
        by_ds = row["by_dataset"]
        lines.append(
            f"| {label} | {row['train_n']} | {row.get('line_train_n', 0)} | "
            f"{fmt(row['overall']['cer'])} | "
            f"{fmt(by_ds['hkr_words']['cer'])} | "
            f"{fmt(by_ds['cyrillic_handwriting']['cer'])} | "
            f"{fmt(by_ds['school_notebooks_clean']['cer'])} | "
            f"{fmt(by_ds['school_notebooks_clean']['wer'])} | "
            f"{fmt(by_ds['school_notebooks_clean']['exact'])} |"
        )

    lines.extend([
        "",
        "Paired bootstrap:",
        f"- +5k overall ΔCER: {fmt(bootstrap['+5k lines']['overall']['mean_delta_cer'])}, CI [{fmt(bootstrap['+5k lines']['overall']['ci95_low'])}, {fmt(bootstrap['+5k lines']['overall']['ci95_high'])}]",
        f"- +5k School ΔCER: {fmt(bootstrap['+5k lines']['school_notebooks_clean']['mean_delta_cer'])}, CI [{fmt(bootstrap['+5k lines']['school_notebooks_clean']['ci95_low'])}, {fmt(bootstrap['+5k lines']['school_notebooks_clean']['ci95_high'])}]",
        f"- +10k overall ΔCER: {fmt(bootstrap['+10k lines']['overall']['mean_delta_cer'])}, CI [{fmt(bootstrap['+10k lines']['overall']['ci95_low'])}, {fmt(bootstrap['+10k lines']['overall']['ci95_high'])}]",
        f"- +10k School ΔCER: {fmt(bootstrap['+10k lines']['school_notebooks_clean']['mean_delta_cer'])}, CI [{fmt(bootstrap['+10k lines']['school_notebooks_clean']['ci95_low'])}, {fmt(bootstrap['+10k lines']['school_notebooks_clean']['ci95_high'])}]",
        "",
        "Canonical checkpoints:",
        "- CER canonical: +10k context",
        "- balanced canonical: +5k context",
        "",
        "## 6. Selective prediction",
        "",
        "Canonical selective model:",
        "- HTR model: +10k context",
        "- risk model: confidence_graph",
        "",
        "Risk quality:",
        "- feature_only: AUC around 0.60",
        "- model_confidence: AUC around 0.80",
        "- confidence_graph: AUC around 0.81 overall and 0.83 on School",
        "",
        "School operating points for +10k confidence_graph:",
    ])

    for point in ["strict", "balanced", "broad"]:
        row = selective["operating_points"][point]["test"]["school"]
        lines.append(
            f"- {point}: coverage {fmt(row['coverage'])}, CER {fmt(row['cer'])}, exact {fmt(row['exact'])}"
        )
    full_school = runs["+10k lines"]["by_dataset"]["school_notebooks_clean"]
    lines.append(
        f"- full: coverage 1.0000, CER {fmt(full_school['cer'])}, exact {fmt(full_school['exact'])}"
    )

    lines.extend([
        "",
        "Limitation:",
        "- global thresholds are not coverage-fair across datasets/token types;",
        "- numeric/mixed and HKR/Cyrillic are rejected more often;",
        "- short 1–3 samples are accepted more readily, so short-token errors are overconfident ambiguity errors.",
        "",
        "## 7. Structural gold diagnostic",
        "",
        "Gold subset:",
        f"- total: {gold['n']}",
        f"- structural_usable: {gold['rates']['structural_usable'] * 100:.0f}%",
        f"- foreground_ok: {gold['rates']['foreground_ok'] * 100:.0f}%",
        f"- skeleton_ok: {gold['rates']['skeleton_ok'] * 100:.0f}%",
        f"- graph_ok: {gold['rates']['graph_ok'] * 100:.0f}%",
        "",
        "All annotated HTR errors were marked as not explained by structural extraction defects.",
        "",
        "Conclusion:",
        "- foreground/skeleton/graph extraction is usable on the diagnostic subset;",
        "- remaining errors are primarily model/ambiguity/token-level rather than extraction failures.",
        "",
        "## 8. Graph-fusion pilot",
        "",
        "The graph-fusion pilot produced a mixed result. Compared with the image-only +10k model, graph-fusion significantly improved School CER, with the strongest gain on hard_real samples, but significantly degraded HKR and Cyrillic CER. Overall CER was statistically neutral/slightly negative. Zero-graph ablation substantially reduced performance, indicating that the graph branch was actively used.",
        "",
        "Compared to image-only +10k seed42:",
        f"- overall ΔCER: {fmt(graph_vs_image['bootstrap']['overall']['mean_delta_cer_per_sample'])}, CI [{fmt(graph_vs_image['bootstrap']['overall']['ci95_low'])}, {fmt(graph_vs_image['bootstrap']['overall']['ci95_high'])}]",
        f"- School ΔCER: {fmt(graph_vs_image['bootstrap']['school_notebooks_clean']['mean_delta_cer_per_sample'])}, CI [{fmt(graph_vs_image['bootstrap']['school_notebooks_clean']['ci95_low'])}, {fmt(graph_vs_image['bootstrap']['school_notebooks_clean']['ci95_high'])}]",
        f"- HKR ΔCER: {fmt(graph_vs_image['bootstrap']['hkr_words']['mean_delta_cer_per_sample'])}, CI [{fmt(graph_vs_image['bootstrap']['hkr_words']['ci95_low'])}, {fmt(graph_vs_image['bootstrap']['hkr_words']['ci95_high'])}]",
        f"- Cyrillic ΔCER: {fmt(graph_vs_image['bootstrap']['cyrillic_handwriting']['mean_delta_cer_per_sample'])}, CI [{fmt(graph_vs_image['bootstrap']['cyrillic_handwriting']['ci95_low'])}, {fmt(graph_vs_image['bootstrap']['cyrillic_handwriting']['ci95_high'])}]",
        "",
        "School hard_real:",
        f"- CER {fmt(graph_vs_image['by_school_quality_bucket']['hard_real']['baseline']['cer'])} -> {fmt(graph_vs_image['by_school_quality_bucket']['hard_real']['candidate']['cer'])}",
        f"- exact {fmt(graph_vs_image['by_school_quality_bucket']['hard_real']['baseline']['exact'])} -> {fmt(graph_vs_image['by_school_quality_bucket']['hard_real']['candidate']['exact'])}",
        "",
        "Zero-graph ablation:",
        f"- normal graph-fusion CER: {fmt(graph_models['graph_fusion']['cer'])}",
        f"- zero-graph CER: {fmt(graph_models['zero_graph']['cer'])}",
        "",
        "Interpretation:",
        "- graph features contain recognition-relevant structural signal for School;",
        "- naive ungated late fusion is not safe as a universal mixed-dataset recognizer;",
        "- a future controlled variant would be dataset-gated graph fusion, but it is not part of Iteration 2.",
        "",
        "## 9. Main conclusion",
        "",
        "Iteration 2 demonstrates a data-centric HTR improvement:",
        "- corrected foreground extraction;",
        "- natural-line context augmentation gives statistically supported CER gains;",
        "- confidence-aware selective prediction works strongly;",
        "- structural graph features are useful for diagnostics and confidence calibration;",
        "- graph fusion provides School-specific gains, especially on hard_real, but naive global fusion harms non-School datasets.",
        "",
        "## 10. Main limitations",
        "",
    ])
    for item in result["main_limitations"]:
        lines.append(f"- {item};")
    return "\n".join(lines) + "\n"


def build_artifact_index() -> str:
    sections = {
        "Preprocessing and Quality": [
            "src/preprocessing/school_rectangular_v2.py",
            "data/experiments/htr_graph_v1/graph_ready/tri10k_mixed_school_lineaware_v3/summary.json",
            "data/experiments/iter2_quality_manifests/school_notebooks_lineaware_v3/summary.json",
            "outputs/iter2_data_audit/school_notebooks_v1/rendered_line_sanity_v1/validation_result.json",
        ],
        "Line Groups and Corpus": [
            "data/experiments/iter2_line_groups/school_notebooks_full_coco/summary.json",
            "data/experiments/iter2_line_groups/school_notebooks_full_coco_geometry_v1/summary.json",
            "data/experiments/iter2_line_corpus/school_notebooks_full_line_v1/summary.json",
            "data/experiments/iter2_line_corpus/school_notebooks_full_line_v1_sampled_5k_rendered/render_summary.json",
        ],
        "HTR Runs and Eval": [
            "outputs/htr_graph_v1/tri10k_image_only_v1",
            "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_5k_context_v1",
            "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1",
            "outputs/htr_graph_v1/line_aug_dose_response_context_v1/summary.md",
            "outputs/htr_graph_v1/line_aug_dose_response_context_v1/paired_5k_vs_10k.md",
        ],
        "Graph Fusion Pilot": [
            "data/experiments/htr_graph_v1/graph_fusion_ready/tri10k_mixed_plus_school_lines_10k_context_v1/summary.json",
            "outputs/htr_graph_v1/tri10k_graph_fusion_plus_school_lines_10k_context_v1/best.pt",
            "outputs/htr_graph_v1/eval_tri10k_graph_fusion_plus_school_lines_10k_context_v1_test_final/summary.json",
            "outputs/htr_graph_v1/eval_tri10k_graph_fusion_plus_school_lines_10k_context_v1_test_final_zero_graph/summary.json",
            "outputs/htr_graph_v1/graph_fusion_iter2_context10k_v1/summary.md",
            "outputs/htr_graph_v1/graph_fusion_iter2_context10k_v1/paired_vs_image10k.md",
            "outputs/htr_graph_v1/graph_fusion_iter2_context10k_v1/paired_vs_baseline.md",
            "outputs/htr_graph_v1/graph_fusion_iter2_context10k_v1/result_card.md",
        ],
        "Selective Prediction": [
            "outputs/htr_graph_v1/selective_iter2_lineaug_v1/selective_summary.md",
            "outputs/htr_graph_v1/selective_iter2_confidence_v1/selective_summary.md",
            "outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_points.md",
            "outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_point_strata.md",
            "outputs/htr_graph_v1/selective_iter2_confidence_v1/accepted_errors_high_confidence.jsonl",
            "outputs/htr_graph_v1/selective_iter2_confidence_v1/rejected_correct_low_confidence.jsonl",
        ],
        "Structural Gold": [
            "outputs/iter2_structural_gold_v1/sample_plan.json",
            "outputs/iter2_structural_gold_v1/sample_manifest.jsonl",
            "outputs/iter2_structural_gold_v1/annotation_browser_structural.html",
            "outputs/iter2_structural_gold_v1/annotations_structural_filled.csv",
            "outputs/iter2_structural_gold_v1/annotation_summary.md",
        ],
        "Package": [
            "outputs/iter2_package_v1/result_card.md",
            "outputs/iter2_package_v1/result_card.json",
            "outputs/iter2_package_v1/artifact_index.md",
        ],
    }
    lines = ["# Iteration 2 Artifact Index", ""]
    for section, paths in sections.items():
        lines.extend([f"## {section}", ""])
        for path in paths:
            exists = Path(path).exists()
            status = "exists" if exists else "missing"
            lines.append(f"- `{path}` ({status})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    result = build_result()
    (OUT_ROOT / "result_card.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_ROOT / "result_card.md").write_text(
        build_md(result),
        encoding="utf-8",
    )
    artifact_index = OUT_ROOT / "artifact_index.md"
    artifact_index.touch()
    artifact_index.write_text(build_artifact_index(), encoding="utf-8")
    print(json.dumps({
        "out_root": str(OUT_ROOT),
        "files": [
            str(OUT_ROOT / "result_card.md"),
            str(OUT_ROOT / "result_card.json"),
            str(OUT_ROOT / "artifact_index.md"),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
