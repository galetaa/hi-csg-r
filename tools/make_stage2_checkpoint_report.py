from __future__ import annotations

import json
from pathlib import Path


GRAPH_PILOT_REPORT = Path("outputs/graph_pilot_v2/graph_pilot_v2_report.md")
CLEAN_SUMMARY = Path("outputs/graph_pilot_v2/clean_graph_subset_v1_summary.json")
FEATURE_SUMMARY = Path("outputs/graph_pilot_v2/graph_features_v1_summary.json")
QUALITY_SUMMARY = Path("outputs/graph_pilot_v2/graph_quality_metrics_v1_summary.json")

OUT = Path("docs/02_stage2_checkpoint_report.md")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def block_json(obj) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def main() -> None:
    clean_summary = load_json(CLEAN_SUMMARY)
    feature_summary = load_json(FEATURE_SUMMARY)
    quality_summary = load_json(QUALITY_SUMMARY)

    clean = clean_summary["clean_subset"]
    stress = clean_summary["page_stress_subset"]

    clean_features = feature_summary["clean_features"]
    stress_features = feature_summary["page_stress_features"]

    primary_variants = quality_summary["primary_variants"]

    lines = []

    lines.append("# Stage 2 checkpoint report — HI-CSG-R graph pilot v2\n")

    lines.append("## 1. Purpose\n")
    lines.append(
        "Stage 2 established the first reproducible HI-CSG-R graph extraction pipeline "
        "for offline handwritten images. The stage moved from preprocessed handwriting images "
        "to binary masks, skeletons, pixel graphs, canonical graph JSON files, visual overlays, "
        "diagnostics, normalized graph metrics, and clean/stress graph subsets.\n"
    )

    lines.append("## 2. Current graph pipeline\n")
    lines.append("```text")
    lines.append("feature image")
    lines.append("→ binarization")
    lines.append("→ skeletonization")
    lines.append("→ pixel graph")
    lines.append("→ node detection")
    lines.append("→ junction clustering")
    lines.append("→ edge tracing")
    lines.append("→ graph.json")
    lines.append("→ overlay")
    lines.append("→ normalized graph metrics")
    lines.append("```")
    lines.append("")

    lines.append("## 3. Datasets used in graph pilot v2\n")
    lines.append("```text")
    lines.append("IAM                  → English line-level clean graph")
    lines.append("Cyrillic Handwriting → Russian word/phrase clean graph")
    lines.append("HKR Words            → Russian/Kazakh Cyrillic word/phrase clean graph")
    lines.append("School Notebooks     → Russian polygon crop clean graph")
    lines.append("HWR200               → scan/photo/dark page-level stress-test")
    lines.append("HKR Forms            → form/page stress-test")
    lines.append("```")
    lines.append("")

    lines.append("## 4. Primary graph variants\n")
    lines.append(block_json(primary_variants))
    lines.append("")

    lines.append("## 5. Methodological decisions\n")
    lines.append("- IAM uses `otsu` as the primary graph variant.\n")
    lines.append("- Cyrillic Handwriting uses `otsu` as the primary graph variant.\n")
    lines.append("- HKR Words uses `otsu` as the primary graph variant.\n")
    lines.append("- School Notebooks uses `sauvola` as the primary graph variant because Otsu often over-selects polygon/background foreground.\n")
    lines.append("- HWR200 uses `otsu_gridless` only as a page-level stress-test variant.\n")
    lines.append("- HKR Forms uses `otsu_gridless` only as a page/form stress-test variant.\n")
    lines.append("- `adaptive_gaussian` is not used for graph-builder because it tends to increase skeleton noise.\n")
    lines.append("- Component filtering is not used by default on word/line/polygon crops.\n")
    lines.append("- `min_skel=8` is only a diagnostic page-noise filter for page/form datasets.\n")
    lines.append("- Clean graph metrics and page stress metrics must not be mixed in one evaluation table.\n")
    lines.append("- `mean_width_proxy` is a stroke-width proxy, not true pen pressure.\n")
    lines.append("")

    lines.append("## 6. Clean graph subset v1\n")
    lines.append("Clean graph subset contains only primary graph variants from crop/line datasets with zero warning risk.\n")
    lines.append(block_json({
        "path": clean["path"],
        "count": clean["count"],
        "by_dataset": clean["by_dataset"],
        "by_method": clean["by_method"],
        "metrics_by_dataset": clean["metrics_by_dataset"],
    }))
    lines.append("")

    lines.append("## 7. Page stress graph subset v1\n")
    lines.append(
        "Page stress subset contains HWR200 and HKR Forms primary stress variants. "
        "These graphs are useful for robustness and failure analysis, but they are not treated "
        "as clean canonical stroke graphs.\n"
    )
    lines.append(block_json({
        "path": stress["path"],
        "count": stress["count"],
        "by_dataset": stress["by_dataset"],
        "by_method": stress["by_method"],
        "metrics_by_dataset": stress["metrics_by_dataset"],
    }))
    lines.append("")

    lines.append("## 8. Clean graph features v1\n")
    lines.append("The first clean graph feature table has been exported.\n")
    lines.append(block_json({
        "path": clean_features["path"],
        "count": clean_features["count"],
        "by_dataset": clean_features["by_dataset"],
        "by_method": clean_features["by_method"],
        "metrics_by_dataset": clean_features["metrics_by_dataset"],
    }))
    lines.append("")

    lines.append("## 9. Page stress graph features v1\n")
    lines.append("The page stress graph feature table has also been exported, but it must remain separate from clean graph features.\n")
    lines.append(block_json({
        "path": stress_features["path"],
        "count": stress_features["count"],
        "by_dataset": stress_features["by_dataset"],
        "by_method": stress_features["by_method"],
        "metrics_by_dataset": stress_features["metrics_by_dataset"],
    }))
    lines.append("")

    lines.append("## 10. Current interpretation\n")
    lines.append("### 10.1 Clean crop/line graphs\n")
    lines.append(
        "IAM, Cyrillic Handwriting, HKR Words, and School Notebooks all produced clean primary graph subsets "
        "with zero warning risk. The normalized features show meaningful dataset-level differences. "
        "For example, IAM has longer line-level samples and therefore larger absolute skeleton length, "
        "while Cyrillic and School Notebooks have higher normalized node/component density in this pilot.\n"
    )

    lines.append("### 10.2 Page/form stress graphs\n")
    lines.append(
        "HWR200 and HKR Forms remain high-complexity stress-test datasets. Their warning risk is expected, "
        "because full-page forms, grid backgrounds, and document layout structures create many components, "
        "endpoints, and junctions. These datasets should be used for robustness and failure analysis, "
        "not for clean graph feature training without crop/region selection.\n"
    )

    lines.append("### 10.3 School Notebooks thresholding\n")
    lines.append(
        "School Notebooks uses polygon-masked crops. Otsu often over-selects the polygon/background area, "
        "while Sauvola better follows the handwritten stroke structure visually. Therefore, Sauvola is the "
        "primary graph variant for School Notebooks.\n"
    )

    lines.append("## 11. Artifacts produced\n")
    lines.append("```text")
    lines.append("data/pilot/graph_pilot_v2.jsonl")
    lines.append("data/pilot/graph_pilot_v2_summary.json")
    lines.append("outputs/graph_pilot_v2/binary_skeleton_pilot_summary.json")
    lines.append("outputs/graph_pilot_v2/graph_builder_pilot_report.json")
    lines.append("outputs/graph_pilot_v2/graph_pilot_v2_report.md")
    lines.append("outputs/graph_pilot_v2/graph_failure_cases_v2.json")
    lines.append("outputs/graph_pilot_v2/graph_quality_metrics_v1.csv")
    lines.append("outputs/graph_pilot_v2/graph_quality_metrics_v1_summary.json")
    lines.append("outputs/graph_pilot_v2/clean_graph_subset_v1.jsonl")
    lines.append("outputs/graph_pilot_v2/page_stress_graph_subset_v1.jsonl")
    lines.append("outputs/graph_pilot_v2/graph_features_clean_v1.csv")
    lines.append("outputs/graph_pilot_v2/graph_features_page_stress_v1.csv")
    lines.append("outputs/graph_pilot_v2/graph_features_v1_summary.json")
    lines.append("```")
    lines.append("")

    lines.append("## 12. Acceptance criteria status\n")
    lines.append("```text")
    lines.append("[x] HI-CSG-R graph schema documented")
    lines.append("[x] Pilot subset created")
    lines.append("[x] Binary masks produced")
    lines.append("[x] Skeletons produced")
    lines.append("[x] Pixel graph produced")
    lines.append("[x] Canonical graph JSON saved")
    lines.append("[x] Graph overlays saved")
    lines.append("[x] Diagnostics saved")
    lines.append("[x] Dataset-specific binarization decisions made")
    lines.append("[x] Failure cases selected")
    lines.append("[x] Normalized graph metrics exported")
    lines.append("[x] Clean graph subset separated from page stress subset")
    lines.append("```")
    lines.append("")

    lines.append("## 13. Next stage\n")
    lines.append("Recommended next stage:\n")
    lines.append("```text")
    lines.append("Stage 3 — HTR baselines and graph-aware experimental protocol")
    lines.append("```")
    lines.append("")
    lines.append("Stage 3 should include:\n")
    lines.append("- image-only HTR baselines for IAM, Cyrillic Handwriting, HKR Words, and School Notebooks;\n")
    lines.append("- graph feature sanity checks on clean_graph_subset_v1;\n")
    lines.append("- larger clean graph extraction run for crop/line datasets;\n")
    lines.append("- graph-only baseline prototype;\n")
    lines.append("- image+graph fusion design;\n")
    lines.append("- robustness/failure analysis using page_stress_graph_subset_v1.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print("wrote:", OUT)


if __name__ == "__main__":
    main()