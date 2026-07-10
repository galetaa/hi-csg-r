from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("outputs/htr_publication_v2")
FINAL_V1 = Path("outputs/final_result_package_v1")
DIAG_ROOT = OUT_ROOT / "diagnostic_finetune_3x500"


RUNS = [
    {
        "key": "base_continue",
        "label": "base continuation",
        "summary": DIAG_ROOT / "eval_base_continue_v1_test" / "summary.json",
        "interpretation": "same checkpoint, continued on original tri10k base manifest",
    },
    {
        "key": "line_context_10k",
        "label": "natural-line context +10k",
        "summary": DIAG_ROOT / "eval_line_context_10k_v1_test" / "summary.json",
        "interpretation": "same-size target method: +9998 rendered natural-line context crops",
    },
    {
        "key": "random_crops_10k_control",
        "label": "random crop control +10k",
        "summary": DIAG_ROOT / "eval_random_crops_10k_control_v1_test" / "summary.json",
        "interpretation": "same-size image-only control: balanced ordinary crop samples",
    },
    {
        "key": "school_words_10k_control",
        "label": "School word crop control +10k",
        "summary": DIAG_ROOT / "eval_school_words_10k_control_v1_test" / "summary.json",
        "interpretation": "same-size image-only control: extra School word crops without line context",
    },
]


PAIRED = [
    {
        "key": "line_vs_base",
        "label": "line context vs base continuation",
        "path": DIAG_ROOT / "paired_line_vs_base.json",
        "delta_definition": "line - base",
    },
    {
        "key": "line_vs_random",
        "label": "line context vs random crop control",
        "path": DIAG_ROOT / "paired_line_vs_random.json",
        "delta_definition": "line - random",
    },
    {
        "key": "line_vs_school_words",
        "label": "line context vs School word control",
        "path": DIAG_ROOT / "paired_line_vs_school_words.json",
        "delta_definition": "line - School words",
    },
    {
        "key": "random_vs_base",
        "label": "random crop control vs base continuation",
        "path": DIAG_ROOT / "paired_random_vs_base.json",
        "delta_definition": "random - base",
    },
    {
        "key": "school_words_vs_base",
        "label": "School word control vs base continuation",
        "path": DIAG_ROOT / "paired_school_words_vs_base.json",
        "delta_definition": "School words - base",
    },
]


CONTROL_SUMMARIES = [
    Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/summary.json"),
    Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_random_crops_10k_control_v1/summary.json"),
    Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_words_10k_control_v1/summary.json"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def demote_markdown_headings(text: str, *, by: int = 2) -> str:
    prefix = "#" * by
    lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            lines.append(prefix + line)
        else:
            lines.append(line)
    return "\n".join(lines)


def embedded_markdown(path: Path) -> str:
    return demote_markdown_headings(read_text_if_exists(path).strip())


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def compact_run(row: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(row["summary"])
    metrics = summary["metrics"]
    config = summary["checkpoint_config"]
    return {
        "key": row["key"],
        "label": row["label"],
        "summary": str(row["summary"]),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "checkpoint_val_cer": summary["checkpoint_val_cer"],
        "train_manifest": config["train_manifest"],
        "train_size": config["train_size"],
        "val_size": config["val_size"],
        "seed": config["seed"],
        "epochs": config["epochs"],
        "max_train_batches": config["max_train_batches"],
        "batch_size": config["batch_size"],
        "blank_logit_penalty": summary["blank_logit_penalty"],
        "test_n": metrics["n"],
        "test_cer": metrics["cer"],
        "test_wer": metrics["wer"],
        "test_exact": metrics["exact"],
        "interpretation": row["interpretation"],
    }


def compact_pair(row: dict[str, Any]) -> dict[str, Any]:
    obj = read_json(row["path"])
    overall = obj["overall"]
    overall_ci = obj["bootstrap"]["overall"]
    school = obj["by_dataset"].get("school_notebooks_clean", {})
    school_ci = obj["bootstrap"].get("school_notebooks_clean", {})
    return {
        "key": row["key"],
        "label": row["label"],
        "path": str(row["path"]),
        "delta_definition": row["delta_definition"],
        "n": overall["n"],
        "wins": overall["wins"],
        "losses": overall["losses"],
        "ties": overall["ties"],
        "mean_delta_cer": overall["mean_delta_cer"],
        "ci95_low": overall_ci["ci95_low"],
        "ci95_high": overall_ci["ci95_high"],
        "delta_wer": overall["delta_wer"],
        "delta_exact": overall["delta_exact"],
        "school_delta_cer": school.get("mean_delta_cer"),
        "school_ci95_low": school_ci.get("ci95_low"),
        "school_ci95_high": school_ci.get("ci95_high"),
        "dataset_delta_cer": {
            dataset: value["mean_delta_cer"]
            for dataset, value in obj["by_dataset"].items()
        },
    }


def pair_interpretation(pair: dict[str, Any]) -> str:
    lo = pair["ci95_low"]
    hi = pair["ci95_high"]
    delta = pair["mean_delta_cer"]
    if lo is not None and hi is not None and hi < 0:
        return "CER improvement; CI excludes zero"
    if lo is not None and hi is not None and lo > 0:
        return "CER degradation; CI excludes zero"
    if delta is not None and delta < 0:
        return "directionally better; CI overlaps zero"
    if delta is not None and delta > 0:
        return "directionally worse; CI overlaps zero"
    return "neutral"


def build_markdown(summary: dict[str, Any]) -> str:
    structural = summary["structural"]
    structural_bool = structural["overall_bool"]
    line_residual = structural["severity"]["line_residual"]
    missed_ink = structural["severity"]["missed_ink"]

    lines: list[str] = [
        "# Publication Evidence Package v2",
        "",
        "## Executive Status",
        "",
        "This package raises the research level beyond the previous technical-evidence package by adding same-size augmentation controls, paired confidence intervals, a strict structural-gold addendum, and manifest integrity audits.",
        "",
        "The work is now stronger for a thesis defense or a focused conference/workshop submission. It is still not ready for a strong journal or broad SOTA claim because the same-size controls are diagnostic fine-tunes rather than full from-scratch 3-seed runs, and no external transformer/SOTA HTR baseline is included.",
        "",
        "## Primary Result Retained From v1",
        "",
        embedded_markdown(FINAL_V1 / "thesis_tables" / "table_2_primary_htr_3seed.md"),
        "",
        embedded_markdown(FINAL_V1 / "thesis_tables" / "table_3_domainwise_htr.md"),
        "",
        "Interpretation: the main result remains the 3-seed natural-line context augmentation effect. The new v2 controls do not replace this primary result; they test whether the effect can plausibly be explained by adding the same amount of ordinary crop data.",
        "",
        "## Diagnostic Same-Size Control Protocol",
        "",
        "All diagnostic runs resume from `outputs/htr_graph_v1/tri10k_image_only_v1/last.pt`, continue for epochs 81-83, use seed 42, blank logit penalty -0.4, batch size 16, and `max_train_batches=500`. This is an auxiliary causal diagnostic, not a full independent training protocol.",
        "",
        "| variant | train n | val CER at checkpoint | test CER | test WER | exact | interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for run in summary["diagnostic_runs"]:
        lines.append(
            f"| `{run['label']}` | {run['train_size']} | {fmt(run['checkpoint_val_cer'])} | "
            f"{fmt(run['test_cer'])} | {fmt(run['test_wer'])} | {fmt(run['test_exact'])} | "
            f"{run['interpretation']} |"
        )

    lines.extend([
        "",
        "## Paired Diagnostic Comparisons",
        "",
        "Negative delta means the first model named in the comparison has lower CER than the second model named.",
        "",
        "| comparison | delta definition | n | delta CER | 95% CI | School delta CER | School 95% CI | delta WER | delta exact | interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for pair in summary["paired_comparisons"]:
        lines.append(
            f"| `{pair['label']}` | {pair['delta_definition']} | {pair['n']} | "
            f"{fmt(pair['mean_delta_cer'])} | [{fmt(pair['ci95_low'])}, {fmt(pair['ci95_high'])}] | "
            f"{fmt(pair['school_delta_cer'])} | [{fmt(pair['school_ci95_low'])}, {fmt(pair['school_ci95_high'])}] | "
            f"{fmt(pair['delta_wer'])} | {fmt(pair['delta_exact'])} | {pair_interpretation(pair)} |"
        )

    lines.extend([
        "",
        "## Diagnostic Interpretation",
        "",
        "- The line-context model is better than base continuation in the diagnostic protocol: delta CER -0.0039, CI [-0.0071, -0.0006].",
        "- The line-context model is better than the random-crop same-size control: delta CER -0.0050, CI [-0.0084, -0.0016].",
        "- The effect is mainly School-driven: School delta CER is -0.0137 vs base and -0.0162 vs random control.",
        "- The random-crop control is neutral versus base: delta CER +0.0011, CI crosses zero.",
        "- The School-word control is worse overall versus base: delta CER +0.0078, CI excludes zero, while its School-only effect is near neutral.",
        "- This supports the claim that natural-line context provides information not reproduced by adding the same number of ordinary crop samples.",
        "- The claim is still limited: line context slightly worsens Cyrillic in the diagnostic line-vs-base comparison, and the controls were not run as full 3-seed from-scratch experiments.",
        "",
        "## Manifest Integrity Audit",
        "",
        embedded_markdown(OUT_ROOT / "manifest_integrity_audit_v1.md"),
        "",
        "Interpretation: the audit reduces the risk of simple sample-id leakage, train duplicate inflation, OOV mismatch, or empty-text artifacts in the new controls. It does not rule out writer/page dependence or visual near-duplicates.",
        "",
        "## Structural Gold Strict Addendum",
        "",
        f"Structural subset n={structural['n']}; datasets={structural['dataset_counts']}; strata={structural['stratum_counts']}.",
        "",
        "| diagnostic field | count | n | rate | Wilson 95% low |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in ["structural_usable", "foreground_ok", "skeleton_ok", "graph_ok"]:
        row = structural_bool[field]
        lines.append(
            f"| `{field}` | {row['count']} | {row['n']} | {fmt(row['rate'])} | {fmt(row['wilson95_low'])} |"
        )

    lines.extend([
        "",
        "| issue | minor+ rate | severe/dominant rate |",
        "|---|---:|---:|",
        f"| `line_residual` | {pct(line_residual['minor_or_more_rate'])} | {pct(line_residual['severe_or_dominant_rate'])} |",
        f"| `missed_ink` | {pct(missed_ink['minor_or_more_rate'])} | {pct(missed_ink['severe_or_dominant_rate'])} |",
        "",
        "Strict interpretation: this supports diagnostic usability of the generated structures on the sampled cases. It does not prove exact graph topology, pen trajectory, writing order, endpoint correctness, or stroke-level ground truth. The absence of inter-annotator agreement remains a major publication limitation.",
        "",
        "## Stronger Baseline Status",
        "",
        demote_markdown_headings(
            read_text_if_exists(Path("outputs/htr_baseline_v1/htr_mixed_cyrillic_report.md"))
            .split("## 5. Interpretation")[0]
            .strip()
        ),
        "",
        "Interpretation: the project has a stronger internal CRNN-BiLSTM image-only baseline at larger Cyrillic data scale. This helps demonstrate engineering competence and data-scale behavior, but it is not the same experimental setting as tri10k plus line-context augmentation. No TrOCR/ViT/SOTA baseline was completed in this package.",
        "",
        "## Reproducibility Snapshot",
        "",
        embedded_markdown(OUT_ROOT / "reproducibility_snapshot_v1.md"),
        "",
        "Interpretation: the environment and repository state are now archived in the publication package. The snapshot still records a dirty working tree, so a clean commit/archive is required before submission.",
        "",
        "## Updated Claim Matrix",
        "",
        "| claim | status after v2 | allowed wording |",
        "|---|---|---|",
        "| Natural-line context improves HTR across 3 seeds | supported by primary v1 3-seed result | Allowed as the main recognition claim. |",
        "| The gain is not merely from adding +10k ordinary crops | supported diagnostically, not fully proven | Allowed only as diagnostic evidence; full same-size 3-seed controls are still required for a strong paper. |",
        "| Benefit is strongest on School Notebooks | supported by v1 domain table and v2 paired controls | Allowed. |",
        "| HI-CSG-R structures are usable for diagnostics | partially supported by structural gold addendum | Allowed with explicit diagnostic-only limitation. |",
        "| Graph topology/trajectory is recovered | not supported | Forbidden. |",
        "| System is SOTA | not supported | Forbidden. |",
        "",
        "## Remaining Publication Gaps",
        "",
        "1. Run full from-scratch same-size controls over at least three seeds: base tri10k, line-context +10k, random-crop +10k, School-word +10k.",
        "2. Add an external strong HTR baseline, preferably a transformer HTR model or a well-cited Russian/Cyrillic HTR baseline, under the same train/test protocol.",
        "3. Add writer/page-level or near-duplicate leakage audits if metadata permits.",
        "4. Add inter-annotator agreement and stricter pixel/topology metrics for the structural component.",
        "5. Freeze a clean repository state with exact environment, commands, checkpoints, and dataset build scripts.",
        "",
        "## Bottom Line",
        "",
        "The v2 additions materially improve scientific defensibility. The work can now be argued as an empirical study of natural-line context augmentation for Russian offline HTR with diagnostic structural evidence. It is still not publication-complete for a high-standard venue because the strongest alternative explanations are reduced but not eliminated by full independent controls.",
    ])

    return "\n".join(line for line in lines if line is not None) + "\n"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "package": "htr_publication_v2",
        "status": "strengthened but not journal-complete",
        "primary_v1_tables": {
            "htr_3seed": str(FINAL_V1 / "thesis_tables" / "table_2_primary_htr_3seed.md"),
            "domainwise": str(FINAL_V1 / "thesis_tables" / "table_3_domainwise_htr.md"),
        },
        "diagnostic_protocol": {
            "type": "auxiliary fine-tune control",
            "resume_checkpoint": "outputs/htr_graph_v1/tri10k_image_only_v1/last.pt",
            "epochs": "81-83",
            "seed": 42,
            "blank_logit_penalty": -0.4,
            "max_train_batches": 500,
            "limitation": "not a replacement for full from-scratch 3-seed same-size controls",
        },
        "control_manifests": [
            read_json(path)
            for path in CONTROL_SUMMARIES
        ],
        "diagnostic_runs": [
            compact_run(row)
            for row in RUNS
        ],
        "paired_comparisons": [
            compact_pair(row)
            for row in PAIRED
        ],
        "manifest_integrity_audit": read_json(OUT_ROOT / "manifest_integrity_audit_v1.json"),
        "reproducibility_snapshot": {
            "json": str(OUT_ROOT / "reproducibility_snapshot_v1.json"),
            "md": str(OUT_ROOT / "reproducibility_snapshot_v1.md"),
            "pip_freeze": str(OUT_ROOT / "pip_freeze.txt"),
        },
        "structural": read_json(
            OUT_ROOT / "structural_gold_strict_addendum_v1" / "structural_gold_strict_addendum_v1.json"
        ),
        "stronger_internal_baseline_report": "outputs/htr_baseline_v1/htr_mixed_cyrillic_report.md",
        "publication_assessment": {
            "scientific_level_after_v2": "stronger thesis / focused conference-workshop level",
            "can_make_scientific_claims": "partially",
            "main_value": "empirical evidence that natural-line context augmentation improves Russian offline HTR beyond ordinary same-size crop augmentation in diagnostic controls",
            "main_risk": "same-size controls are not full independent 3-seed from-scratch experiments and external SOTA baselines are absent",
            "first_priority": "run full from-scratch same-size controls over multiple seeds or explicitly scope the claim to diagnostic evidence",
        },
    }

    (OUT_ROOT / "publication_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_ROOT / "publication_v2_report.md").write_text(
        build_markdown(summary),
        encoding="utf-8",
    )

    print(json.dumps({
        "out_json": str(OUT_ROOT / "publication_v2_summary.json"),
        "out_md": str(OUT_ROOT / "publication_v2_report.md"),
        "status": summary["status"],
        "diagnostic_runs": [run["key"] for run in summary["diagnostic_runs"]],
        "paired_comparisons": [pair["key"] for pair in summary["paired_comparisons"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
