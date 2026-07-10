from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


OUT_ROOT = Path("outputs/htr_publication_v3")
CONTROLS_ROOT = OUT_ROOT / "full_same_size_controls"
EVAL_ROOT = CONTROLS_ROOT / "eval_fixed_m04"
PAIRED_ROOT = CONTROLS_ROOT / "paired_fixed_m04"
TROCR_ROOT = OUT_ROOT / "external_trocr_zero_shot_full"
TROCR_FINETUNED_ROOT = OUT_ROOT / "external_trocr_finetuned_tri10k_base_test"
VALIDITY_ROOT = OUT_ROOT / "validity_addendum_v1"
REMAINING_ROOT = OUT_ROOT / "remaining_addendum_v1"

SEEDS = [42, 43, 44]
VARIANTS = [
    "tri10k_base",
    "line_context_10k",
    "random_crops_10k_control",
    "school_words_10k_control",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_json(path: Path) -> Any | None:
    return read_json(path) if path.exists() else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def eval_summary(variant: str, seed: int) -> dict[str, Any] | None:
    return maybe_json(EVAL_ROOT / f"{variant}_seed{seed}_test" / "summary.json")


def eval_row(variant: str, seed: int) -> dict[str, Any]:
    obj = eval_summary(variant, seed)
    if obj is None:
        return {
            "variant": variant,
            "seed": seed,
            "exists": False,
        }
    metrics = obj["metrics"]
    return {
        "variant": variant,
        "seed": seed,
        "exists": True,
        "summary": str(EVAL_ROOT / f"{variant}_seed{seed}_test" / "summary.json"),
        "predictions": str(EVAL_ROOT / f"{variant}_seed{seed}_test" / "predictions.jsonl"),
        "checkpoint": obj.get("checkpoint"),
        "checkpoint_epoch": obj.get("checkpoint_epoch"),
        "checkpoint_val_cer": obj.get("checkpoint_val_cer"),
        "blank_logit_penalty": obj.get("blank_logit_penalty"),
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "exact": metrics["exact"],
    }


def aggregate_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    values = [row for row in rows if row["variant"] == variant and row.get("exists")]
    cers = [row["cer"] for row in values]
    wers = [row["wer"] for row in values]
    exacts = [row["exact"] for row in values]
    return {
        "variant": variant,
        "completed_seeds": [row["seed"] for row in values],
        "n_completed": len(values),
        "mean_cer": statistics.mean(cers) if cers else None,
        "std_cer": statistics.stdev(cers) if len(cers) > 1 else None,
        "mean_wer": statistics.mean(wers) if wers else None,
        "mean_exact": statistics.mean(exacts) if exacts else None,
    }


def validity_digest() -> dict[str, Any] | None:
    obj = maybe_json(VALIDITY_ROOT / "summary.json")
    if obj is None:
        return None

    interp = obj["publication_interpretation"]
    metadata_rows = []
    for variant, value in obj["metadata_leakage_audit"]["variants"].items():
        tt = value["overlaps"]["train_vs_test"]
        metadata_rows.append({
            "variant": variant,
            "sample_id_overlap": tt["sample_id"]["overlap_unique_keys"],
            "image_path_overlap": tt["image_path"]["overlap_unique_keys"],
            "page_overlap": tt["page_key"]["overlap_unique_keys"],
            "line_overlap": tt["line_key"]["overlap_unique_keys"],
            "text_overlap": tt["text"]["overlap_unique_keys"],
        })

    visual_rows = []
    for variant, value in obj["visual_duplicate_audit"]["variants"].items():
        visual_rows.append({
            "variant": variant,
            "sha1_overlap": value["exact_file_sha1_overlap_unique"],
            "dhash_candidate_overlap": value["exact_dhash_candidate_overlap_unique"],
            "train_paths_hashed": value["train_unique_paths_hashed"],
            "test_paths_hashed": value["test_unique_paths_hashed"],
        })

    stress_rows = []
    for variant, value in obj["group_stress_eval"]["variants"].items():
        for subset in [
            "all_test",
            "all_test_minus_high_risk_visual_near_duplicates",
            "page_disjoint_from_train",
            "page_seen_in_train",
        ]:
            metrics = value["aggregated"].get(subset)
            if metrics:
                stress_rows.append({
                    "variant": variant,
                    "subset": subset,
                    "n_min": metrics["n_min"],
                    "n_max": metrics["n_max"],
                    "mean_cer": metrics["mean_cer"],
                    "std_cer": metrics["std_cer"],
                    "mean_wer": metrics["mean_wer"],
                    "mean_exact": metrics["mean_exact"],
                })

    return {
        "out_json": str(VALIDITY_ROOT / "summary.json"),
        "out_md": str(VALIDITY_ROOT / "report.md"),
        "exact_metadata_leakage_flag_variants": interp["exact_metadata_leakage_flag_variants"],
        "page_overlap_flag_variants": interp["page_overlap_flag_variants"],
        "exact_visual_file_duplicate_flag_variants": interp["exact_visual_file_duplicate_flag_variants"],
        "high_risk_visual_near_duplicate_flag_variants": interp.get("high_risk_visual_near_duplicate_flag_variants", []),
        "high_risk_visual_near_duplicate_pairs": interp.get("high_risk_visual_near_duplicate_pairs", []),
        "claim_boundary": interp["claim_boundary"],
        "next_required_for_journal_level": interp["next_required_for_journal_level"],
        "dose_response_ready": interp["dose_response_ready"],
        "metadata_rows": metadata_rows,
        "visual_rows": visual_rows,
        "stress_rows": stress_rows,
        "dose_response_rows": obj["dose_response_fixed_m04"]["rows"],
        "dose_response_interpretation": obj["dose_response_fixed_m04"]["interpretation"],
        "line_vs_random_dataset_delta_cer": interp["line_vs_random_dataset_delta_cer"],
        "line_vs_school_words_dataset_delta_cer": interp["line_vs_school_words_dataset_delta_cer"],
    }


def remaining_digest() -> dict[str, Any] | None:
    obj = maybe_json(REMAINING_ROOT / "summary.json")
    if obj is None:
        return None

    page = obj["page_disjoint"]
    annotation = obj["annotation_reliability"]
    baselines = obj["strong_internal_baselines"]
    ann_data = annotation.get("data")
    repeated = ann_data.get("repeated_annotation_consistency") if ann_data else None
    independent = ann_data.get("independent_annotation_v1") if ann_data else None

    return {
        "out_json": str(REMAINING_ROOT / "summary.json"),
        "out_md": str(REMAINING_ROOT / "report.md"),
        "page_manifest_ready": page["is_manifest_ready"],
        "page_control_manifest_ready": page.get("is_control_manifest_ready"),
        "page_base_line_retrain_complete": page.get("is_base_line_retrain_complete"),
        "page_control_retrain_complete": page.get("is_control_retrain_complete"),
        "page_full_retrain_complete": page["is_full_retrain_complete"],
        "page_run_status": page["run_status"],
        "page_run_status_rows": page["run_status_rows"] or [],
        "page_completed_model_evals": page["completed_model_evals"],
        "page_eval_rows": page.get("eval_rows", []),
        "page_aggregates": page.get("aggregates", []),
        "page_paired_line_vs_base": page.get("paired_line_vs_base", []),
        "page_paired_line_vs_controls": page.get("paired_line_vs_controls", {}),
        "page_mean_delta_cer": page.get("mean_delta_cer"),
        "page_mean_delta_wer": page.get("mean_delta_wer"),
        "page_mean_delta_exact": page.get("mean_delta_exact"),
        "page_line_vs_control_mean_deltas": page.get("line_vs_control_mean_deltas", {}),
        "page_recommended_full_command": page["recommended_full_command"],
        "page_recommended_control_command": page.get("recommended_control_command"),
        "page_recommended_control_comparison_command": page.get("recommended_control_comparison_command"),
        "strong_internal_baselines": baselines["rows"],
        "strong_internal_baseline_interpretation": baselines["interpretation"],
        "cached_hf_models": baselines["cached_hf_models"],
        "annotation_report": annotation["report"],
        "annotation_overlap_n": repeated["overlap_n"] if repeated else None,
        "independent_annotation_package_ready": bool(independent and independent.get("package_ready")),
        "formal_iaa_ready": bool(independent and independent.get("formal_iaa_ready")),
        "independent_annotation_browser": independent.get("browser") if independent else None,
        "independent_annotation_expected_filled_csv": independent.get("expected_filled_csv") if independent else None,
        "independent_annotation_minimally_complete_rows": (
            independent.get("minimally_complete_rows") if independent else None
        ),
        "weak_annotation_fields": obj["publication_interpretation"]["weak_annotation_fields"],
        "still_not_fully_solved": [
            item for item in obj["publication_interpretation"]["still_not_fully_solved"]
            if item
        ],
        "strict_claim_boundary": obj["publication_interpretation"]["strict_claim_boundary"],
    }


def paired_seed(seed: int) -> dict[str, Any] | None:
    obj = maybe_json(PAIRED_ROOT / f"line_vs_base_seed{seed}.json")
    if obj is None:
        return None
    return {
        "seed": seed,
        "n": obj["overall"]["n"],
        "delta_cer": obj["overall"]["mean_delta_cer"],
        "ci95_low": obj["bootstrap"]["overall"]["ci95_low"],
        "ci95_high": obj["bootstrap"]["overall"]["ci95_high"],
        "school_delta_cer": obj["by_dataset"]["school_notebooks_clean"]["mean_delta_cer"],
        "school_ci95_low": obj["bootstrap"]["school_notebooks_clean"]["ci95_low"],
        "school_ci95_high": obj["bootstrap"]["school_notebooks_clean"]["ci95_high"],
        "dataset_delta_cer": {
            dataset: value["mean_delta_cer"]
            for dataset, value in obj["by_dataset"].items()
        },
    }


def build_summary() -> dict[str, Any]:
    eval_rows = [
        eval_row(variant, seed)
        for variant in VARIANTS
        for seed in SEEDS
    ]
    aggregates = [
        aggregate_variant(eval_rows, variant)
        for variant in VARIANTS
    ]
    paired = [row for row in (paired_seed(seed) for seed in SEEDS) if row is not None]
    trocr = maybe_json(TROCR_ROOT / "summary.json")
    trocr_finetuned = maybe_json(TROCR_FINETUNED_ROOT / "summary.json")
    if trocr_finetuned is not None and "trocr_finetuned" in str(trocr_finetuned.get("model_id", "")):
        trocr_finetuned["protocol"] = "decoder-only TrOCR adaptation; encoder frozen due 6GB GPU memory limit"
        trocr_finetuned["publication_limitation"] = (
            "This is an external TrOCR adaptation baseline, but not full end-to-end fine-tuning. "
            "The encoder was frozen because full TrOCR-base fine-tuning did not fit in 6GB GPU memory. "
            "The resulting test quality is weak and should be treated as a negative/limited external baseline."
        )

    base = next(row for row in aggregates if row["variant"] == "tri10k_base")
    line = next(row for row in aggregates if row["variant"] == "line_context_10k")

    normalized_delta = None
    if base["mean_cer"] is not None and line["mean_cer"] is not None:
        normalized_delta = line["mean_cer"] - base["mean_cer"]

    control_missing = [
        {"variant": row["variant"], "seed": row["seed"]}
        for row in eval_rows
        if row["variant"] in {"random_crops_10k_control", "school_words_10k_control"}
        and not row.get("exists")
    ]
    controls_complete = not control_missing
    finetuned_complete = trocr_finetuned is not None
    validity = validity_digest()
    validity_complete = validity is not None
    remaining = remaining_digest()
    remaining_complete = remaining is not None
    page_base_line_complete = bool(remaining and remaining.get("page_base_line_retrain_complete"))
    page_controls_complete = bool(remaining and remaining.get("page_control_retrain_complete"))
    page_retrain_complete = bool(remaining and remaining["page_full_retrain_complete"])
    formal_iaa_ready = bool(remaining and remaining.get("formal_iaa_ready"))

    still_missing = []
    if not controls_complete:
        still_missing.append("completed 3-seed from-scratch random-crop and School-word controls")
    if not finetuned_complete:
        still_missing.append("fine-tuned external transformer/HTR baseline")
    if not validity_complete:
        still_missing.append("writer/page or visual near-duplicate leakage audit if metadata allows")
    if not remaining_complete:
        still_missing.append("page-disjoint retraining setup, annotation reliability addendum, and stronger baseline addendum")
    if remaining_complete:
        still_missing.extend(remaining["still_not_fully_solved"])
    else:
        still_missing.append("inter-annotator agreement or repeated-annotation reliability evidence")

    if controls_complete and finetuned_complete and remaining_complete and page_retrain_complete:
        blockers = []
        if not formal_iaa_ready:
            blockers.append("formal independent IAA")
        blockers.append("lack of a competitive external Russian/Cyrillic HTR baseline")
        verdict = (
            "full same-size v3 controls, validity addenda, and strict 3-seed page-disjoint "
            "HKR+School retraining are complete; journal-level readiness is still mainly blocked by "
            + ", ".join(blockers)
        )
    elif controls_complete and finetuned_complete and remaining_complete and not page_retrain_complete:
        blockers = ["pending 3-seed page-disjoint same-size controls and paired line-vs-control comparisons"]
        if not formal_iaa_ready:
            blockers.append("formal independent IAA")
        blockers.append("lack of a competitive external Russian/Cyrillic HTR baseline")
        verdict = (
            "full same-size v3 controls and validity addenda are complete, and strict page-disjoint "
            "manifests are prepared; journal-level readiness is still blocked by "
            + ", ".join(blockers)
        )
    elif controls_complete and finetuned_complete:
        verdict = (
            "full planned v3 compute package is complete; publication readiness is now limited "
            "mainly by remaining validity/annotation audits and by the weak external TrOCR result"
        )
    else:
        verdict = (
            "stronger than v2, but still not full publication-level until the missing full controls "
            "and a fine-tuned external baseline are completed"
        )

    deduped_still_missing = []
    for item in still_missing:
        if item and item not in deduped_still_missing:
            deduped_still_missing.append(item)

    return {
        "package": "htr_publication_v3",
        "hardening_plan": "docs/publication_hardening_plan_v1.md",
        "fixed_penalty_protocol": {
            "blank_logit_penalty": -0.4,
            "purpose": "remove mixed test-time penalty selection from earlier summaries",
        },
        "eval_rows": eval_rows,
        "aggregates": aggregates,
        "normalized_line_vs_base": {
            "mean_delta_cer": normalized_delta,
            "base_mean_cer": base["mean_cer"],
            "line_mean_cer": line["mean_cer"],
            "base_completed_seeds": base["completed_seeds"],
            "line_completed_seeds": line["completed_seeds"],
            "paired_by_seed": paired,
        },
        "trocr_zero_shot": trocr,
        "trocr_finetuned": trocr_finetuned,
        "validity_addendum": validity,
        "remaining_addendum": remaining,
        "full_control_status": {
            "required_control_variants": [
                "random_crops_10k_control",
                "school_words_10k_control",
            ],
            "completed": [
                row for row in eval_rows
                if row["variant"] in {"random_crops_10k_control", "school_words_10k_control"}
                and row.get("exists")
            ],
            "missing": [
                row for row in control_missing
            ],
            "historical_interrupted_attempts": [
                {
                    "variant": "random_crops_10k_control",
                    "seed": 42,
                    "out_dir": str(CONTROLS_ROOT / "checkpoints" / "random_crops_10k_control_seed42"),
                    "status": "interrupted before epoch 1 checkpoint",
                    "observed_progress": "stdout flushed after interrupt showed epoch 1 reached approximately batch 950/2500",
                }
            ],
            "compute_estimate": {
                "observed": "approximately 950/2500 epoch-1 batches in roughly 8-10 interactive minutes before interrupt",
                "estimated_epoch_time_minutes": "25-30",
                "estimated_one_80_epoch_model_hours": "33-40",
                "estimated_six_missing_models_days": "8-10",
                "interpretation": (
                    "Strict full controls are feasible with a long dedicated compute window, "
                    "but not feasible to complete inside the current interactive review session."
                ),
            },
        },
        "publication_assessment": {
            "improved_now": [
                "fixed-penalty normalized 3-seed base-vs-line evaluation",
                "paired CI for normalized base-vs-line comparisons",
                "completed 3-seed from-scratch same-size random-crop and School-word controls" if controls_complete else None,
                "external pretrained TrOCR zero-shot baseline on the full test split",
                "fine-tuned/decode-adapted external TrOCR baseline on the full test split" if finetuned_complete else None,
                "automated metadata leakage, visual duplicate, group-stress, domain, error, and fixed dose-response addendum" if validity_complete else None,
                "completed strict 3-seed HKR+School page-disjoint base-vs-line retraining" if page_base_line_complete else None,
                "prepared strict page-disjoint same-size control manifests" if remaining and remaining.get("page_control_manifest_ready") else None,
                "completed strict 3-seed HKR+School page-disjoint same-size controls" if page_controls_complete else None,
                "strict HKR+School page-disjoint manifests and train-page-only line augmentation setup" if remaining_complete and not page_retrain_complete else None,
                "annotation repeated-consistency addendum and Wilson intervals for line-quality checks" if remaining_complete else None,
                "blind second-annotation package for formal IAA" if remaining and remaining["independent_annotation_package_ready"] else None,
                "strong data-rich internal CRNN baselines on the same tri10k test" if remaining_complete else None,
            ],
            "still_missing_for_strong_publication": [
                item for item in deduped_still_missing if item
            ],
            "verdict": verdict,
        },
    }


def build_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Publication V3 Status Report",
        "",
        "## Plan",
        "",
        f"Full hardening checklist: `{summary['hardening_plan']}`.",
        "",
        "## Normalized Fixed-Penalty Evaluation",
        "",
        "All rows below use fixed test-time `blank_logit_penalty=-0.4`.",
        "",
        "| variant | seed | CER | WER | exact | checkpoint epoch | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["eval_rows"]:
        if row.get("exists"):
            lines.append(
                f"| `{row['variant']}` | {row['seed']} | {fmt(row['cer'])} | "
                f"{fmt(row['wer'])} | {fmt(row['exact'])} | {row['checkpoint_epoch']} | complete |"
            )
        else:
            lines.append(
                f"| `{row['variant']}` | {row['seed']} | n/a | n/a | n/a | n/a | missing |"
            )

    lines.extend([
        "",
        "## Aggregate By Variant",
        "",
        "| variant | completed seeds | mean CER | std CER | mean WER | mean exact |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in summary["aggregates"]:
        lines.append(
            f"| `{row['variant']}` | {row['completed_seeds']} | {fmt(row['mean_cer'])} | "
            f"{fmt(row['std_cer'])} | {fmt(row['mean_wer'])} | {fmt(row['mean_exact'])} |"
        )

    norm = summary["normalized_line_vs_base"]
    lines.extend([
        "",
        "## Normalized Line-Context Effect",
        "",
        f"Mean base CER: {fmt(norm['base_mean_cer'])}. Mean line-context CER: {fmt(norm['line_mean_cer'])}. Mean delta CER: {fmt(norm['mean_delta_cer'])}.",
        "",
        "| seed | delta CER | 95% CI | School delta CER | School 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in norm["paired_by_seed"]:
        lines.append(
            f"| {row['seed']} | {fmt(row['delta_cer'])} | "
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | "
            f"{fmt(row['school_delta_cer'])} | "
            f"[{fmt(row['school_ci95_low'])}, {fmt(row['school_ci95_high'])}] |"
        )

    trocr = summary["trocr_zero_shot"]
    lines.extend([
        "",
        "## External TrOCR Baseline",
        "",
    ])
    if trocr is None:
        lines.append("TrOCR baseline is missing.")
    else:
        metrics = trocr["metrics"]
        lines.extend([
            f"- model: `{trocr['model_id']}`",
            f"- protocol: {trocr['protocol']}",
            f"- n: {metrics['n']}",
            f"- CER: {fmt(metrics['cer'])}",
            f"- WER: {fmt(metrics['wer'])}",
            f"- exact: {fmt(metrics['exact'])}",
            "",
            "Interpretation: this is an external pretrained zero-shot reference, not a competitive fine-tuned Russian HTR baseline. The result is weak and cannot satisfy a strong-publication baseline requirement by itself.",
        ])

    trocr_finetuned = summary["trocr_finetuned"]
    lines.extend([
        "",
        "## Fine-Tuned TrOCR Baseline",
        "",
    ])
    if trocr_finetuned is None:
        lines.append("Fine-tuned TrOCR baseline is not complete yet. It is queued in `tools/run_publication_v3_long_pipeline.py`.")
    else:
        metrics = trocr_finetuned["metrics"]
        protocol = trocr_finetuned.get("protocol")
        if "trocr_finetuned" in str(trocr_finetuned.get("model_id", "")):
            protocol = "decoder-only TrOCR adaptation; encoder frozen due 6GB GPU memory limit"
        lines.extend([
            f"- model: `{trocr_finetuned['model_id']}`",
            f"- protocol: {protocol}",
            f"- n: {metrics['n']}",
            f"- CER: {fmt(metrics['cer'])}",
            f"- WER: {fmt(metrics['wer'])}",
            f"- exact: {fmt(metrics['exact'])}",
            "",
            "Interpretation: this external baseline is complete but weak. It does not outperform the CRNN controls and should be reported as a negative/limited external-baseline result.",
        ])

    control = summary["full_control_status"]
    lines.extend([
        "",
        "## Full Same-Size Control Status",
        "",
        "| control variant | seed | status |",
        "|---|---:|---|",
    ])
    for row in control["completed"]:
        lines.append(
            f"| `{row['variant']}` | {row['seed']} | complete |"
        )
    for row in control["missing"]:
        lines.append(
            f"| `{row['variant']}` | {row['seed']} | missing |"
        )

    if control["missing"]:
        est = control["compute_estimate"]
        lines.extend([
            "",
            "Compute estimate:",
            f"- observed: {est['observed']}",
            f"- estimated epoch time: {est['estimated_epoch_time_minutes']} minutes",
            f"- estimated 80-epoch model time: {est['estimated_one_80_epoch_model_hours']} hours",
            f"- estimated six missing models: {est['estimated_six_missing_models_days']} days",
        ])

    validity = summary["validity_addendum"]
    lines.extend([
        "",
        "## Validity Addendum",
        "",
    ])
    if validity is None:
        lines.append("Validity addendum is not complete yet.")
    else:
        lines.extend([
            f"Full addendum report: `{validity['out_md']}`.",
            "",
            "Claim boundary:",
        ])
        for item in validity["claim_boundary"]:
            lines.append(f"- {item}")

        lines.extend([
            "",
            "Metadata leakage audit, train vs test:",
            "",
            "| variant | sample_id overlap | image_path overlap | page overlap | line overlap | text overlap |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in validity["metadata_rows"]:
            lines.append(
                f"| `{row['variant']}` | {row['sample_id_overlap']} | {row['image_path_overlap']} | "
                f"{row['page_overlap']} | {row['line_overlap']} | {row['text_overlap']} |"
            )

        lines.extend([
            "",
            "Visual duplicate audit:",
            "",
            "| variant | SHA1 overlap | dHash candidates | train paths hashed | test paths hashed |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in validity["visual_rows"]:
            lines.append(
                f"| `{row['variant']}` | {row['sha1_overlap']} | {row['dhash_candidate_overlap']} | "
                f"{row['train_paths_hashed']} | {row['test_paths_hashed']} |"
            )
        if validity.get("high_risk_visual_near_duplicate_pairs"):
            lines.extend([
                "",
                "High-risk dHash near-duplicate candidates:",
                "",
                "| train sample | test sample | train text | test text | variants |",
                "|---|---|---|---|---|",
            ])
            for pair in validity["high_risk_visual_near_duplicate_pairs"][:10]:
                lines.append(
                    f"| `{pair['train_sample_id']}` | `{pair['test_sample_id']}` | "
                    f"{pair['train_text']} | {pair['test_text']} | "
                    f"{', '.join(f'`{variant}`' for variant in pair['seen_in_variants'])} |"
                )

        lines.extend([
            "",
            "Fixed-penalty dose response:",
            "",
            "| run | line train n | CER | WER | exact | delta CER vs base |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in validity["dose_response_rows"]:
            lines.append(
                f"| `{row['key']}` | {row['line_train_n']} | {fmt(row.get('cer'))} | {fmt(row.get('wer'))} | "
                f"{fmt(row.get('exact'))} | {fmt(row.get('delta_cer_vs_baseline'))} |"
            )
        lines.append("")
        for item in validity["dose_response_interpretation"]:
            lines.append(f"- {item}")

    remaining = summary["remaining_addendum"]
    lines.extend([
        "",
        "## Remaining Addendum",
        "",
    ])
    if remaining is None:
        lines.append("Remaining publication addendum is not complete yet.")
    else:
        lines.extend([
            f"Full addendum report: `{remaining['out_md']}`.",
            "",
            "Strong internal baselines on the same tri10k test:",
            "",
            "| baseline | n | CER | WER | exact | checkpoint epoch | status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in remaining["strong_internal_baselines"]:
            lines.append(
                f"| `{row['name']}` | {row.get('n', 'n/a')} | {fmt(row.get('cer'))} | "
                f"{fmt(row.get('wer'))} | {fmt(row.get('exact'))} | "
                f"{row.get('checkpoint_epoch', 'n/a')} | {row['status']} |"
            )
        lines.extend([
            "",
            f"Baseline interpretation: {remaining['strong_internal_baseline_interpretation']}",
            f"Cached HuggingFace models: `{remaining['cached_hf_models']}`.",
            "",
            "Page-disjoint HKR+School status:",
            f"- manifest ready: {remaining['page_manifest_ready']}",
            f"- control manifest ready: {remaining['page_control_manifest_ready']}",
            f"- 3-seed base-vs-line retrain complete: {remaining['page_base_line_retrain_complete']}",
            f"- 3-seed same-size controls complete: {remaining['page_control_retrain_complete']}",
            f"- full strict page-disjoint package complete: {remaining['page_full_retrain_complete']}",
            f"- run status: `{remaining['page_run_status']}`",
            f"- full command: `{remaining['page_recommended_full_command']}`",
            f"- control command: `{remaining['page_recommended_control_command']}`",
            f"- control comparison command: `{remaining['page_recommended_control_comparison_command']}`",
        ])
        if remaining["page_run_status_rows"]:
            lines.extend([
                "",
                "| variant | seed | last epoch | best exists | eval returncode | status |",
                "|---|---:|---:|---|---:|---|",
            ])
            for row in remaining["page_run_status_rows"]:
                lines.append(
                    f"| `{row['variant']}` | {row['seed']} | {row.get('last_epoch_after')} | "
                    f"{row.get('best_exists')} | {row.get('eval_returncode')} | {row.get('status')} |"
                )
        if remaining.get("page_eval_rows"):
            lines.extend([
                "",
                "Page-disjoint fixed-penalty evaluation:",
                "",
                "| variant | seed | n | CER | WER | exact | checkpoint epoch |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for row in remaining["page_eval_rows"]:
                if row.get("exists"):
                    lines.append(
                        f"| `{row['variant']}` | {row['seed']} | {row['n']} | {fmt(row['cer'])} | "
                        f"{fmt(row['wer'])} | {fmt(row['exact'])} | {row['checkpoint_epoch']} |"
                    )
                else:
                    lines.append(
                        f"| `{row['variant']}` | {row['seed']} | n/a | n/a | n/a | n/a | n/a |"
                    )
        if remaining.get("page_aggregates"):
            lines.extend([
                "",
                "Page-disjoint aggregate:",
                "",
                "| variant | completed seeds | mean CER | std CER | mean WER | mean exact |",
                "|---|---|---:|---:|---:|---:|",
            ])
            for row in remaining["page_aggregates"]:
                lines.append(
                    f"| `{row['variant']}` | {row['completed_seeds']} | {fmt(row['mean_cer'])} | "
                    f"{fmt(row['std_cer'])} | {fmt(row['mean_wer'])} | {fmt(row['mean_exact'])} |"
                )
            lines.extend([
                "",
                f"Mean `page_line_10k - page_base` delta: CER {fmt(remaining.get('page_mean_delta_cer'))}, "
                f"WER {fmt(remaining.get('page_mean_delta_wer'))}, exact {fmt(remaining.get('page_mean_delta_exact'))}.",
            ])
            if remaining.get("page_line_vs_control_mean_deltas"):
                lines.extend([
                    "",
                    "Mean `page_line_10k - control` deltas:",
                    "",
                    "| control | delta CER | delta WER | delta exact |",
                    "|---|---:|---:|---:|",
                ])
                for control_variant, deltas in remaining["page_line_vs_control_mean_deltas"].items():
                    if deltas is None:
                        lines.append(f"| `{control_variant}` | n/a | n/a | n/a |")
                        continue
                    lines.append(
                        f"| `{control_variant}` | {fmt(deltas['mean_delta_cer'])} | "
                        f"{fmt(deltas['mean_delta_wer'])} | {fmt(deltas['mean_delta_exact'])} |"
                    )
        if remaining.get("page_paired_line_vs_base"):
            lines.extend([
                "",
                "Page-disjoint paired line-vs-base comparison:",
                "",
                "| seed | n | delta CER | 95% CI | School delta CER | School 95% CI | delta WER | delta exact |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in remaining["page_paired_line_vs_base"]:
                lines.append(
                    f"| {row['seed']} | {row['n']} | {fmt(row['delta_cer'])} | "
                    f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | "
                    f"{fmt(row['school_delta_cer'])} | "
                    f"[{fmt(row['school_ci95_low'])}, {fmt(row['school_ci95_high'])}] | "
                    f"{fmt(row['delta_wer'])} | {fmt(row['delta_exact'])} |"
                )
        if remaining.get("page_paired_line_vs_controls"):
            lines.extend([
                "",
                "Page-disjoint paired line-vs-control comparison:",
                "",
                "| comparison | seed | n | delta CER | 95% CI | School delta CER | School 95% CI | delta WER | delta exact |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for key, rows in remaining["page_paired_line_vs_controls"].items():
                if not rows:
                    lines.append(f"| `{key}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                    continue
                for row in rows:
                    lines.append(
                        f"| `{key}` | {row['seed']} | {row['n']} | {fmt(row['delta_cer'])} | "
                        f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | "
                        f"{fmt(row['school_delta_cer'])} | "
                        f"[{fmt(row['school_ci95_low'])}, {fmt(row['school_ci95_high'])}] | "
                        f"{fmt(row['delta_wer'])} | {fmt(row['delta_exact'])} |"
                    )
        lines.extend([
            "",
            "Annotation reliability:",
            f"- report: `{remaining['annotation_report']}`",
            f"- repeated annotation overlap n: {remaining['annotation_overlap_n']}",
            f"- independent package ready: {remaining['independent_annotation_package_ready']}",
            f"- independent browser: `{remaining['independent_annotation_browser']}`",
            f"- expected filled CSV: `{remaining['independent_annotation_expected_filled_csv']}`",
            f"- independent minimally complete rows: {remaining['independent_annotation_minimally_complete_rows']}",
            f"- formal IAA ready: {remaining['formal_iaa_ready']}",
            f"- weak fields: `{remaining['weak_annotation_fields']}`",
            f"- claim boundary: {remaining['strict_claim_boundary']}",
        ])

    lines.extend([
        "",
        "## Publication Assessment",
        "",
        "Completed now:",
    ])
    for item in summary["publication_assessment"]["improved_now"]:
        if item:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("Still missing:")
    for item in summary["publication_assessment"]["still_missing_for_strong_publication"]:
        if item:
            lines.append(f"- {item}")

    lines.extend([
        "",
        f"Verdict: {summary['publication_assessment']['verdict']}.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (OUT_ROOT / "publication_v3_status_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_ROOT / "publication_v3_status_report.md").write_text(
        build_md(summary),
        encoding="utf-8",
    )
    print(json.dumps({
        "out_json": str(OUT_ROOT / "publication_v3_status_summary.json"),
        "out_md": str(OUT_ROOT / "publication_v3_status_report.md"),
        "verdict": summary["publication_assessment"]["verdict"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
