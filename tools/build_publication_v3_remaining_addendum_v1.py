from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


OUT_DIR = Path("outputs/htr_publication_v3/remaining_addendum_v1")
PAGE_OUT = Path("outputs/htr_publication_v3/page_disjoint_hkr_school_v1")
PAGE_EVAL = PAGE_OUT / "eval_fixed_m04"
PAGE_PAIRED = PAGE_OUT / "paired_fixed_m04"
PAGE_MANIFEST_ROOTS = {
    "page_base": Path("data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1"),
    "page_line_10k": Path("data/experiments/htr_publication_v3/page_disjoint_hkr_school_plus_lines_10k_v1"),
    "page_random_crops_8k_control": Path(
        "data/experiments/htr_publication_v3/page_disjoint_hkr_school_random_crops_8k_control_v1"
    ),
    "page_school_words_8k_control": Path(
        "data/experiments/htr_publication_v3/page_disjoint_hkr_school_school_words_8k_control_v1"
    ),
}
PAGE_BASE = PAGE_MANIFEST_ROOTS["page_base"]
PAGE_LINE = PAGE_MANIFEST_ROOTS["page_line_10k"]
ANNOTATION = Path("outputs/htr_publication_v3/annotation_reliability_addendum_v1")
EXTERNAL_BASELINE_AVAILABILITY = Path("outputs/htr_publication_v3/external_baseline_availability_v1")
PAGE_SEEDS = [42, 43, 44]
PAGE_CORE_VARIANTS = ["page_base", "page_line_10k"]
PAGE_CONTROL_VARIANTS = ["page_random_crops_8k_control", "page_school_words_8k_control"]
PAGE_VARIANTS = PAGE_CORE_VARIANTS + PAGE_CONTROL_VARIANTS
PAGE_LINE_VS_CONTROL_KEYS = ["line_vs_random_crops_control", "line_vs_school_words_control"]
STRONG_BASELINES = {
    "mixed_cyrillic_natural_full_v1": Path(
        "outputs/htr_publication_v3/strong_internal_baselines/mixed_cyrillic_natural_full_v1_tri10k_test_fixed_m04/summary.json"
    ),
    "mixed_cyrillic_balanced50k_v1": Path(
        "outputs/htr_publication_v3/strong_internal_baselines/mixed_cyrillic_balanced50k_v1_tri10k_test_fixed_m04/summary.json"
    ),
}


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


def cached_hf_models() -> list[str]:
    root = Path.home() / ".cache" / "huggingface" / "hub"
    if not root.exists():
        return []
    models = []
    for path in root.glob("models--*"):
        if path.is_dir():
            models.append(path.name.removeprefix("models--").replace("--", "/"))
    return sorted(models)


def strong_baselines() -> dict[str, Any]:
    rows = []
    for name, path in STRONG_BASELINES.items():
        obj = maybe_json(path)
        if obj is None:
            rows.append({"name": name, "status": "missing", "summary": str(path)})
            continue
        metrics = obj["metrics"]
        rows.append({
            "name": name,
            "status": "complete",
            "summary": str(path),
            "checkpoint": obj.get("checkpoint"),
            "checkpoint_epoch": obj.get("checkpoint_epoch"),
            "checkpoint_val_cer": obj.get("checkpoint_val_cer"),
            "blank_logit_penalty": obj.get("blank_logit_penalty"),
            "n": metrics["n"],
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "exact": metrics["exact"],
        })
    return {
        "rows": rows,
        "cached_hf_models": cached_hf_models(),
        "interpretation": (
            "The only cached external HuggingFace OCR/HTR model found is TrOCR-base-handwritten. "
            "The additional strong baselines are internal CRNN baselines trained on larger in-domain data; "
            "they are useful for positioning but are not external SOTA."
        ),
    }


def page_eval_summary(variant: str, seed: int) -> dict[str, Any]:
    path = PAGE_EVAL / f"{variant}_seed{seed}_test" / "summary.json"
    obj = maybe_json(path)
    if obj is None:
        return {
            "variant": variant,
            "seed": seed,
            "exists": False,
            "summary": str(path),
        }
    metrics = obj["metrics"]
    return {
        "variant": variant,
        "seed": seed,
        "exists": True,
        "summary": str(path),
        "predictions": str(PAGE_EVAL / f"{variant}_seed{seed}_test" / "predictions.jsonl"),
        "checkpoint": obj.get("checkpoint"),
        "checkpoint_epoch": obj.get("checkpoint_epoch"),
        "checkpoint_val_cer": obj.get("checkpoint_val_cer"),
        "n": metrics["n"],
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "exact": metrics["exact"],
    }


def aggregate_page_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
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


def page_paired_seed(key: str, seed: int) -> dict[str, Any] | None:
    obj = maybe_json(PAGE_PAIRED / f"{key}_seed{seed}.json")
    if obj is None:
        return None
    overall = obj["overall"]
    boot = obj["bootstrap"]["overall"]
    school = obj["by_dataset"].get("school_notebooks_clean", {})
    school_boot = obj["bootstrap"].get("school_notebooks_clean", {})
    return {
        "key": key,
        "seed": seed,
        "n": overall["n"],
        "delta_cer": overall["mean_delta_cer"],
        "delta_wer": overall["delta_wer"],
        "delta_exact": overall["delta_exact"],
        "ci95_low": boot["ci95_low"],
        "ci95_high": boot["ci95_high"],
        "school_delta_cer": school.get("mean_delta_cer"),
        "school_ci95_low": school_boot.get("ci95_low"),
        "school_ci95_high": school_boot.get("ci95_high"),
        "dataset_delta_cer": {
            dataset: value["mean_delta_cer"]
            for dataset, value in obj["by_dataset"].items()
        },
    }


def page_disjoint_status() -> dict[str, Any]:
    manifests = {
        variant: maybe_json(root / "summary.json")
        for variant, root in PAGE_MANIFEST_ROOTS.items()
    }
    base = manifests["page_base"]
    line = manifests["page_line_10k"]
    run_status = maybe_json(PAGE_OUT / "run_status.json")
    eval_rows = [
        page_eval_summary(variant, seed)
        for variant in PAGE_VARIANTS
        for seed in PAGE_SEEDS
    ]
    aggregates = [
        aggregate_page_variant(eval_rows, variant)
        for variant in PAGE_VARIANTS
    ]
    paired = [
        row for row in (page_paired_seed("line_vs_base", seed) for seed in PAGE_SEEDS)
        if row is not None
    ]
    paired_controls = {
        key: [
            row for row in (page_paired_seed(key, seed) for seed in PAGE_SEEDS)
            if row is not None
        ]
        for key in PAGE_LINE_VS_CONTROL_KEYS
    }
    completed = [row for row in eval_rows if row.get("exists")]
    base_agg = next(row for row in aggregates if row["variant"] == "page_base")
    line_agg = next(row for row in aggregates if row["variant"] == "page_line_10k")
    core_eval_rows = [row for row in eval_rows if row["variant"] in PAGE_CORE_VARIANTS]
    control_eval_rows = [row for row in eval_rows if row["variant"] in PAGE_CONTROL_VARIANTS]
    core_eval_complete = all(row.get("exists") for row in core_eval_rows)
    controls_eval_complete = all(row.get("exists") for row in control_eval_rows)
    all_eval_complete = core_eval_complete and controls_eval_complete
    controls_manifest_ready = all(manifests[variant] is not None for variant in PAGE_CONTROL_VARIANTS)
    paired_controls_complete = all(len(rows) == len(PAGE_SEEDS) for rows in paired_controls.values())
    mean_delta_cer = None
    mean_delta_wer = None
    mean_delta_exact = None
    if base_agg["mean_cer"] is not None and line_agg["mean_cer"] is not None:
        mean_delta_cer = line_agg["mean_cer"] - base_agg["mean_cer"]
        mean_delta_wer = line_agg["mean_wer"] - base_agg["mean_wer"]
        mean_delta_exact = line_agg["mean_exact"] - base_agg["mean_exact"]
    line_vs_control_mean_deltas = {}
    for control_variant in PAGE_CONTROL_VARIANTS:
        control_agg = next(row for row in aggregates if row["variant"] == control_variant)
        if line_agg["mean_cer"] is None or control_agg["mean_cer"] is None:
            line_vs_control_mean_deltas[control_variant] = None
            continue
        line_vs_control_mean_deltas[control_variant] = {
            "mean_delta_cer": line_agg["mean_cer"] - control_agg["mean_cer"],
            "mean_delta_wer": line_agg["mean_wer"] - control_agg["mean_wer"],
            "mean_delta_exact": line_agg["mean_exact"] - control_agg["mean_exact"],
        }
    return {
        "base_summary": str(PAGE_BASE / "summary.json"),
        "line_summary": str(PAGE_LINE / "summary.json"),
        "control_summaries": {
            variant: str(PAGE_MANIFEST_ROOTS[variant] / "summary.json")
            for variant in PAGE_CONTROL_VARIANTS
        },
        "run_status": str(PAGE_OUT / "run_status.json"),
        "base": base,
        "line": line,
        "controls": {variant: manifests[variant] for variant in PAGE_CONTROL_VARIANTS},
        "run_status_rows": run_status,
        "completed_model_evals": completed,
        "eval_rows": eval_rows,
        "aggregates": aggregates,
        "paired_line_vs_base": paired,
        "paired_line_vs_controls": paired_controls,
        "mean_delta_cer": mean_delta_cer,
        "mean_delta_wer": mean_delta_wer,
        "mean_delta_exact": mean_delta_exact,
        "line_vs_control_mean_deltas": line_vs_control_mean_deltas,
        "all_eval_complete": all_eval_complete,
        "core_eval_complete": core_eval_complete,
        "controls_eval_complete": controls_eval_complete,
        "paired_ci_complete": len(paired) == len(PAGE_SEEDS),
        "paired_controls_complete": paired_controls_complete,
        "is_manifest_ready": base is not None and line is not None,
        "is_control_manifest_ready": controls_manifest_ready,
        "is_base_line_retrain_complete": core_eval_complete,
        "is_control_retrain_complete": controls_eval_complete,
        "is_full_retrain_complete": all_eval_complete,
        "recommended_full_command": (
            "python -u tools/run_page_disjoint_hkr_school_v1.py --seeds 42 43 44 --epochs 80 --num_workers 4"
        ),
        "recommended_control_command": (
            "python -u tools/run_page_disjoint_hkr_school_v1.py "
            "--variants page_random_crops_8k_control page_school_words_8k_control "
            "--seeds 42 43 44 --epochs 80 --num_workers 4"
        ),
        "recommended_control_comparison_command": "python tools/build_page_disjoint_control_comparisons_v1.py",
    }


def annotation_status() -> dict[str, Any]:
    obj = maybe_json(ANNOTATION / "summary.json")
    return {
        "summary": str(ANNOTATION / "summary.json"),
        "report": str(ANNOTATION / "report.md"),
        "data": obj,
    }


def build_summary() -> dict[str, Any]:
    page = page_disjoint_status()
    annotation = annotation_status()
    baselines = strong_baselines()
    external_availability = maybe_json(EXTERNAL_BASELINE_AVAILABILITY / "summary.json")
    weak_fields = []
    independent = None
    independent_package_ready = False
    formal_iaa_ready = False
    if annotation["data"] is not None:
        weak_fields = [
            row["field"]
            for row in annotation["data"]["publication_interpretation"]["weak_reliability_fields"]
        ]
        independent = annotation["data"].get("independent_annotation_v1")
        independent_package_ready = bool(independent and independent.get("package_ready"))
        formal_iaa_ready = bool(independent and independent.get("formal_iaa_ready"))
    return {
        "package": "publication_v3_remaining_addendum_v1",
        "page_disjoint": page,
        "annotation_reliability": annotation,
        "external_baseline_availability": {
            "summary": str(EXTERNAL_BASELINE_AVAILABILITY / "summary.json"),
            "report": str(EXTERNAL_BASELINE_AVAILABILITY / "report.md"),
            "data": external_availability,
        },
        "strong_internal_baselines": baselines,
        "publication_interpretation": {
            "now_added": [
                "page-disjoint HKR+School manifests with zero train/val/test page overlap",
                "page-disjoint line augmentation restricted to train pages",
                "page-disjoint same-size random-crop and School-word control manifests"
                if page["is_control_manifest_ready"] else None,
                "completed 3-seed page-disjoint base-vs-line retraining"
                if page["is_base_line_retrain_complete"] else None,
                "completed 3-seed page-disjoint same-size controls"
                if page["is_control_retrain_complete"] else None,
                "annotation repeated-consistency and line-quality Wilson intervals",
                "blind second-annotation package for formal IAA" if independent_package_ready else None,
                "strong data-rich internal CRNN baselines on the same tri10k test",
            ],
            "still_not_fully_solved": [
                "formal independent inter-annotator agreement" if not formal_iaa_ready else None,
                "competitive external Russian/Cyrillic HTR baseline beyond cached TrOCR",
                "completed 3-seed page-disjoint same-size controls" if not page["is_control_retrain_complete"] else None,
                "paired page-disjoint line-vs-control CIs" if not page["paired_controls_complete"] else None,
            ],
            "weak_annotation_fields": weak_fields,
            "independent_annotation_package_ready": independent_package_ready,
            "formal_iaa_ready": formal_iaa_ready,
            "strict_claim_boundary": (
                "The strict page-disjoint base, line, and same-size controls are complete. A unique "
                "natural-line-context claim is allowed only if the paired line-vs-control deltas support it."
                if page["is_full_retrain_complete"] else
                "The strict page-disjoint base-vs-line effect is supported if base/line evaluations are complete, "
                "but uniqueness of natural-line context remains unproven until the page-disjoint same-size controls "
                "and paired line-vs-control comparisons finish."
                if page["is_base_line_retrain_complete"] else
                "The new page-disjoint manifests make the required strict retraining feasible and reproducible. "
                "Until the long retrain finishes, they should be reported as prepared/queued rather than final result evidence."
            ),
        },
    }


def build_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Publication V3 Remaining Addendum v1",
        "",
        "## Strong Internal Baselines",
        "",
        summary["strong_internal_baselines"]["interpretation"],
        "",
        "| baseline | n | CER | WER | exact | checkpoint epoch | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["strong_internal_baselines"]["rows"]:
        lines.append(
            f"| `{row['name']}` | {row.get('n', 'n/a')} | {fmt(row.get('cer'))} | "
            f"{fmt(row.get('wer'))} | {fmt(row.get('exact'))} | {row.get('checkpoint_epoch', 'n/a')} | {row['status']} |"
        )
    lines.extend([
        "",
        f"Cached HuggingFace models: `{summary['strong_internal_baselines']['cached_hf_models']}`.",
        "",
        "External baseline availability:",
    ])
    external = summary["external_baseline_availability"]
    external_data = external.get("data")
    if external_data is None:
        lines.append("- availability report missing")
    else:
        external_interp = external_data["publication_interpretation"]
        lines.extend([
            f"- report: `{external['report']}`",
            f"- competitive external Russian/Cyrillic baseline available locally: {external_interp['competitive_external_russian_cyrillic_baseline_available_locally']}",
            f"- prepared EasyOCR wrapper: `{external_interp.get('prepared_easyocr_eval_wrapper')}`",
            f"- EasyOCR command after install: `{external_interp.get('example_easyocr_command')}`",
            f"- boundary: {external_interp['current_external_boundary']}",
        ])
    lines.extend([
        "",
        "## Page-Disjoint HKR+School Split",
        "",
    ])

    page = summary["page_disjoint"]
    if not page["is_manifest_ready"]:
        lines.append("Page-disjoint manifests are missing.")
    else:
        base = page["base"]
        line = page["line"]
        lines.extend([
            f"- base manifest root: `{Path(page['base_summary']).parent}`",
            f"- line manifest root: `{Path(page['line_summary']).parent}`",
            f"- base train/val/test n: {base['splits']['train']['n']}/{base['splits']['val']['n']}/{base['splits']['test']['n']}",
            f"- line train n: {line['train']['n']} (line samples selected: {line['line_train']['selected_n']})",
            f"- train-vs-test page overlap: {base['page_overlap']['train_vs_test']}",
            f"- cyrillic limitation: {base['metadata_limitation']}",
            f"- full retrain command: `{page['recommended_full_command']}`",
            f"- control retrain command: `{page['recommended_control_command']}`",
            f"- control comparison command: `{page['recommended_control_comparison_command']}`",
        ])
        if page["controls"]:
            lines.extend([
                "",
                "Page-disjoint same-size control manifests:",
                "",
                "| control | train n | added n | train-vs-test page overlap | ready |",
                "|---|---:|---:|---|---:|",
            ])
            for variant, control in page["controls"].items():
                if control is None:
                    lines.append(f"| `{variant}` | n/a | n/a | n/a | False |")
                    continue
                lines.append(
                    f"| `{variant}` | {control['train']['n']} | {control['target_total']} | "
                    f"{control['page_overlap']['train_vs_test']} | True |"
                )
        if page["run_status_rows"]:
            lines.extend([
                "",
                "| variant | seed | last epoch | best exists | eval returncode | status |",
                "|---|---:|---:|---|---:|---|",
            ])
            for row in page["run_status_rows"]:
                lines.append(
                    f"| `{row['variant']}` | {row['seed']} | {row.get('last_epoch_after')} | "
                    f"{row.get('best_exists')} | {row.get('eval_returncode')} | {row.get('status')} |"
                )
        if page["eval_rows"]:
            lines.extend([
                "",
                "Page-disjoint fixed-penalty evaluation:",
                "",
                "| variant | seed | n | CER | WER | exact | checkpoint epoch |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for row in page["eval_rows"]:
                if row.get("exists"):
                    lines.append(
                        f"| `{row['variant']}` | {row['seed']} | {row['n']} | {fmt(row['cer'])} | "
                        f"{fmt(row['wer'])} | {fmt(row['exact'])} | {row['checkpoint_epoch']} |"
                    )
                else:
                    lines.append(
                        f"| `{row['variant']}` | {row['seed']} | n/a | n/a | n/a | n/a | n/a |"
                    )
            lines.extend([
                "",
                "Page-disjoint aggregate:",
                "",
                "| variant | completed seeds | mean CER | std CER | mean WER | mean exact |",
                "|---|---|---:|---:|---:|---:|",
            ])
            for row in page["aggregates"]:
                lines.append(
                    f"| `{row['variant']}` | {row['completed_seeds']} | {fmt(row['mean_cer'])} | "
                    f"{fmt(row['std_cer'])} | {fmt(row['mean_wer'])} | {fmt(row['mean_exact'])} |"
                )
            lines.extend([
                "",
                f"Mean `page_line_10k - page_base` delta: CER {fmt(page['mean_delta_cer'])}, "
                f"WER {fmt(page['mean_delta_wer'])}, exact {fmt(page['mean_delta_exact'])}.",
            ])
            if page.get("line_vs_control_mean_deltas"):
                lines.extend([
                    "",
                    "Mean `page_line_10k - control` deltas:",
                    "",
                    "| control | delta CER | delta WER | delta exact |",
                    "|---|---:|---:|---:|",
                ])
                for control_variant, deltas in page["line_vs_control_mean_deltas"].items():
                    if deltas is None:
                        lines.append(f"| `{control_variant}` | n/a | n/a | n/a |")
                        continue
                    lines.append(
                        f"| `{control_variant}` | {fmt(deltas['mean_delta_cer'])} | "
                        f"{fmt(deltas['mean_delta_wer'])} | {fmt(deltas['mean_delta_exact'])} |"
                    )
        if page["paired_line_vs_base"]:
            lines.extend([
                "",
                "Page-disjoint paired line-vs-base comparison:",
                "",
                "| seed | n | delta CER | 95% CI | School delta CER | School 95% CI | delta WER | delta exact |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in page["paired_line_vs_base"]:
                lines.append(
                    f"| {row['seed']} | {row['n']} | {fmt(row['delta_cer'])} | "
                    f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | "
                    f"{fmt(row['school_delta_cer'])} | "
                    f"[{fmt(row['school_ci95_low'])}, {fmt(row['school_ci95_high'])}] | "
                    f"{fmt(row['delta_wer'])} | {fmt(row['delta_exact'])} |"
                )
        if page["paired_line_vs_controls"]:
            lines.extend([
                "",
                "Page-disjoint paired line-vs-control comparison:",
                "",
                "| comparison | seed | n | delta CER | 95% CI | School delta CER | School 95% CI | delta WER | delta exact |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for key, rows in page["paired_line_vs_controls"].items():
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
        "## Annotation Reliability",
        "",
    ])
    ann = summary["annotation_reliability"]["data"]
    if ann is None:
        lines.append("Annotation reliability addendum is missing.")
    else:
        repeated = ann["repeated_annotation_consistency"]
        independent = ann.get("independent_annotation_v1")
        lines.extend([
            f"Full report: `{summary['annotation_reliability']['report']}`.",
            f"Repeated annotation overlap n: {repeated['overlap_n']}.",
            "",
            "| field | agreement | kappa | weighted kappa |",
            "|---|---:|---:|---:|",
        ])
        for field, metrics in repeated["fields"].items():
            lines.append(
                f"| `{field}` | {fmt(metrics['agreement_rate'], 3)} | "
                f"{fmt(metrics['cohen_kappa'], 3)} | {fmt(metrics['quadratic_weighted_kappa'], 3)} |"
            )
        lines.extend([
            "",
            f"Interpretation: {ann['publication_interpretation']['not_supported']}",
        ])
        if independent:
            lines.extend([
                "",
                "Independent annotation package:",
                f"- package ready: {independent['package_ready']}",
                f"- browser: `{independent['browser']}`",
                f"- expected filled CSV: `{independent['expected_filled_csv']}`",
                f"- minimally complete rows: {independent['minimally_complete_rows']}",
                f"- formal IAA ready: {independent['formal_iaa_ready']}",
            ])

    interp = summary["publication_interpretation"]
    lines.extend([
        "",
        "## Remaining Boundary",
        "",
    ])
    for item in interp["now_added"]:
        if item:
            lines.append(f"- Added: {item}")
    for item in interp["still_not_fully_solved"]:
        if item:
            lines.append(f"- Still not fully solved: {item}")
    lines.append(f"- Claim boundary: {interp['strict_claim_boundary']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "report.md").write_text(build_md(summary), encoding="utf-8")
    print(json.dumps({
        "out_json": str(OUT_DIR / "summary.json"),
        "out_md": str(OUT_DIR / "report.md"),
        "page_manifest_ready": summary["page_disjoint"]["is_manifest_ready"],
        "page_full_retrain_complete": summary["page_disjoint"]["is_full_retrain_complete"],
        "strong_baselines": summary["strong_internal_baselines"]["rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
