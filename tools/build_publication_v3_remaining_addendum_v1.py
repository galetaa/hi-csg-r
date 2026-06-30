from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("outputs/htr_publication_v3/remaining_addendum_v1")
PAGE_OUT = Path("outputs/htr_publication_v3/page_disjoint_hkr_school_v1")
PAGE_BASE = Path("data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1")
PAGE_LINE = Path("data/experiments/htr_publication_v3/page_disjoint_hkr_school_plus_lines_10k_v1")
ANNOTATION = Path("outputs/htr_publication_v3/annotation_reliability_addendum_v1")
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


def page_disjoint_status() -> dict[str, Any]:
    base = maybe_json(PAGE_BASE / "summary.json")
    line = maybe_json(PAGE_LINE / "summary.json")
    run_status = maybe_json(PAGE_OUT / "run_status.json")
    completed = []
    if isinstance(run_status, list):
        completed = [
            row for row in run_status
            if row.get("status") == "complete"
            and row.get("best_exists")
            and row.get("eval_returncode") == 0
        ]
    return {
        "base_summary": str(PAGE_BASE / "summary.json"),
        "line_summary": str(PAGE_LINE / "summary.json"),
        "run_status": str(PAGE_OUT / "run_status.json"),
        "base": base,
        "line": line,
        "run_status_rows": run_status,
        "completed_model_evals": completed,
        "is_manifest_ready": base is not None and line is not None,
        "is_full_retrain_complete": len(completed) >= 6,
        "recommended_full_command": (
            "python -u tools/run_page_disjoint_hkr_school_v1.py --seeds 42 43 44 --epochs 80 --num_workers 4"
        ),
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
        "strong_internal_baselines": baselines,
        "publication_interpretation": {
            "now_added": [
                "page-disjoint HKR+School manifests with zero train/val/test page overlap",
                "page-disjoint line augmentation restricted to train pages",
                "annotation repeated-consistency and line-quality Wilson intervals",
                "blind second-annotation package for formal IAA" if independent_package_ready else None,
                "strong data-rich internal CRNN baselines on the same tri10k test",
            ],
            "still_not_fully_solved": [
                "formal independent inter-annotator agreement" if not formal_iaa_ready else None,
                "competitive external Russian/Cyrillic HTR baseline beyond cached TrOCR",
                "completed 3-seed page-disjoint from-scratch retraining" if not page["is_full_retrain_complete"] else None,
            ],
            "weak_annotation_fields": weak_fields,
            "independent_annotation_package_ready": independent_package_ready,
            "formal_iaa_ready": formal_iaa_ready,
            "strict_claim_boundary": (
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
        ])
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
