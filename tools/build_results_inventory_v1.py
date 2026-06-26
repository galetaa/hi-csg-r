from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SUMMARY_NAMES = {
    "summary.json",
    "metrics.json",
    "eval_summary.json",
    "report.json",
}

PREDICTION_NAMES = {
    "predictions.jsonl",
    "predictions.csv",
}

CONFIG_NAMES = {
    "config.json",
    "config.yaml",
    "config.yml",
    "args.json",
    "params.json",
}

CHECKPOINT_NAMES = {
    "best.pt",
    "checkpoint.pt",
    "model.pt",
}


REQUIRED_RESULTS = [
    {
        "required_id": "image_only_baseline_seed42",
        "group": "primary",
        "description": "Canonical image-only baseline, seed 42.",
        "path_hint": "tri10k_image_only",
        "must_have_summary": True,
        "must_have_checkpoint": True,
    },
    {
        "required_id": "image_only_plus_10k_seed42",
        "group": "primary",
        "description": "Canonical +10k natural-line augmentation, seed 42.",
        "path_hint": "tri10k_image_only_plus_school_lines_10k_context",
        "must_have_summary": True,
        "must_have_checkpoint": True,
    },
    {
        "required_id": "image_only_baseline_seed43",
        "group": "primary_missing_or_pending",
        "description": "Seed confirmation baseline, seed 43.",
        "path_hint": "seed43",
        "must_have_summary": True,
        "must_have_checkpoint": True,
    },
    {
        "required_id": "image_only_plus_10k_seed43",
        "group": "primary_missing_or_pending",
        "description": "Seed confirmation +10k, seed 43.",
        "path_hint": "10k_context_v1_seed43",
        "must_have_summary": True,
        "must_have_checkpoint": True,
    },
    {
        "required_id": "image_only_baseline_seed44",
        "group": "primary_missing_or_pending",
        "description": "Seed confirmation baseline, seed 44.",
        "path_hint": "seed44",
        "must_have_summary": True,
        "must_have_checkpoint": True,
    },
    {
        "required_id": "image_only_plus_10k_seed44",
        "group": "primary_missing_or_pending",
        "description": "Seed confirmation +10k, seed 44.",
        "path_hint": "10k_context_v1_seed44",
        "must_have_summary": True,
        "must_have_checkpoint": True,
    },
    {
        "required_id": "line_aug_dose_response",
        "group": "primary",
        "description": "Dose-response comparison: baseline, +2k, +5k, +10k.",
        "path_hint": "line_aug_dose_response",
        "must_have_summary": False,
        "must_have_checkpoint": False,
    },
    {
        "required_id": "paired_5k_vs_10k",
        "group": "secondary",
        "description": "Paired comparison between +5k and +10k.",
        "path_hint": "paired_5k_vs_10k",
        "must_have_summary": False,
        "must_have_checkpoint": False,
    },
    {
        "required_id": "structural_gold",
        "group": "diagnostic",
        "description": "Structural gold diagnostic usability subset.",
        "path_hint": "structural_gold",
        "must_have_summary": False,
        "must_have_checkpoint": False,
    },
    {
        "required_id": "selective_prediction",
        "group": "secondary",
        "description": "Confidence / graph diagnostics selective prediction.",
        "path_hint": "selective",
        "must_have_summary": False,
        "must_have_checkpoint": False,
    },
    {
        "required_id": "graph_fusion_exploratory",
        "group": "exploratory",
        "description": "Graph-fusion / graph-aware HTR exploratory results.",
        "path_hint": "graph_fusion",
        "must_have_summary": True,
        "must_have_checkpoint": False,
    },
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_nearest_file(root: Path, names: set[str]) -> str:
    if not root.exists():
        return ""
    for p in sorted(root.iterdir()):
        if p.is_file() and p.name in names:
            return str(p)
    return ""


def find_any_descendant(root: Path, names: set[str], max_depth: int = 2) -> str:
    if not root.exists():
        return ""

    root_parts = len(root.parts)
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if len(p.parts) - root_parts > max_depth:
            continue
        if p.name in names:
            return str(p)
    return ""


def extract_metric(summary: dict[str, Any], key: str) -> str:
    if key in summary:
        return str(summary[key])

    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return str(metrics[key])

    return ""


def extract_grouped(summary: dict[str, Any]) -> dict[str, Any]:
    grouped = summary.get("grouped")
    if isinstance(grouped, dict):
        return grouped

    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        grouped = metrics.get("grouped")
        if isinstance(grouped, dict):
            return grouped

    return {}


def infer_seed(path: Path, summary: dict[str, Any]) -> str:
    for key in ["seed", "random_seed"]:
        if key in summary:
            return str(summary[key])

    text = str(path)
    m = re.search(r"seed[_-]?(\d+)", text)
    if m:
        return m.group(1)

    return ""


def infer_status(path: Path) -> str:
    text = str(path).lower()

    if "graph_fusion" in text or "gated" in text or "graph-aware" in text:
        return "exploratory"
    if "selective" in text or "confidence" in text:
        return "secondary"
    if "structural_gold" in text or "gold" in text:
        return "diagnostic"
    if "line_aug" in text or "10k_context" in text or "5k_context" in text:
        return "primary"
    if "baseline" in text or "image_only" in text:
        return "primary"

    return "needs_review"


def infer_model_name(path: Path, summary: dict[str, Any]) -> str:
    for key in ["model", "model_name", "run_name", "experiment", "experiment_id"]:
        if key in summary:
            return str(summary[key])

    return path.parent.name


def scan_summaries(search_roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    seen: set[Path] = set()

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for summary_path in sorted(search_root.rglob("*")):
            if not summary_path.is_file():
                continue
            if summary_path.name not in SUMMARY_NAMES:
                continue
            if summary_path in seen:
                continue
            seen.add(summary_path)

            run_dir = summary_path.parent
            summary = read_json(summary_path)

            checkpoint_path = (
                find_nearest_file(run_dir, CHECKPOINT_NAMES)
                or find_any_descendant(run_dir, CHECKPOINT_NAMES, max_depth=1)
            )
            config_path = (
                find_nearest_file(run_dir, CONFIG_NAMES)
                or find_any_descendant(run_dir, CONFIG_NAMES, max_depth=1)
            )
            predictions_path = (
                find_nearest_file(run_dir, PREDICTION_NAMES)
                or find_any_descendant(run_dir, PREDICTION_NAMES, max_depth=1)
            )

            grouped = extract_grouped(summary)

            row = {
                "experiment_id": run_dir.name,
                "result_group": infer_status(run_dir),
                "model": infer_model_name(summary_path, summary),
                "dataset": str(summary.get("dataset", "")),
                "seed": infer_seed(summary_path, summary),
                "summary_path": str(summary_path),
                "checkpoint_path": checkpoint_path,
                "config_path": config_path,
                "predictions_path": predictions_path,
                "n": extract_metric(summary, "n"),
                "cer": extract_metric(summary, "cer"),
                "wer": extract_metric(summary, "wer"),
                "exact": extract_metric(summary, "exact"),
                "pred_len_mean": extract_metric(summary, "pred_len_mean"),
                "blank_logit_penalty": str(summary.get("blank_logit_penalty", "")),
                "checkpoint_epoch": str(summary.get("checkpoint_epoch", "")),
                "checkpoint_val_cer": str(summary.get("checkpoint_val_cer", "")),
                "grouped_domains": ",".join(sorted(grouped.keys())),
                "included_in_thesis": "needs_decision",
                "notes": "",
            }

            for domain in ["cyrillic_handwriting", "hkr_words", "school_notebooks_clean", "school"]:
                d = grouped.get(domain)
                if isinstance(d, dict):
                    row[f"{domain}_cer"] = str(d.get("cer", ""))
                    row[f"{domain}_wer"] = str(d.get("wer", ""))
                    row[f"{domain}_exact"] = str(d.get("exact", ""))

            rows.append(row)

    return rows


def check_required(search_roots: list[Path]) -> list[dict[str, Any]]:
    all_paths: list[Path] = []
    for root in search_roots:
        if root.exists():
            all_paths.extend([p for p in root.rglob("*")])

    rows: list[dict[str, Any]] = []

    for req in REQUIRED_RESULTS:
        hint = req["path_hint"].lower()
        matches = [p for p in all_paths if hint in str(p).lower()]
        summary_matches = [p for p in matches if p.is_file() and p.name in SUMMARY_NAMES]
        checkpoint_matches = [p for p in matches if p.is_file() and p.name in CHECKPOINT_NAMES]

        status = "found"
        problems: list[str] = []

        if not matches:
            status = "missing"
            problems.append("no path matched path_hint")

        if req.get("must_have_summary") and not summary_matches:
            status = "incomplete" if matches else "missing"
            problems.append("required summary not found")

        if req.get("must_have_checkpoint") and not checkpoint_matches:
            status = "incomplete" if matches else "missing"
            problems.append("required checkpoint not found")

        rows.append(
            {
                "required_id": req["required_id"],
                "group": req["group"],
                "description": req["description"],
                "path_hint": req["path_hint"],
                "status": status,
                "matched_paths_n": len(matches),
                "summary_paths": ";".join(str(p) for p in summary_matches[:5]),
                "checkpoint_paths": ";".join(str(p) for p in checkpoint_matches[:5]),
                "problems": "; ".join(problems),
            }
        )

    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_missing_md(required_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Missing / incomplete result items\n")
    lines.append("This file is generated by `tools/build_results_inventory_v1.py`.\n")

    missing = [r for r in required_rows if r["status"] != "found"]

    if not missing:
        lines.append("No missing required items detected.\n")
    else:
        for r in missing:
            lines.append(f"## {r['required_id']}\n")
            lines.append(f"- group: `{r['group']}`")
            lines.append(f"- status: `{r['status']}`")
            lines.append(f"- description: {r['description']}")
            lines.append(f"- path_hint: `{r['path_hint']}`")
            lines.append(f"- problems: {r['problems'] or 'not specified'}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["outputs", "data/experiments"],
        help="Roots to scan.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/final_result_package_v1",
    )
    args = parser.parse_args()

    search_roots = [Path(p) for p in args.roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = scan_summaries(search_roots)
    required_rows = check_required(search_roots)

    inventory_csv = out_dir / "results_inventory.csv"
    required_csv = out_dir / "required_results_check.csv"
    missing_md = out_dir / "missing_items.md"
    summary_json = out_dir / "inventory_summary.json"

    write_csv(inventory_rows, inventory_csv)
    write_csv(required_rows, required_csv)
    write_missing_md(required_rows, missing_md)

    summary = {
        "inventory_rows_n": len(inventory_rows),
        "required_items_n": len(required_rows),
        "required_found_n": sum(1 for r in required_rows if r["status"] == "found"),
        "required_incomplete_n": sum(1 for r in required_rows if r["status"] == "incomplete"),
        "required_missing_n": sum(1 for r in required_rows if r["status"] == "missing"),
        "outputs": {
            "results_inventory": str(inventory_csv),
            "required_results_check": str(required_csv),
            "missing_items": str(missing_md),
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
