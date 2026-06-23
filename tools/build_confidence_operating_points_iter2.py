from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.build_confidence_selective_report_iter2 import (
    fit_risk_model,
    load_manifest,
    overlay_quality,
    predict_risk,
    read_jsonl,
    write_json,
    write_jsonl,
)


TARGETS = {
    "strict": 0.50,
    "balanced": 0.80,
    "broad": 0.90,
}


def metrics(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    accepted = [
        row for row in rows
        if float(row["risk"]) <= threshold
    ]
    n = len(rows)
    if not accepted:
        return {
            "n_total": n,
            "n_accepted": 0,
            "coverage": 0.0,
            "cer": None,
            "wer": None,
            "exact": None,
        }
    return {
        "n_total": n,
        "n_accepted": len(accepted),
        "coverage": len(accepted) / max(n, 1),
        "cer": float(np.mean([float(row["cer"]) for row in accepted])),
        "wer": float(np.mean([float(row["wer"]) for row in accepted])),
        "exact": float(np.mean([float(row["exact"]) for row in accepted])),
    }


def attach_risk(
    rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    risks: np.ndarray,
) -> list[dict[str, Any]]:
    out = []
    for row, risk in zip(rows, risks):
        item = dict(row)
        meta = manifest[str(row["sample_id"])]
        item["risk"] = float(risk)
        item["school_quality_bucket"] = str(meta.get("iter2_quality_bucket", ""))
        item["school_quality_reasons"] = ";".join(meta.get("iter2_quality_reasons") or [])
        features = meta.get("_features", {})
        for key in [
            "fg_fraction",
            "skel_fraction",
            "cc_count",
            "dir_h_frac",
            "stroke_width_mean",
            "ruling_response_mean",
            "ruling_response_p95",
        ]:
            item[key] = float(features.get(key, 0.0))
        return_text = str(meta.get("text") or meta.get("normalized_transcription") or item.get("target", ""))
        item["text"] = return_text
        out.append(item)
    return out


def threshold_for_coverage(
    rows: list[dict[str, Any]],
    *,
    target_coverage: float,
) -> float:
    if not rows:
        return 0.0
    risks = np.asarray([float(row["risk"]) for row in rows], dtype=np.float64)
    keep_n = max(1, int(round(len(risks) * target_coverage)))
    risks_sorted = np.sort(risks)
    return float(risks_sorted[min(keep_n - 1, len(risks_sorted) - 1)])


def audit_row(row: dict[str, Any], *, operating_point: str, threshold: float) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "dataset": row.get("dataset"),
        "operating_point": operating_point,
        "threshold": threshold,
        "risk": row["risk"],
        "school_quality_bucket": row.get("school_quality_bucket", ""),
        "school_quality_reasons": row.get("school_quality_reasons", ""),
        "target": row.get("target", row.get("text", "")),
        "pred": row.get("pred", ""),
        "cer": row.get("cer"),
        "wer": row.get("wer"),
        "exact": row.get("exact"),
        "fg_fraction": row.get("fg_fraction"),
        "skel_fraction": row.get("skel_fraction"),
        "cc_count": row.get("cc_count"),
        "dir_h_frac": row.get("dir_h_frac"),
        "stroke_width_mean": row.get("stroke_width_mean"),
        "ruling_response_mean": row.get("ruling_response_mean"),
        "ruling_response_p95": row.get("ruling_response_p95"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--school_quality_root", required=True)
    parser.add_argument("--model", default="plus_10k_context")
    parser.add_argument("--risk_method", default="confidence_graph")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    manifests = {
        "val": load_manifest(Path(args.val_manifest)),
        "test": load_manifest(Path(args.test_manifest)),
    }
    overlay_quality(manifests, Path(args.school_quality_root))

    val_rows_raw = read_jsonl(out_root / "confidence_predictions" / f"{args.model}.val.jsonl")
    test_rows_raw = read_jsonl(out_root / "confidence_predictions" / f"{args.model}.test.jsonl")

    scaler, clf = fit_risk_model(
        val_rows_raw,
        manifests["val"],
        mode=args.risk_method,
    )

    val_rows = attach_risk(
        val_rows_raw,
        manifests["val"],
        predict_risk(
            val_rows_raw,
            manifests["val"],
            scaler,
            clf,
            mode=args.risk_method,
        ),
    )
    test_rows = attach_risk(
        test_rows_raw,
        manifests["test"],
        predict_risk(
            test_rows_raw,
            manifests["test"],
            scaler,
            clf,
            mode=args.risk_method,
        ),
    )

    val_school = [
        row for row in val_rows
        if row["dataset"] == "school_notebooks_clean"
    ]
    test_school = [
        row for row in test_rows
        if row["dataset"] == "school_notebooks_clean"
    ]

    result = {
        "model": args.model,
        "risk_method": args.risk_method,
        "threshold_source": "val school coverage",
        "operating_points": {},
    }

    for name, target_coverage in TARGETS.items():
        threshold = threshold_for_coverage(
            val_school,
            target_coverage=target_coverage,
        )
        result["operating_points"][name] = {
            "target_school_coverage": target_coverage,
            "threshold": threshold,
            "val": {
                "all": metrics(val_rows, threshold=threshold),
                "school": metrics(val_school, threshold=threshold),
            },
            "test": {
                "all": metrics(test_rows, threshold=threshold),
                "school": metrics(test_school, threshold=threshold),
            },
        }

    write_json(out_root / "operating_points.json", result)

    strict_threshold = result["operating_points"]["strict"]["threshold"]
    accepted_errors = [
        audit_row(row, operating_point="strict", threshold=strict_threshold)
        for row in test_rows
        if float(row["risk"]) <= strict_threshold
        and float(row["exact"]) < 1.0
    ]
    rejected_correct = [
        audit_row(row, operating_point="strict", threshold=strict_threshold)
        for row in test_rows
        if float(row["risk"]) > strict_threshold
        and float(row["exact"]) >= 1.0
    ]

    accepted_errors = sorted(
        accepted_errors,
        key=lambda row: (-float(row["cer"]), float(row["risk"]), str(row["sample_id"])),
    )
    rejected_correct = sorted(
        rejected_correct,
        key=lambda row: (-float(row["risk"]), str(row["sample_id"])),
    )

    write_jsonl(
        accepted_errors,
        out_root / "accepted_errors_high_confidence.jsonl",
    )
    write_jsonl(
        rejected_correct,
        out_root / "rejected_correct_low_confidence.jsonl",
    )

    lines = [
        "# Confidence Operating Points",
        "",
        f"Model: `{args.model}`",
        f"Risk method: `{args.risk_method}`",
        "",
        "Thresholds are selected on validation School samples to match target School coverage, then applied unchanged to test.",
        "",
        "## Operating Points",
        "",
        "| point | threshold | split | scope | target School coverage | actual coverage | n accepted | CER | WER | exact |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for point, data in result["operating_points"].items():
        for split in ["val", "test"]:
            for scope in ["school", "all"]:
                row = data[split][scope]
                lines.append(
                    f"| `{point}` | {fmt(data['threshold'])} | `{split}` | `{scope}` | "
                    f"{fmt(data['target_school_coverage'])} | {fmt(row['coverage'])} | {row['n_accepted']} | "
                    f"{fmt(row['cer'])} | {fmt(row['wer'])} | {fmt(row['exact'])} |"
                )

    lines.extend([
        "",
        "## Error Audit",
        "",
        f"- strict accepted errors: {len(accepted_errors)}",
        f"- strict rejected correct: {len(rejected_correct)}",
        "",
        "Files:",
        "- `accepted_errors_high_confidence.jsonl`",
        "- `rejected_correct_low_confidence.jsonl`",
    ])

    (out_root / "operating_points.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "out_root": str(out_root),
        "model": args.model,
        "risk_method": args.risk_method,
        "operating_points": result["operating_points"],
        "accepted_errors_high_confidence": len(accepted_errors),
        "rejected_correct_low_confidence": len(rejected_correct),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
