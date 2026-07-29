from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def graph_feature_dict(row: dict[str, Any]) -> dict[str, float]:
    names = row.get("graph_feature_names") or []
    values = row.get("graph_features") or []

    return {
        str(name): float(value)
        for name, value in zip(names, values)
    }


def text_len(row: dict[str, Any], features: dict[str, float]) -> int:
    if "text_len" in features:
        return int(features["text_len"])

    for key in ["text", "normalized_transcription", "raw_transcription"]:
        value = row.get(key)
        if value is not None:
            return len(str(value))

    return 0


def load_diagnostics(graph_root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = graph_root / "diagnostics" / f"{split}.jsonl"

    if not path.exists():
        return {}

    return {
        str(row.get("sample_id", "")): row
        for row in read_jsonl(path)
        if row.get("sample_id")
    }


def foreground_diagnostics(
    row: dict[str, Any],
    diagnostics_by_id: dict[str, dict[str, Any]],
) -> dict[str, float]:
    sample_id = str(row.get("sample_id", ""))
    diag = diagnostics_by_id.get(sample_id, {})
    row_diag = row.get("school_foreground_diagnostics") or {}

    if not isinstance(row_diag, dict):
        row_diag = {}

    return {
        "ruling_response_mean": float(
            row_diag.get(
                "ruling_response_mean",
                diag.get("ruling_response_mean", row.get("ruling_response_mean", 0.0)),
            )
        ),
        "ruling_response_p95": float(
            row_diag.get(
                "ruling_response_p95",
                diag.get("ruling_response_p95", row.get("ruling_response_p95", 0.0)),
            )
        ),
    }


def classify_school_sample(
    row: dict[str, Any],
    diagnostics_by_id: dict[str, dict[str, Any]],
) -> tuple[str, list[str], dict[str, float], dict[str, float]]:
    features = graph_feature_dict(row)
    fg_diag = foreground_diagnostics(row, diagnostics_by_id)

    fg = float(features.get("fg_fraction", 0.0))
    skel = float(features.get("skel_fraction", 0.0))
    cc = float(features.get("cc_count", 0.0))
    dir_h = float(features.get("dir_h_frac", 0.0))
    sw = float(features.get("stroke_width_mean", 0.0))
    warnings = float(features.get("warning_count", row.get("graph_warning_count", 0.0)))
    ruling = float(fg_diag["ruling_response_mean"])
    tl = text_len(row, features)

    reasons: list[str] = []

    invalid_conditions = [
        ("fg_too_low_invalid", fg < 0.015),
        ("fg_too_high_invalid", fg > 0.28),
        ("skel_too_low_invalid", skel < 0.003),
        ("cc_extreme_invalid", cc > 45),
        ("stroke_width_extreme_invalid", sw > 8.0),
    ]

    for reason, condition in invalid_conditions:
        if condition:
            reasons.append(reason)

    if reasons:
        return "invalid_or_review", reasons, features, fg_diag

    hard_conditions = [
        ("fg_low", fg < 0.055),
        ("fg_high", fg > 0.175),
        ("skel_low", skel < 0.015),
        ("skel_high", skel > 0.065),
        ("cc_high", cc > 12),
        ("dir_h_high", dir_h > 0.50),
        ("stroke_width_high", sw > 5.0),
        ("ruling_high", ruling > 22.0),
        ("text_len_short", tl <= 1),
        ("has_graph_warning", warnings > 0),
    ]

    for reason, condition in hard_conditions:
        if condition:
            reasons.append(reason)

    if reasons:
        return "hard_real", reasons, features, fg_diag

    return "clean_core", [], features, fg_diag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    graph_root = Path(args.graph_root)
    out_root = Path(args.out_root)
    out_csv = Path(args.out_csv)

    csv_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "graph_root": str(graph_root),
        "out_root": str(out_root),
        "splits": {},
        "rules": {
            "clean_core": {
                "warning_count": "== 0",
                "fg_fraction": [0.055, 0.175],
                "skel_fraction": [0.015, 0.065],
                "cc_count": "<= 12",
                "dir_h_frac": "<= 0.50",
                "stroke_width_mean": "<= 5.0",
                "ruling_response_mean": "<= 22.0",
                "text_len": ">= 2",
            },
            "invalid_or_review": {
                "fg_fraction": ["< 0.015", "> 0.28"],
                "skel_fraction": "< 0.003",
                "cc_count": "> 45",
                "stroke_width_mean": "> 8.0",
            },
        },
    }

    for split in ["train", "val", "test"]:
        rows = read_jsonl(graph_root / f"{split}.jsonl")
        diagnostics_by_id = load_diagnostics(graph_root, split)

        buckets = {
            "clean_core": [],
            "hard_real": [],
            "invalid_or_review": [],
        }

        reason_counts = Counter()

        for row in rows:
            if row.get("dataset") != "school_notebooks_clean":
                continue

            label, reasons, features, fg_diag = classify_school_sample(
                row,
                diagnostics_by_id,
            )

            out_row = dict(row)
            out_row["iter2_quality_bucket"] = label
            out_row["iter2_quality_reasons"] = reasons
            out_row["school_foreground_diagnostics"] = fg_diag

            buckets[label].append(out_row)

            for reason in reasons:
                reason_counts[reason] += 1

            csv_rows.append({
                "sample_id": row.get("sample_id", ""),
                "split": split,
                "bucket": label,
                "reasons": ";".join(reasons),
                "text": row.get("text", ""),
                "fg_fraction": features.get("fg_fraction", ""),
                "skel_fraction": features.get("skel_fraction", ""),
                "cc_count": features.get("cc_count", ""),
                "dir_h_frac": features.get("dir_h_frac", ""),
                "stroke_width_mean": features.get("stroke_width_mean", ""),
                "ruling_response_mean": fg_diag.get("ruling_response_mean", ""),
                "ruling_response_p95": fg_diag.get("ruling_response_p95", ""),
                "warning_count": features.get("warning_count", row.get("graph_warning_count", "")),
            })

        for bucket, bucket_rows in buckets.items():
            write_jsonl(
                bucket_rows,
                out_root / f"{split}.{bucket}.jsonl",
            )

        split_n = sum(len(v) for v in buckets.values())

        summary["splits"][split] = {
            "n": split_n,
            "bucket_counts": {
                bucket: len(bucket_rows)
                for bucket, bucket_rows in buckets.items()
            },
            "bucket_rates": {
                bucket: len(bucket_rows) / max(split_n, 1)
                for bucket, bucket_rows in buckets.items()
            },
            "reason_counts": dict(reason_counts),
        }

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "split",
        "bucket",
        "reasons",
        "text",
        "fg_fraction",
        "skel_fraction",
        "cc_count",
        "dir_h_frac",
        "stroke_width_mean",
        "ruling_response_mean",
        "ruling_response_p95",
        "warning_count",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
