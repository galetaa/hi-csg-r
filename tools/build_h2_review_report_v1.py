from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOOL_FIELDS = [
    "foreground_ok",
    "skeleton_usable",
    "graph_usable",
    "components_reasonable",
    "endpoints_reasonable",
    "junctions_reasonable",
    "usable_for_diagnostics",
]

OPTIONAL_BOOL_FIELDS = [
    "loops_preserved",
]

NEGATIVE_FIELDS = [
    "severe_topology_error",
]

MINIMAL_THRESHOLDS = {
    "foreground_ok": 0.90,
    "skeleton_usable": 0.85,
    "graph_usable": 0.80,
    "usable_for_diagnostics": 0.80,
    "severe_topology_error": 0.20,
}

STRONG_THRESHOLDS = {
    "foreground_ok": 0.95,
    "skeleton_usable": 0.90,
    "graph_usable": 0.85,
    "usable_for_diagnostics": 0.85,
    "severe_topology_error": 0.15,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value: Any) -> int | None:
    if value is None:
        return None

    s = str(value).strip().lower()

    if s in {"1", "true", "yes", "y", "да", "ok"}:
        return 1

    if s in {"0", "false", "no", "n", "нет", "bad"}:
        return 0

    if s in {"na", "n/a", "none", "", "-"}:
        return None

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def rate(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    vals: list[int] = []
    for row in rows:
        v = parse_bool(row.get(field))
        if v is not None:
            vals.append(v)

    return {
        "field": field,
        "n_valid": len(vals),
        "rate": sum(vals) / len(vals) if vals else None,
        "count_1": sum(vals),
        "count_0": len(vals) - sum(vals),
    }


def group_by(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(field, "unknown") or "unknown")].append(row)
    return dict(out)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def threshold_status(rates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    minimal = {}
    strong = {}

    for field, threshold in MINIMAL_THRESHOLDS.items():
        value = rates.get(field, {}).get("rate")
        if value is None:
            minimal[field] = False
        elif field in NEGATIVE_FIELDS:
            minimal[field] = value <= threshold
        else:
            minimal[field] = value >= threshold

    for field, threshold in STRONG_THRESHOLDS.items():
        value = rates.get(field, {}).get("rate")
        if value is None:
            strong[field] = False
        elif field in NEGATIVE_FIELDS:
            strong[field] = value <= threshold
        else:
            strong[field] = value >= threshold

    return {
        "minimal": minimal,
        "strong": strong,
        "minimal_supported": all(minimal.values()),
        "strongly_supported": all(strong.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    review_csv = Path(args.review_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(review_csv)

    overall_rates_list: list[dict[str, Any]] = []

    for field in BOOL_FIELDS + OPTIONAL_BOOL_FIELDS + NEGATIVE_FIELDS:
        overall_rates_list.append(rate(rows, field))

    overall_rates = {
        row["field"]: row
        for row in overall_rates_list
    }

    dataset_rows: list[dict[str, Any]] = []
    for dataset, group in group_by(rows, "dataset").items():
        for field in BOOL_FIELDS + OPTIONAL_BOOL_FIELDS + NEGATIVE_FIELDS:
            r = rate(group, field)
            dataset_rows.append(
                {
                    "dataset": dataset,
                    **r,
                }
            )

    failure_counter = Counter()
    for row in rows:
        ft = str(row.get("failure_type", "") or "none").strip()
        if not ft:
            ft = "none"
        failure_counter[ft] += 1

    failure_rows = [
        {
            "failure_type": k,
            "count": v,
            "rate": v / max(1, len(rows)),
        }
        for k, v in failure_counter.most_common()
    ]

    status = threshold_status(overall_rates)

    summary = {
        "review_csv": str(review_csv),
        "n": len(rows),
        "overall_rates": overall_rates_list,
        "threshold_status": status,
        "failure_taxonomy": failure_rows,
        "interpretation_guardrails": [
            "H2 does not claim exact pen trajectory recovery.",
            "H2 does not claim topology-perfect graph reconstruction.",
            "H2 is supported only as structural usability / diagnostic validity if graph_usable and usable_for_diagnostics are high enough.",
        ],
    }

    (out_dir / "h2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(overall_rates_list, out_dir / "h2_review_aggregate.csv")
    write_csv(dataset_rows, out_dir / "h2_dataset_breakdown.csv")
    write_csv(failure_rows, out_dir / "h2_failure_taxonomy.csv")

    md_lines = []
    md_lines.append("# H2 structural validity closure")
    md_lines.append("")
    md_lines.append(f"Review samples: **{len(rows)}**")
    md_lines.append("")
    md_lines.append("## Overall rates")
    md_lines.append("")
    md_lines.append("| field | n valid | rate | count 1 | count 0 |")
    md_lines.append("|---|---:|---:|---:|---:|")

    for r in overall_rates_list:
        value = r["rate"]
        value_s = "NA" if value is None else f"{value:.4f}"
        md_lines.append(
            f"| {r['field']} | {r['n_valid']} | {value_s} | {r['count_1']} | {r['count_0']} |"
        )

    md_lines.append("")
    md_lines.append("## Acceptance status")
    md_lines.append("")
    md_lines.append(f"- minimally supported: `{status['minimal_supported']}`")
    md_lines.append(f"- strongly supported: `{status['strongly_supported']}`")
    md_lines.append("")

    md_lines.append("## Failure taxonomy")
    md_lines.append("")
    md_lines.append("| failure type | count | rate |")
    md_lines.append("|---|---:|---:|")
    for r in failure_rows:
        md_lines.append(f"| {r['failure_type']} | {r['count']} | {r['rate']:.4f} |")

    md_lines.append("")
    md_lines.append("## Interpretation")
    md_lines.append("")
    md_lines.append(
        "H2 should be interpreted as structural usability of HI-CSG-R, not as exact recovery of pen trajectory or perfect topology."
    )

    (out_dir / "h2_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()
