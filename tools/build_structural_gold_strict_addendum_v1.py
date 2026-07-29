from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOOL_FIELDS = [
    "structural_usable",
    "foreground_ok",
    "skeleton_ok",
    "graph_ok",
]

SEVERITY_FIELDS = [
    "line_residual",
    "neighbor_noise",
    "missed_ink",
    "false_ink",
    "false_branches",
    "broken_strokes",
    "overconnected",
    "segmentation_issue",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def bool_summary(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for row in rows if str(row.get(field, "")).strip() == "1")
    lo, hi = wilson(k, n)
    return {
        "n": n,
        "count": k,
        "rate": k / n if n else None,
        "wilson95_low": lo,
        "wilson95_high": hi,
    }


def severity_summary(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    counts = Counter(int(str(row.get(field, "0") or 0)) for row in rows)
    n = len(rows)
    minor_or_more = counts[1] + counts[2]
    severe = counts[2]
    return {
        "n": n,
        "none": counts[0],
        "minor": counts[1],
        "severe_or_dominant": counts[2],
        "minor_or_more_rate": minor_or_more / n if n else None,
        "severe_or_dominant_rate": severe / n if n else None,
    }


def group_rows(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(field, "") or "unknown")].append(row)
    return dict(out)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--sample_plan", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    annotations = Path(args.annotations)
    sample_plan_path = Path(args.sample_plan)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(annotations)
    sample_plan = json.loads(sample_plan_path.read_text(encoding="utf-8"))

    overall_bool = {
        field: bool_summary(rows, field)
        for field in BOOL_FIELDS
    }
    severity = {
        field: severity_summary(rows, field)
        for field in SEVERITY_FIELDS
    }

    by_dataset = {
        key: {
            field: bool_summary(value, field)
            for field in BOOL_FIELDS
        }
        for key, value in sorted(group_rows(rows, "dataset").items())
    }
    by_stratum = {
        key: {
            field: bool_summary(value, field)
            for field in BOOL_FIELDS
        }
        for key, value in sorted(group_rows(rows, "stratum").items())
    }

    htr_explained = Counter(row.get("htr_error_explained_by_structure", "") for row in rows)
    dataset_counts = Counter(row.get("dataset", "") for row in rows)
    stratum_counts = Counter(row.get("stratum", "") for row in rows)
    token_type_counts = Counter(row.get("token_type", "") for row in rows)

    result = {
        "annotations": str(annotations),
        "sample_plan": str(sample_plan_path),
        "n": len(rows),
        "sample_plan_summary": sample_plan,
        "dataset_counts": dict(dataset_counts),
        "stratum_counts": dict(stratum_counts),
        "token_type_counts": dict(token_type_counts),
        "overall_bool": overall_bool,
        "severity": severity,
        "by_dataset": by_dataset,
        "by_stratum": by_stratum,
        "htr_error_explained_by_structure": dict(htr_explained),
        "publication_interpretation": (
            "This addendum supports diagnostic usability only. It does not establish "
            "pixel-level or graph-topology ground truth because annotation fields are "
            "coarse binary/severity judgments and no inter-annotator agreement is available."
        ),
    }

    (out_dir / "structural_gold_strict_addendum_v1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Structural Gold Strict Addendum v1",
        "",
        "This addendum tightens the reporting around the existing 200-sample structural gold subset.",
        "It does not convert the subset into a topology benchmark.",
        "",
        "## Sample Composition",
        "",
        f"- n: {len(rows)}",
        f"- datasets: {dict(dataset_counts)}",
        f"- strata: {dict(stratum_counts)}",
        f"- token types: {dict(token_type_counts)}",
        "",
        "## Binary Diagnostic Usability Metrics",
        "",
        "| field | count | n | rate | Wilson 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, row in overall_bool.items():
        lines.append(
            f"| `{field}` | {row['count']} | {row['n']} | {fmt(row['rate'])} | "
            f"[{fmt(row['wilson95_low'])}, {fmt(row['wilson95_high'])}] |"
        )

    lines.extend([
        "",
        "## Severity Profile",
        "",
        "| issue | none | minor | severe/dominant | minor+ rate | severe rate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for field, row in severity.items():
        lines.append(
            f"| `{field}` | {row['none']} | {row['minor']} | {row['severe_or_dominant']} | "
            f"{fmt(row['minor_or_more_rate'])} | {fmt(row['severe_or_dominant_rate'])} |"
        )

    lines.extend([
        "",
        "## Dataset-Level Caution",
        "",
        "| dataset | n | structural usable | foreground ok | skeleton ok | graph ok |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dataset, fields in by_dataset.items():
        lines.append(
            f"| `{dataset}` | {fields['structural_usable']['n']} | "
            f"{fmt(fields['structural_usable']['rate'])} | "
            f"{fmt(fields['foreground_ok']['rate'])} | "
            f"{fmt(fields['skeleton_ok']['rate'])} | "
            f"{fmt(fields['graph_ok']['rate'])} |"
        )

    lines.extend([
        "",
        "## HTR Error Attribution",
        "",
        f"`htr_error_explained_by_structure`: {dict(htr_explained)}",
        "",
        "## Strict Interpretation",
        "",
        "- The subset supports diagnostic usability on the sampled cases.",
        "- It does not prove exact topology recovery, endpoint/junction correctness, stroke order, or pen trajectory.",
        "- No inter-annotator agreement is available; this remains a major publication limitation.",
        "- The subset is dominated by School Notebooks, so HKR/Cyrillic conclusions are weak at dataset level.",
    ])

    (out_dir / "structural_gold_strict_addendum_v1.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "out_dir": str(out_dir),
        "n": len(rows),
        "overall_bool": overall_bool,
        "severity": severity,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
