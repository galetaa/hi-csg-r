from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


REFERENCE = Path("outputs/h2_gold_audit_v1/annotations/annotation_100_filled.csv")
SECOND = Path("outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_second_filled.csv")
OUT_DIR = Path("outputs/htr_publication_v3/independent_annotation_v1/scored")

FIELDS = [
    "audit_usable",
    "ink_visible_ok",
    "skeleton_follows_ink",
    "missed_visible_stroke",
    "spurious_stroke",
    "endpoint_error",
    "junction_error",
    "loop_error",
    "critical_topology_error",
    "graph_quality_0_3",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def clean(value: Any) -> str | None:
    s = str(value if value is not None else "").strip()
    if s.lower() in {"", "nan", "none"}:
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({value for pair in pairs for value in pair})
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    pe = sum((left[label] / n) * (right[label] / n) for label in labels)
    if abs(1 - pe) < 1e-12:
        return 1.0 if abs(1 - po) < 1e-12 else None
    return (po - pe) / (1 - pe)


def quadratic_weighted_kappa(pairs: list[tuple[int, int]], min_rating: int = 0, max_rating: int = 3) -> float | None:
    if not pairs:
        return None
    ratings = list(range(min_rating, max_rating + 1))
    n = len(pairs)
    observed = {(i, j): 0 for i in ratings for j in ratings}
    left = Counter()
    right = Counter()
    for a, b in pairs:
        observed[(a, b)] += 1
        left[a] += 1
        right[b] += 1

    denom_rating = (max_rating - min_rating) ** 2
    if denom_rating == 0:
        return None

    observed_weighted = 0.0
    expected_weighted = 0.0
    for i in ratings:
        for j in ratings:
            weight = ((i - j) ** 2) / denom_rating
            observed_weighted += weight * observed[(i, j)] / n
            expected_weighted += weight * (left[i] * right[j]) / (n * n)
    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else None
    return 1 - observed_weighted / expected_weighted


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def compare(reference_path: Path, second_path: Path) -> dict[str, Any]:
    ref_rows = {row["sample_id"]: row for row in read_csv(reference_path)}
    second_rows = read_csv(second_path) if second_path.exists() else []
    second_by_id = {row["sample_id"]: row for row in second_rows if clean(row.get("sample_id"))}

    sample_ids = sorted(set(ref_rows) & set(second_by_id))
    field_metrics = {}
    disagreements = []
    completion_by_row = []

    for sample_id in sample_ids:
        row = second_by_id[sample_id]
        filled_required = [
            field for field in ["audit_usable", "critical_topology_error", "graph_quality_0_3"]
            if clean(row.get(field)) is not None
        ]
        completion_by_row.append({
            "sample_id": sample_id,
            "filled_required_n": len(filled_required),
            "is_minimally_complete": len(filled_required) == 3,
        })

    for field in FIELDS:
        pairs_str: list[tuple[str, str]] = []
        pairs_int: list[tuple[int, int]] = []
        missing_reference = 0
        missing_second = 0
        for sample_id in sample_ids:
            a = clean(ref_rows[sample_id].get(field))
            b = clean(second_by_id[sample_id].get(field))
            if a is None:
                missing_reference += 1
                continue
            if b is None:
                missing_second += 1
                continue
            pairs_str.append((a, b))
            if a != b:
                disagreements.append({
                    "sample_id": sample_id,
                    "field": field,
                    "reference": a,
                    "second": b,
                })
            if field == "graph_quality_0_3":
                try:
                    pairs_int.append((int(a), int(b)))
                except ValueError:
                    pass

        n = len(pairs_str)
        agree = sum(1 for a, b in pairs_str if a == b)
        lo, hi = wilson(agree, n)
        field_metrics[field] = {
            "n_compared": n,
            "agreement_count": agree,
            "agreement_rate": agree / n if n else None,
            "agreement_wilson95_low": lo,
            "agreement_wilson95_high": hi,
            "cohen_kappa": cohen_kappa(pairs_str),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(pairs_int) if field == "graph_quality_0_3" else None,
            "missing_reference": missing_reference,
            "missing_second": missing_second,
        }

    complete_rows = sum(1 for row in completion_by_row if row["is_minimally_complete"])
    return {
        "package": "independent_annotation_v1_score",
        "reference": str(reference_path),
        "second": str(second_path),
        "reference_rows": len(ref_rows),
        "second_rows": len(second_rows),
        "matched_sample_ids": len(sample_ids),
        "minimally_complete_rows": complete_rows,
        "is_formal_iaa_ready": complete_rows >= 40 and any(
            metrics["n_compared"] >= 40 for metrics in field_metrics.values()
        ),
        "field_metrics": field_metrics,
        "disagreement_examples": disagreements[:100],
        "completion_by_row": completion_by_row,
        "publication_interpretation": {
            "supported_if_independent": (
                "If the second CSV was filled by a genuinely independent annotator, these metrics support "
                "formal inter-annotator agreement reporting for fields with adequate n."
            ),
            "not_supported_if_same_annotator_or_ai": (
                "If the second CSV was filled by the same annotator or by an AI assistant, report this as "
                "repeated/AI consistency rather than formal independent IAA."
            ),
        },
    }


def build_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Independent Annotation v1 Score",
        "",
        f"- reference: `{summary['reference']}`",
        f"- second: `{summary['second']}`",
        f"- reference rows: {summary['reference_rows']}",
        f"- second rows: {summary['second_rows']}",
        f"- matched sample ids: {summary['matched_sample_ids']}",
        f"- minimally complete rows: {summary['minimally_complete_rows']}",
        f"- formal IAA ready: {summary['is_formal_iaa_ready']}",
        "",
        "## Agreement",
        "",
        "| field | n | agreement | Wilson 95% CI | Cohen kappa | weighted kappa | missing second |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field, metrics in summary["field_metrics"].items():
        lines.append(
            f"| `{field}` | {metrics['n_compared']} | {fmt(metrics['agreement_rate'])} | "
            f"[{fmt(metrics['agreement_wilson95_low'])}, {fmt(metrics['agreement_wilson95_high'])}] | "
            f"{fmt(metrics['cohen_kappa'])} | {fmt(metrics['quadratic_weighted_kappa'])} | "
            f"{metrics['missing_second']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- {summary['publication_interpretation']['supported_if_independent']}",
        f"- {summary['publication_interpretation']['not_supported_if_same_annotator_or_ai']}",
    ])

    if summary["disagreement_examples"]:
        lines.extend([
            "",
            "## Disagreement Examples",
            "",
            "| sample_id | field | reference | second |",
            "|---|---|---|---|",
        ])
        for row in summary["disagreement_examples"][:30]:
            lines.append(
                f"| `{row['sample_id']}` | `{row['field']}` | {row['reference']} | {row['second']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=str(REFERENCE))
    parser.add_argument("--second", default=str(SECOND))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = compare(Path(args.reference), Path(args.second))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(build_md(summary), encoding="utf-8")
    print(json.dumps({
        "out_json": str(out_dir / "summary.json"),
        "out_md": str(out_dir / "report.md"),
        "second_exists": Path(args.second).exists(),
        "minimally_complete_rows": summary["minimally_complete_rows"],
        "formal_iaa_ready": summary["is_formal_iaa_ready"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
