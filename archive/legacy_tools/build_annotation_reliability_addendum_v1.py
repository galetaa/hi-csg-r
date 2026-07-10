from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PILOT_40 = Path("outputs/h2_gold_audit_v1/annotations/annotation_pilot_40_filled.csv")
ANNOTATION_100 = Path("outputs/h2_gold_audit_v1/annotations/annotation_100_filled.csv")
INDEPENDENT_ROOT = Path("outputs/htr_publication_v3/independent_annotation_v1")
INDEPENDENT_MANIFEST = INDEPENDENT_ROOT / "blind_annotation_manifest.json"
INDEPENDENT_SCORE = INDEPENDENT_ROOT / "scored" / "summary.json"
INDEPENDENT_REPORT = INDEPENDENT_ROOT / "scored" / "report.md"

LINE_AUDITS = {
    "full_natural_lines_150": {
        "path": Path("outputs/iter2_data_audit/school_notebooks_v1/full_natural_lines_v1/annotations_filled.csv"),
        "positive_fields": ["valid_line", "correct_order", "complete_enough", "good_for_line_train"],
        "negative_fields": ["neighbor_noise"],
    },
    "rendered_line_sanity_80": {
        "path": Path("outputs/iter2_data_audit/school_notebooks_v1/rendered_line_sanity_v1/annotations_filled.csv"),
        "positive_fields": ["readable", "correct_crop", "good_for_htr"],
        "negative_fields": [],
    },
    "natural_lines_120": {
        "path": Path("outputs/iter2_data_audit/school_notebooks_v1/natural_lines_v1/natural_line_validation_annotations_filled.csv"),
        "positive_fields": ["valid_line", "correct_order", "good_for_train_aug"],
        "negative_fields": ["missing_words", "neighbor_noise"],
    },
    "lineaware_quality_gate_120": {
        "path": Path("outputs/iter2_data_audit/school_notebooks_v1/lineaware_quality_gate/lineaware_quality_gate_annotations.csv"),
        "positive_fields": ["usable", "skeleton_follows_ink"],
        "negative_fields": ["ink_loss", "line_residual"],
    },
}

REPEATED_FIELDS = [
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


def maybe_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def clean(value: Any) -> str | None:
    s = str(value if value is not None else "").strip()
    return s if s != "" else None


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({v for pair in pairs for v in pair})
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


def repeated_annotation_consistency() -> dict[str, Any]:
    pilot = {row["sample_id"]: row for row in read_csv(PILOT_40)}
    full = {row["sample_id"]: row for row in read_csv(ANNOTATION_100)}
    overlap_ids = sorted(set(pilot) & set(full))
    fields = {}
    disagreements = []

    for field in REPEATED_FIELDS:
        pairs_str: list[tuple[str, str]] = []
        pairs_int: list[tuple[int, int]] = []
        for sample_id in overlap_ids:
            a = clean(pilot[sample_id].get(field))
            b = clean(full[sample_id].get(field))
            if a is None or b is None:
                continue
            pairs_str.append((a, b))
            if field == "graph_quality_0_3":
                try:
                    pairs_int.append((int(a), int(b)))
                except ValueError:
                    pass
            if a != b:
                disagreements.append({
                    "sample_id": sample_id,
                    "field": field,
                    "pilot_40": a,
                    "annotation_100": b,
                })
        n = len(pairs_str)
        agree = sum(1 for a, b in pairs_str if a == b)
        lo, hi = wilson(agree, n)
        fields[field] = {
            "n_overlap_nonempty": n,
            "agreement_count": agree,
            "agreement_rate": agree / n if n else None,
            "agreement_wilson95_low": lo,
            "agreement_wilson95_high": hi,
            "cohen_kappa": cohen_kappa(pairs_str),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(pairs_int) if field == "graph_quality_0_3" else None,
        }

    return {
        "pilot_40": str(PILOT_40),
        "annotation_100": str(ANNOTATION_100),
        "overlap_n": len(overlap_ids),
        "overlap_sample_ids": overlap_ids,
        "fields": fields,
        "disagreement_examples": disagreements[:50],
        "interpretation": (
            "This is repeated-annotation consistency on overlapping samples. It is useful reliability evidence, "
            "but it is not a formal inter-annotator agreement unless the two files are confirmed to be independent annotators."
        ),
    }


def field_rate(rows: list[dict[str, str]], field: str, positive_value: str = "1") -> dict[str, Any]:
    vals = [clean(row.get(field)) for row in rows]
    vals = [value for value in vals if value is not None]
    n = len(vals)
    k = sum(1 for value in vals if value == positive_value)
    lo, hi = wilson(k, n)
    return {
        "n": n,
        "count": k,
        "rate": k / n if n else None,
        "wilson95_low": lo,
        "wilson95_high": hi,
    }


def line_quality_audits() -> dict[str, Any]:
    out = {}
    for name, cfg in LINE_AUDITS.items():
        path = cfg["path"]
        rows = read_csv(path)
        positive = {field: field_rate(rows, field) for field in cfg["positive_fields"]}
        negative = {field: field_rate(rows, field) for field in cfg["negative_fields"]}
        out[name] = {
            "path": str(path),
            "n_rows": len(rows),
            "positive_fields": positive,
            "negative_issue_fields": negative,
        }
    return out


def independent_annotation_status() -> dict[str, Any]:
    manifest = maybe_json(INDEPENDENT_MANIFEST)
    score = maybe_json(INDEPENDENT_SCORE)
    expected_filled_csv = None
    browser = None
    template_csv = None
    protocol = None
    if manifest is not None:
        expected_filled_csv = manifest.get("expected_filled_csv")
        browser = manifest.get("browser")
        template_csv = manifest.get("template_csv")
        protocol = manifest.get("protocol")

    package_ready = (
        manifest is not None
        and browser is not None
        and Path(browser).exists()
        and template_csv is not None
        and Path(template_csv).exists()
    )
    second_exists = bool(expected_filled_csv and Path(expected_filled_csv).exists())
    formal_ready = bool(score and score.get("is_formal_iaa_ready"))

    return {
        "package_root": str(INDEPENDENT_ROOT),
        "manifest": str(INDEPENDENT_MANIFEST),
        "score_summary": str(INDEPENDENT_SCORE),
        "score_report": str(INDEPENDENT_REPORT),
        "browser": browser,
        "template_csv": template_csv,
        "protocol": protocol,
        "expected_filled_csv": expected_filled_csv,
        "manifest_data": manifest,
        "score_data": score,
        "package_ready": package_ready,
        "second_csv_exists": second_exists,
        "minimally_complete_rows": score.get("minimally_complete_rows") if score else None,
        "matched_sample_ids": score.get("matched_sample_ids") if score else None,
        "formal_iaa_ready": formal_ready,
        "interpretation": (
            "The blind second-annotation package is prepared. It becomes formal IAA evidence only after "
            "a genuinely independent annotator fills the expected CSV and the scorer reports adequate overlap."
        ),
    }


def build_summary() -> dict[str, Any]:
    repeated = repeated_annotation_consistency()
    audits = line_quality_audits()
    independent = independent_annotation_status()
    weak_fields = [
        {
            "field": field,
            **metrics,
        }
        for field, metrics in repeated["fields"].items()
        if metrics.get("cohen_kappa") is not None and float(metrics["cohen_kappa"]) < 0.6
    ]
    return {
        "package": "annotation_reliability_addendum_v1",
        "repeated_annotation_consistency": repeated,
        "independent_annotation_v1": independent,
        "line_quality_audits": audits,
        "publication_interpretation": {
            "strongest_supported_claim": (
                "The existing filled audits provide quantitative quality-control rates and repeated-annotation consistency "
                "for a 40-sample overlap. A blind second-annotation package is prepared but is not itself agreement evidence."
            ),
            "not_supported": (
                "A formal inter-annotator agreement claim is not supported until a genuinely independent second annotator "
                "fills the blind package and the scoring report shows adequate agreement."
                if not independent["formal_iaa_ready"]
                else "Formal IAA metrics are available, but annotator independence still must be documented in the paper."
            ),
            "weak_reliability_fields": weak_fields,
            "formal_iaa_ready": independent["formal_iaa_ready"],
            "remaining_requirement": (
                "Have a second independent annotator fill the blind package and rerun `python tools/score_independent_annotation_v1.py`."
                if not independent["formal_iaa_ready"]
                else "Report the independent-annotation scorer output and document annotator independence."
            ),
        },
    }


def build_md(summary: dict[str, Any]) -> str:
    repeated = summary["repeated_annotation_consistency"]
    lines = [
        "# Annotation Reliability Addendum v1",
        "",
        repeated["interpretation"],
        "",
        "## Repeated Annotation Consistency",
        "",
        f"- overlap n: {repeated['overlap_n']}",
        f"- pilot file: `{repeated['pilot_40']}`",
        f"- comparison file: `{repeated['annotation_100']}`",
        "",
        "| field | n | agreement | Wilson 95% CI | Cohen kappa | weighted kappa |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field, metrics in repeated["fields"].items():
        lines.append(
            f"| `{field}` | {metrics['n_overlap_nonempty']} | {fmt(metrics['agreement_rate'])} | "
            f"[{fmt(metrics['agreement_wilson95_low'])}, {fmt(metrics['agreement_wilson95_high'])}] | "
            f"{fmt(metrics['cohen_kappa'])} | {fmt(metrics['quadratic_weighted_kappa'])} |"
        )

    independent = summary["independent_annotation_v1"]
    lines.extend([
        "",
        "## Independent Blind Second Annotation Package",
        "",
        independent["interpretation"],
        "",
        f"- package ready: {independent['package_ready']}",
        f"- browser: `{independent['browser']}`",
        f"- template CSV: `{independent['template_csv']}`",
        f"- expected filled CSV: `{independent['expected_filled_csv']}`",
        f"- score report: `{independent['score_report']}`",
        f"- second CSV exists: {independent['second_csv_exists']}",
        f"- minimally complete rows: {independent['minimally_complete_rows']}",
        f"- formal IAA ready: {independent['formal_iaa_ready']}",
    ])

    lines.extend([
        "",
        "## Line Quality Audit Rates",
        "",
        "| audit | field | direction | count/n | rate | Wilson 95% CI |",
        "|---|---|---|---:|---:|---:|",
    ])
    for audit_name, audit in summary["line_quality_audits"].items():
        for direction, group_name in [("positive", "positive_fields"), ("issue", "negative_issue_fields")]:
            for field, metrics in audit[group_name].items():
                lines.append(
                    f"| `{audit_name}` | `{field}` | {direction} | {metrics['count']}/{metrics['n']} | "
                    f"{fmt(metrics['rate'])} | [{fmt(metrics['wilson95_low'])}, {fmt(metrics['wilson95_high'])}] |"
                )

    interp = summary["publication_interpretation"]
    lines.extend([
        "",
        "## Publication Interpretation",
        "",
        f"- Strongest supported claim: {interp['strongest_supported_claim']}",
        f"- Not supported: {interp['not_supported']}",
        f"- Remaining requirement: {interp['remaining_requirement']}",
    ])
    if interp["weak_reliability_fields"]:
        lines.append("")
        lines.append("Fields with kappa below 0.6:")
        for row in interp["weak_reliability_fields"]:
            lines.append(f"- `{row['field']}`: kappa={fmt(row['cohen_kappa'])}, agreement={fmt(row['agreement_rate'])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/htr_publication_v3/annotation_reliability_addendum_v1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(build_md(summary), encoding="utf-8")
    print(json.dumps({
        "out_json": str(out_dir / "summary.json"),
        "out_md": str(out_dir / "report.md"),
        "overlap_n": summary["repeated_annotation_consistency"]["overlap_n"],
        "weak_fields": [
            row["field"]
            for row in summary["publication_interpretation"]["weak_reliability_fields"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
