from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METRICS_CSV = Path("outputs/graph_pilot_v2/graph_quality_metrics_v1.csv")
OUT_CLEAN = Path("outputs/graph_pilot_v2/clean_graph_subset_v1.jsonl")
OUT_STRESS = Path("outputs/graph_pilot_v2/page_stress_graph_subset_v1.jsonl")
OUT_REVIEW = Path("outputs/graph_pilot_v2/review_graph_subset_v1.jsonl")
OUT_SUMMARY = Path("outputs/graph_pilot_v2/clean_graph_subset_v1_summary.json")


CLEAN_DATASETS = {
    "iam",
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks",
}

PAGE_STRESS_DATASETS = {
    "hwr200",
    "hkr_forms",
}

PRIMARY_VARIANTS = {
    "iam": "otsu",
    "cyrillic_handwriting": "otsu",
    "hkr_words": "otsu",
    "school_notebooks": "sauvola",
    "hwr200": "otsu_gridless",
    "hkr_forms": "otsu_gridless",
}


def parse_float(value: Any) -> float | None:
    if value is None:
        return None

    s = str(value).strip()

    if not s or s.lower() in {"none", "nan"}:
        return None

    return float(s)


def parse_int(value: Any) -> int | None:
    x = parse_float(value)
    if x is None:
        return None
    return int(round(x))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            row = dict(raw)

            for key in [
                "foreground_ratio",
                "skeleton_density",
                "nodes_per_1k_skeleton",
                "edges_per_1k_skeleton",
                "components_per_1k_skeleton",
                "junctions_per_1k_skeleton",
                "endpoints_per_1k_skeleton",
                "junction_endpoint_ratio",
                "edge_node_ratio",
                "component_node_ratio",
                "mean_width_proxy",
                "warning_risk_score",
            ]:
                row[key] = parse_float(row.get(key))

            for key in [
                "image_width",
                "image_height",
                "image_area",
                "skeleton_pixels",
                "node_count",
                "edge_count",
                "component_count",
                "junction_count",
                "endpoint_count",
                "short_branch_count",
                "loop_candidate_count",
                "warning_count",
            ]:
                row[key] = parse_int(row.get(key))

            row["is_primary_variant"] = parse_bool(row.get("is_primary_variant"))
            row["is_page_stress_dataset"] = parse_bool(row.get("is_page_stress_dataset"))
            row["warnings_list"] = [
                w for w in str(row.get("warnings", "")).split("|") if w
            ]

            rows.append(row)

    return rows


def is_primary(row: dict[str, Any]) -> bool:
    return PRIMARY_VARIANTS.get(row["dataset"]) == row["method"]


def clean_acceptance(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []

    dataset = row["dataset"]

    if dataset not in CLEAN_DATASETS:
        reasons.append("not_clean_dataset")

    if not is_primary(row):
        reasons.append("not_primary_variant")

    if row.get("warning_count") not in {0, None}:
        reasons.append("has_warnings")

    if (row.get("warning_risk_score") or 0) > 0:
        reasons.append("warning_risk_positive")

    fg = row.get("foreground_ratio")
    if fg is None or fg < 0.003:
        reasons.append("foreground_too_low")
    elif fg > 0.40:
        reasons.append("foreground_too_high")

    skel = row.get("skeleton_pixels")
    if skel is None or skel < 20:
        reasons.append("skeleton_too_small")
    elif skel > 50000:
        reasons.append("skeleton_too_large_for_clean_crop")

    comp_density = row.get("components_per_1k_skeleton")
    if comp_density is not None and comp_density > 120:
        reasons.append("component_density_too_high")

    junction_density = row.get("junctions_per_1k_skeleton")
    if junction_density is not None and junction_density > 120:
        reasons.append("junction_density_too_high")

    return len(reasons) == 0, reasons


def stress_acceptance(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []

    dataset = row["dataset"]

    if dataset not in PAGE_STRESS_DATASETS:
        reasons.append("not_page_stress_dataset")

    if not is_primary(row):
        reasons.append("not_primary_variant")

    skel = row.get("skeleton_pixels")
    if skel is None or skel <= 0:
        reasons.append("empty_skeleton")

    return len(reasons) == 0, reasons


def row_to_subset_record(row: dict[str, Any], subset: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "subset": subset,
        "pilot_id": row["pilot_id"],
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "level": row["level"],
        "method": row["method"],
        "graph_path": row["graph_path"],
        "overlay_path": row["overlay_path"],
        "quality": {
            "foreground_ratio": row.get("foreground_ratio"),
            "skeleton_density": row.get("skeleton_density"),
            "skeleton_pixels": row.get("skeleton_pixels"),
            "node_count": row.get("node_count"),
            "edge_count": row.get("edge_count"),
            "component_count": row.get("component_count"),
            "junction_count": row.get("junction_count"),
            "endpoint_count": row.get("endpoint_count"),
            "nodes_per_1k_skeleton": row.get("nodes_per_1k_skeleton"),
            "edges_per_1k_skeleton": row.get("edges_per_1k_skeleton"),
            "components_per_1k_skeleton": row.get("components_per_1k_skeleton"),
            "junctions_per_1k_skeleton": row.get("junctions_per_1k_skeleton"),
            "endpoints_per_1k_skeleton": row.get("endpoints_per_1k_skeleton"),
            "junction_endpoint_ratio": row.get("junction_endpoint_ratio"),
            "edge_node_ratio": row.get("edge_node_ratio"),
            "warning_count": row.get("warning_count"),
            "warning_risk_score": row.get("warning_risk_score"),
            "warnings": row.get("warnings_list", []),
        },
        "selection": {
            "primary_variant": PRIMARY_VARIANTS.get(row["dataset"]),
            "is_primary_variant": is_primary(row),
            "rejection_reasons": reasons,
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mean(xs: list[float | int | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = defaultdict(list)

    for r in rows:
        by_dataset[r["dataset"]].append(r)

    out = {
        "count": len(rows),
        "by_dataset": dict(Counter(r["dataset"] for r in rows)),
        "by_method": dict(Counter(r["method"] for r in rows)),
        "metrics_by_dataset": {},
    }

    for dataset, group in sorted(by_dataset.items()):
        out["metrics_by_dataset"][dataset] = {
            "n": len(group),
            "skeleton_pixels_mean": mean([r["quality"]["skeleton_pixels"] for r in group]),
            "nodes_per_1k_skeleton_mean": mean([r["quality"]["nodes_per_1k_skeleton"] for r in group]),
            "components_per_1k_skeleton_mean": mean([r["quality"]["components_per_1k_skeleton"] for r in group]),
            "junctions_per_1k_skeleton_mean": mean([r["quality"]["junctions_per_1k_skeleton"] for r in group]),
            "warning_risk_score_mean": mean([r["quality"]["warning_risk_score"] for r in group]),
        }

    return out


def main() -> None:
    rows = load_rows(METRICS_CSV)

    clean = []
    stress = []
    review = []

    rejected_reasons = Counter()

    for row in rows:
        clean_ok, clean_reasons = clean_acceptance(row)
        stress_ok, stress_reasons = stress_acceptance(row)

        if clean_ok:
            clean.append(row_to_subset_record(row, "clean_graph_subset_v1", clean_reasons))
        elif stress_ok:
            stress.append(row_to_subset_record(row, "page_stress_graph_subset_v1", stress_reasons))
        else:
            reasons = clean_reasons if row["dataset"] in CLEAN_DATASETS else stress_reasons
            for reason in reasons:
                rejected_reasons[reason] += 1

            # Keep primary rejected rows for manual review.
            if is_primary(row):
                review.append(row_to_subset_record(row, "review_graph_subset_v1", reasons))

    write_jsonl(clean, OUT_CLEAN)
    write_jsonl(stress, OUT_STRESS)
    write_jsonl(review, OUT_REVIEW)

    summary = {
        "source_metrics_csv": str(METRICS_CSV),
        "primary_variants": PRIMARY_VARIANTS,
        "clean_subset": {
            "path": str(OUT_CLEAN),
            **summarize(clean),
        },
        "page_stress_subset": {
            "path": str(OUT_STRESS),
            **summarize(stress),
        },
        "review_subset": {
            "path": str(OUT_REVIEW),
            **summarize(review),
        },
        "rejected_reasons": dict(rejected_reasons),
        "notes": [
            "clean_graph_subset_v1 contains only crop/line datasets and primary variants.",
            "page_stress_graph_subset_v1 contains HWR200/HKR Forms primary page-level stress variants.",
            "Component filtering is not applied to clean crop/line graphs.",
            "Page stress graphs are not considered clean canonical stroke graphs.",
        ],
    }

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "clean": summary["clean_subset"],
        "page_stress": summary["page_stress_subset"],
        "review": summary["review_subset"],
        "rejected_reasons": summary["rejected_reasons"],
        "summary": str(OUT_SUMMARY),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()