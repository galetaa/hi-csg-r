from __future__ import annotations

import argparse
import csv
import json
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

THRESHOLDS = {
    "structural_usable": 0.85,
    "foreground_ok": 0.85,
    "skeleton_ok": 0.80,
    "graph_ok": 0.75,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value: Any) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except Exception:
        return None


def rate(rows: list[dict[str, Any]], field: str, positive: int = 1) -> float | None:
    values = [as_int(row.get(field, "")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(value == positive for value in values) / len(values)


def severity_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        value = str(row.get(field, "")).strip()
        counts[value if value else "missing"] += 1
    return dict(counts)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(rows),
        "completed": sum(
            all(str(row.get(field, "")).strip() != "" for field in BOOL_FIELDS + SEVERITY_FIELDS)
            and str(row.get("htr_error_explained_by_structure", "")).strip() != ""
            for row in rows
        ),
        "rates": {
            field: rate(rows, field)
            for field in BOOL_FIELDS
        },
        "severity_counts": {
            field: severity_counts(rows, field)
            for field in SEVERITY_FIELDS
        },
        "htr_error_explained_by_structure": dict(Counter(
            str(row.get("htr_error_explained_by_structure", "")).strip() or "missing"
            for row in rows
        )),
    }
    out["accepted"] = {
        field: (
            out["rates"][field] is not None
            and out["rates"][field] >= threshold
        )
        for field, threshold in THRESHOLDS.items()
    }
    out["all_acceptance_passed"] = all(out["accepted"].values())
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    annotations = read_csv(Path(args.annotations_csv))
    manifest_by_id = {
        str(row["sample_id"]): row
        for row in read_jsonl(Path(args.sample_manifest))
    }

    rows = []
    for row in annotations:
        sample_id = str(row["sample_id"])
        merged: dict[str, Any] = dict(row)
        source = manifest_by_id.get(sample_id, {})
        merged["exact"] = source.get("exact")
        merged["cer"] = source.get("cer")
        merged["risk"] = source.get("risk")
        rows.append(merged)

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_token_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_stratum[str(row.get("stratum", ""))].append(row)
        by_dataset[str(row.get("dataset", ""))].append(row)
        by_token_type[str(row.get("token_type", ""))].append(row)

    summary = {
        "annotations_csv": args.annotations_csv,
        "sample_manifest": args.sample_manifest,
        "overall": summarize_group(rows),
        "by_stratum": {
            key: summarize_group(value)
            for key, value in sorted(by_stratum.items())
        },
        "by_dataset": {
            key: summarize_group(value)
            for key, value in sorted(by_dataset.items())
        },
        "by_token_type": {
            key: summarize_group(value)
            for key, value in sorted(by_token_type.items())
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Structural Gold Annotation Summary",
        "",
        "## Overall Acceptance",
        "",
        "| metric | rate | threshold | passed |",
        "|---|---:|---:|---|",
    ]
    overall = summary["overall"]
    for field, threshold in THRESHOLDS.items():
        lines.append(
            f"| `{field}` | {fmt(overall['rates'][field])} | {fmt(threshold)} | "
            f"{'yes' if overall['accepted'][field] else 'no'} |"
        )

    lines.extend([
        "",
        f"Completed rows: {overall['completed']} / {overall['n']}",
        f"All acceptance passed: {'yes' if overall['all_acceptance_passed'] else 'no'}",
        "",
        "## By Stratum",
        "",
        "| stratum | n | usable | foreground | skeleton | graph |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, item in summary["by_stratum"].items():
        lines.append(
            f"| `{key}` | {item['n']} | "
            f"{fmt(item['rates']['structural_usable'])} | "
            f"{fmt(item['rates']['foreground_ok'])} | "
            f"{fmt(item['rates']['skeleton_ok'])} | "
            f"{fmt(item['rates']['graph_ok'])} |"
        )

    lines.extend([
        "",
        "## By Dataset",
        "",
        "| dataset | n | usable | foreground | skeleton | graph |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, item in summary["by_dataset"].items():
        lines.append(
            f"| `{key}` | {item['n']} | "
            f"{fmt(item['rates']['structural_usable'])} | "
            f"{fmt(item['rates']['foreground_ok'])} | "
            f"{fmt(item['rates']['skeleton_ok'])} | "
            f"{fmt(item['rates']['graph_ok'])} |"
        )

    lines.extend([
        "",
        "## HTR Error Explained By Structure",
        "",
        "| value | count |",
        "|---|---:|",
    ])
    for key, count in sorted(overall["htr_error_explained_by_structure"].items()):
        lines.append(f"| `{key}` | {count} |")

    lines.extend([
        "",
        "## Severe/Dominant Issue Counts",
        "",
        "| issue | severe_or_dominant_count |",
        "|---|---:|",
    ])
    for field in SEVERITY_FIELDS:
        counts = overall["severity_counts"][field]
        lines.append(f"| `{field}` | {counts.get('2', 0)} |")

    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_json": args.out_json,
        "out_md": args.out_md,
        "overall": summary["overall"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
