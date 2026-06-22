from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOOL_FIELDS = [
    "valid_line",
    "correct_order",
    "complete_enough",
    "neighbor_noise",
    "good_for_line_train",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def parse_bool(value: Any) -> bool | None:
    text = str(value or "").strip().lower()

    if text in {"1", "true", "yes", "y", "да", "д", "+"}:
        return True

    if text in {"0", "false", "no", "n", "нет", "н", "-"}:
        return False

    return None


def rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        parse_bool(row.get(field))
        for row in rows
    ]
    known = [
        value
        for value in values
        if value is not None
    ]

    if not known:
        return None

    return sum(known) / len(known)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "completed": sum(
            all(parse_bool(row.get(field)) is not None for field in BOOL_FIELDS)
            for row in rows
        ),
        "valid_line_rate": rate(rows, "valid_line"),
        "correct_order_rate": rate(rows, "correct_order"),
        "complete_enough_rate": rate(rows, "complete_enough"),
        "neighbor_noise_rate": rate(rows, "neighbor_noise"),
        "good_for_line_train_rate": rate(rows, "good_for_line_train"),
    }


def accepted(summary: dict[str, Any]) -> bool:
    valid_line_rate = summary.get("valid_line_rate")
    correct_order_rate = summary.get("correct_order_rate")
    complete_enough_rate = summary.get("complete_enough_rate")
    neighbor_noise_rate = summary.get("neighbor_noise_rate")
    good_for_line_train_rate = summary.get("good_for_line_train_rate")

    return bool(
        valid_line_rate is not None
        and correct_order_rate is not None
        and complete_enough_rate is not None
        and neighbor_noise_rate is not None
        and good_for_line_train_rate is not None
        and valid_line_rate >= 0.95
        and correct_order_rate >= 0.98
        and complete_enough_rate >= 0.85
        and neighbor_noise_rate <= 0.10
        and good_for_line_train_rate >= 0.85
    )


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--selection_jsonl", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    annotations_path = Path(args.annotations_csv)
    selection_path = Path(args.selection_jsonl)

    with annotations_path.open(encoding="utf-8-sig", newline="") as f:
        annotations = list(csv.DictReader(f))

    selection = {
        str(row["line_group_id"]): row
        for row in read_jsonl(selection_path)
    }

    rows = []
    missing_selection = []

    for row in annotations:
        line_group_id = str(row.get("line_group_id", ""))
        selected = selection.get(line_group_id)

        if selected is None:
            missing_selection.append(line_group_id)

        merged = dict(row)
        merged["validation_stratum"] = (
            selected or {}
        ).get("validation_stratum", "unknown")
        merged["n_words"] = (
            selected or {}
        ).get("n_words")
        merged["max_x_gap"] = (
            selected or {}
        ).get("max_x_gap")
        rows.append(merged)

    overall = summarize_rows(rows)

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_stratum[str(row["validation_stratum"])].append(row)

    stratum_summary = {
        stratum: summarize_rows(items)
        for stratum, items in sorted(by_stratum.items())
    }

    result = {
        "annotations_csv": str(annotations_path),
        "selection_jsonl": str(selection_path),
        "criteria": {
            "valid_line_rate": ">= 0.95",
            "correct_order_rate": ">= 0.98",
            "complete_enough_rate": ">= 0.85",
            "neighbor_noise_rate": "<= 0.10",
            "good_for_line_train_rate": ">= 0.85",
        },
        "overall": {
            **overall,
            "accepted": accepted(overall),
        },
        "by_stratum": {
            stratum: {
                **summary,
                "accepted": accepted(summary),
            }
            for stratum, summary in stratum_summary.items()
        },
        "value_counts": {
            field: dict(
                Counter(
                    str(row.get(field, "")).strip()
                    for row in rows
                )
            )
            for field in BOOL_FIELDS
        },
        "missing_selection_count": len(missing_selection),
        "missing_selection": missing_selection[:20],
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Full Natural Line Validation - Iteration 2",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
    ]

    for key in [
        "n",
        "completed",
        "valid_line_rate",
        "correct_order_rate",
        "complete_enough_rate",
        "neighbor_noise_rate",
        "good_for_line_train_rate",
        "accepted",
    ]:
        lines.append(f"| `{key}` | {fmt(result['overall'].get(key))} |")

    lines.extend([
        "",
        "## By Stratum",
        "",
        "| stratum | n | valid | order | complete | noise | train | accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for stratum, summary in result["by_stratum"].items():
        lines.append(
            f"| `{stratum}` | "
            f"{summary['n']} | "
            f"{fmt(summary['valid_line_rate'])} | "
            f"{fmt(summary['correct_order_rate'])} | "
            f"{fmt(summary['complete_enough_rate'])} | "
            f"{fmt(summary['neighbor_noise_rate'])} | "
            f"{fmt(summary['good_for_line_train_rate'])} | "
            f"{summary['accepted']} |"
        )

    out_md.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
