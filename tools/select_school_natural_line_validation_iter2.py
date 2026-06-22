from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


ANNOTATION_FIELDS = [
    "line_group_id",
    "valid_line",
    "correct_order",
    "missing_words",
    "neighbor_noise",
    "good_for_train_aug",
    "notes",
]


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


def bucket(row: dict[str, Any]) -> str:
    flags = set(row.get("flags") or [])

    if "long_group" in flags:
        return "long_group"

    if "has_hard_real" in flags:
        return "has_hard_real"

    if "short_group" in flags:
        return "short_group"

    if "all_clean_core" in flags:
        return "all_clean_core"

    return "random"


def ordered_pool(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "random":
        return list(rows)

    if name == "all_clean_core":
        return [
            row
            for row in rows
            if "all_clean_core" in set(row.get("flags") or [])
        ]

    if name == "has_hard_real":
        return [
            row
            for row in rows
            if "has_hard_real" in set(row.get("flags") or [])
        ]

    if name == "short_group":
        return [
            row
            for row in rows
            if "short_group" in set(row.get("flags") or [])
        ]

    if name == "long_group":
        return [
            row
            for row in rows
            if "long_group" in set(row.get("flags") or [])
        ]

    raise ValueError(name)


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    plan = [
        ("all_clean_core", 40),
        ("has_hard_real", 30),
        ("short_group", 30),
        ("long_group", 20),
        ("random", 20),
    ]

    summary: dict[str, Any] = {
        "seed": seed,
        "requested": dict(plan),
        "selected": {},
    }

    for name, target in plan:
        pool = [
            row
            for row in ordered_pool(rows, name)
            if str(row["line_group_id"]) not in selected_ids
        ]
        rng.shuffle(pool)

        take = pool[:target]

        for row in take:
            row = dict(row)
            row["validation_stratum"] = name
            selected.append(row)
            selected_ids.add(str(row["line_group_id"]))

        summary["selected"][name] = {
            "target": target,
            "available_after_dedup": len(pool),
            "n": len(take),
        }

    selected.sort(
        key=lambda row: (
            row["validation_stratum"],
            row["split"],
            row["source_image_file"],
            row["bbox_xyxy"][1],
            row["bbox_xyxy"][0],
        )
    )

    summary["total_selected"] = len(selected)

    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.candidates))
    selected, summary = select_rows(rows, seed=args.seed)

    write_jsonl(selected, Path(args.out_jsonl))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()

        for row in selected:
            writer.writerow({
                "line_group_id": row["line_group_id"],
                "valid_line": "",
                "correct_order": "",
                "missing_words": "",
                "neighbor_noise": "",
                "good_for_train_aug": "",
                "notes": "",
            })

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
