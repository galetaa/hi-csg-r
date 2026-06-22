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
    "complete_enough",
    "neighbor_noise",
    "good_for_line_train",
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


def x_gap_score(row: dict[str, Any]) -> tuple[float, float, int, str]:
    max_gap = row.get("max_x_gap")
    median_gap = row.get("median_x_gap")

    return (
        float(max_gap if max_gap is not None else -10**9),
        float(median_gap if median_gap is not None else -10**9),
        int(row.get("n_words", 0)),
        str(row.get("line_group_id", "")),
    )


def pool_for_stratum(rows: list[dict[str, Any]], stratum: str) -> list[dict[str, Any]]:
    if stratum == "groups_2_words":
        return [
            row
            for row in rows
            if int(row.get("n_words", 0)) == 2
        ]

    if stratum == "groups_3_words":
        return [
            row
            for row in rows
            if int(row.get("n_words", 0)) == 3
        ]

    if stratum == "groups_4plus_words":
        return [
            row
            for row in rows
            if 4 <= int(row.get("n_words", 0)) < 8
        ]

    if stratum == "groups_8plus_words":
        return [
            row
            for row in rows
            if int(row.get("n_words", 0)) >= 8
        ]

    if stratum == "large_x_gap":
        return sorted(
            rows,
            key=x_gap_score,
            reverse=True,
        )

    raise ValueError(stratum)


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    plan = [
        ("groups_2_words", 30),
        ("groups_3_words", 30),
        ("groups_4plus_words", 40),
        ("groups_8plus_words", 25),
        ("large_x_gap", 25),
    ]

    summary: dict[str, Any] = {
        "seed": seed,
        "requested": dict(plan),
        "selected": {},
    }

    for stratum, target in plan:
        pool = [
            row
            for row in pool_for_stratum(rows, stratum)
            if str(row["line_group_id"]) not in selected_ids
        ]

        if stratum != "large_x_gap":
            rng.shuffle(pool)

        take = pool[:target]

        for row in take:
            row = dict(row)
            row["validation_stratum"] = stratum
            selected.append(row)
            selected_ids.add(str(row["line_group_id"]))

        summary["selected"][stratum] = {
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
                "complete_enough": "",
                "neighbor_noise": "",
                "good_for_line_train": "",
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
