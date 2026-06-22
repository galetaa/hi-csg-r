from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "readable",
    "correct_crop",
    "good_for_htr",
]


THRESHOLDS = {
    "readable": 0.95,
    "correct_crop": 0.95,
    "good_for_htr": 0.90,
}


def as_bool(value: str) -> bool | None:
    value = str(value).strip().lower()

    if value in {"1", "true", "yes", "y", "да"}:
        return True

    if value in {"0", "false", "no", "n", "нет"}:
        return False

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    rows = []

    with Path(args.annotations_csv).open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    summary = {
        "n": len(rows),
        "completed": 0,
        "rates": {},
        "thresholds": THRESHOLDS,
        "passed": {},
    }

    completed = 0

    for row in rows:
        if all(as_bool(row.get(field, "")) is not None for field in FIELDS):
            completed += 1

    summary["completed"] = completed

    for field in FIELDS:
        values = [
            as_bool(row.get(field, ""))
            for row in rows
        ]
        values = [
            value for value in values
            if value is not None
        ]

        rate = sum(values) / max(len(values), 1)
        summary["rates"][field] = rate
        summary["passed"][field] = rate >= THRESHOLDS[field]

    summary["accepted"] = (
        completed == len(rows)
        and all(summary["passed"].values())
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
