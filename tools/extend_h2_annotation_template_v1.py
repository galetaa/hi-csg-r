from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXTRA_COLUMNS = [
    "audit_usable",
    "exclusion_reason",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for col in EXTRA_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        row.setdefault("audit_usable", "")
        row.setdefault("exclusion_reason", "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("wrote:", out_path)


if __name__ == "__main__":
    main()