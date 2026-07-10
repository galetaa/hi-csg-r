from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXTRA_COLUMNS = [
    "failure_stage",
    "border_artifact",
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

    for r in rows:
        r.setdefault("failure_stage", "")
        r.setdefault("border_artifact", "")

        # Sensible defaults, still manually review school_notebooks.
        if not r["failure_stage"]:
            r["failure_stage"] = "ok"

        if not r["border_artifact"]:
            r["border_artifact"] = "0"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("wrote:", out_path)


if __name__ == "__main__":
    main()