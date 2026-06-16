from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_excluded_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()

    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            str(row.get("sample_id", "")).strip()
            for row in csv.DictReader(f)
            if str(row.get("sample_id", "")).strip()
        }


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "target", "label", "transcription"]:
        if key in row:
            return str(row[key])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exclude_csv", default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    excluded = read_excluded_ids(
        Path(args.exclude_csv) if args.exclude_csv else None
    )

    eligible = [
        row for row in rows
        if str(row.get("sample_id", "")) not in excluded
    ]

    if len(eligible) < args.n:
        raise RuntimeError(
            f"Only {len(eligible)} eligible samples for requested n={args.n}"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.n)
    selected.sort(key=lambda row: str(row.get("sample_id", "")))

    out_rows = []
    for row in selected:
        out_rows.append({
            "sample_id": row.get("sample_id", ""),
            "dataset": row.get(
                "dataset",
                row.get("source_dataset", "school_notebooks_clean"),
            ),
            "level": row.get("level", "unknown"),
            "category": row.get("category", "unknown"),
            "image_path": row.get("image_path", ""),
            "target": get_text(row),
            "pred": "",
            "cer": "",
            "structural_risk_score": "",
            "audit_cell": "random_validation",
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    metadata = {
        "manifest": args.manifest,
        "exclude_csv": args.exclude_csv,
        "seed": args.seed,
        "requested_n": args.n,
        "manifest_n": len(rows),
        "excluded_n": len(excluded),
        "eligible_n": len(eligible),
        "selected_n": len(selected),
    }

    out_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("wrote:", out_path)


if __name__ == "__main__":
    main()