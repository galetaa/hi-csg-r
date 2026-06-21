from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


KEYS = [
    "fg_fraction",
    "skel_fraction",
    "cc_count",
    "skel_components",
    "endpoint_count",
    "branchpoint_count",
    "dir_h_frac",
    "stroke_width_mean",
    "ruling_response_mean",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def q(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "p05": None,
            "median": None,
            "p95": None,
        }

    arr = np.asarray(values, dtype=np.float64)

    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def summarize(root: Path) -> dict[str, Any]:
    out = {}

    for split in ["train", "val", "test"]:
        path = root / "diagnostics" / f"{split}.jsonl"

        if not path.exists():
            continue

        rows = read_jsonl(path)

        out[split] = {
            key: q([
                float(row[key])
                for row in rows
                if key in row
                and row[key] is not None
            ])
            for key in KEYS
        }

    return out


def diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    result = {}

    for split in sorted(set(old) | set(new)):
        result[split] = {}

        for key in KEYS:
            old_value = (
                old.get(split, {})
                .get(key, {})
                .get("mean")
            )
            new_value = (
                new.get(split, {})
                .get(key, {})
                .get("mean")
            )

            if old_value is None or new_value is None:
                delta = None
            else:
                delta = float(new_value) - float(old_value)

            result[split][key] = {
                "old_mean": old_value,
                "new_mean": new_value,
                "delta_mean": delta,
            }

    return result


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.5f}"
    except Exception:
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_root", required=True)
    parser.add_argument("--new_root", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    old_summary = summarize(Path(args.old_root))
    new_summary = summarize(Path(args.new_root))
    delta = diff(old_summary, new_summary)

    result = {
        "old_root": args.old_root,
        "new_root": args.new_root,
        "old": old_summary,
        "new": new_summary,
        "delta": delta,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("# School graph distribution comparison - Iteration 2")
    lines.append("")

    for split in ["train", "val", "test"]:
        lines.append(f"## {split}")
        lines.append("")
        lines.append("| feature | old mean | new mean | delta |")
        lines.append("|---|---:|---:|---:|")

        for key in KEYS:
            row = delta.get(split, {}).get(key, {})
            lines.append(
                f"| `{key}` | "
                f"{fmt(row.get('old_mean'))} | "
                f"{fmt(row.get('new_mean'))} | "
                f"{fmt(row.get('delta_mean'))} |"
            )

        lines.append("")

    out_md.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("wrote:", out_json)
    print("wrote:", out_md)


if __name__ == "__main__":
    main()
