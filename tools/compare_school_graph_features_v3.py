from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


KEYS = [
    "fg_fraction",
    "cc_count",
    "cc_area_max_frac",
    "skel_pixels",
    "skel_fraction",
    "skel_components",
    "graph_endpoint_count",
    "graph_branchpoint_count",
    "endpoint_per_100_skel",
    "branchpoint_per_100_skel",
    "stroke_width_mean",
    "warning_count",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, 0.0)))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"n": len(rows)}
    for k in KEYS:
        out[f"{k}_mean"] = mean(rows, k)

    out["very_high_foreground_rate"] = sum(
        "very_high_foreground" in str(r.get("warnings", ""))
        for r in rows
    ) / max(len(rows), 1)

    out["very_low_foreground_rate"] = sum(
        "very_low_foreground" in str(r.get("warnings", ""))
        for r in rows
    ) / max(len(rows), 1)

    out["too_few_skeleton_pixels_rate"] = sum(
        "too_few_skeleton_pixels" in str(r.get("warnings", ""))
        for r in rows
    ) / max(len(rows), 1)

    return out


def fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_features_dir", required=True)
    parser.add_argument("--new_features_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="school_notebooks_clean")
    args = parser.parse_args()

    old_dir = Path(args.old_features_dir)
    new_dir = Path(args.new_features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "old_features_dir": str(old_dir),
        "new_features_dir": str(new_dir),
        "dataset": args.dataset,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        old_path = old_dir / f"{args.dataset}_{split}_features.jsonl"
        new_path = new_dir / f"{args.dataset}_{split}_features.jsonl"

        old_rows = read_jsonl(old_path)
        new_rows = read_jsonl(new_path)

        old_by_id = {str(r["sample_id"]): r for r in old_rows}
        new_by_id = {str(r["sample_id"]): r for r in new_rows}

        common = sorted(set(old_by_id) & set(new_by_id))

        deltas = []
        for sid in common:
            o = old_by_id[sid]
            n = new_by_id[sid]
            d = {"sample_id": sid}
            for k in KEYS:
                try:
                    d[f"{k}_old"] = float(o.get(k, 0.0))
                    d[f"{k}_new"] = float(n.get(k, 0.0))
                    d[f"{k}_delta"] = d[f"{k}_new"] - d[f"{k}_old"]
                except Exception:
                    d[f"{k}_old"] = None
                    d[f"{k}_new"] = None
                    d[f"{k}_delta"] = None
            deltas.append(d)

        result["splits"][split] = {
            "old": summarize(old_rows),
            "new": summarize(new_rows),
            "common_n": len(common),
            "mean_deltas": {
                k: float(np.mean([d[f"{k}_delta"] for d in deltas if d[f"{k}_delta"] is not None]))
                for k in KEYS
            },
        }

    out_json = out_dir / "school_graph_features_v3_comparison.json"
    out_md = out_dir / "school_graph_features_v3_comparison.md"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# School graph features v3 comparison")
    lines.append("")
    lines.append(f"old: `{old_dir}`")
    lines.append(f"new: `{new_dir}`")
    lines.append("")
    lines.append("## Summary by split")
    lines.append("")
    lines.append("| split | n | old fg | new fg | old skel | new skel | old warnings | new warnings | old high-fg | new high-fg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for split, s in result["splits"].items():
        o = s["old"]
        n = s["new"]
        lines.append(
            f"| `{split}` | {s['common_n']} | "
            f"{fmt(o['fg_fraction_mean'])} | {fmt(n['fg_fraction_mean'])} | "
            f"{fmt(o['skel_fraction_mean'])} | {fmt(n['skel_fraction_mean'])} | "
            f"{fmt(o['warning_count_mean'])} | {fmt(n['warning_count_mean'])} | "
            f"{fmt(o['very_high_foreground_rate'])} | {fmt(n['very_high_foreground_rate'])} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "A useful foreground fix should reduce excessive foreground and skeleton clutter without collapsing skeletons to near zero. "
        "Inspect the contact sheet and this table together; numeric reduction alone is not sufficient."
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_json)


if __name__ == "__main__":
    main()