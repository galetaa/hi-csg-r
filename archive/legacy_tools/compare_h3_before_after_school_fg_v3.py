from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_best(summary: dict[str, Any], feature_set: str) -> dict[str, Any] | None:
    by = summary.get("best_by_feature_set") or {}

    if feature_set in by:
        return by[feature_set]

    # compatibility with older h3_final report shape
    if feature_set == "structural_core" and summary.get("main_positive_result"):
        return summary["main_positive_result"]

    return None


def fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_h3_json", required=True)
    parser.add_argument("--new_h3_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    old = load(args.old_h3_json)
    new = load(args.new_h3_json)

    feature_sets = [
        "quality_only",
        "geometry_control",
        "structural_core",
        "all_non_geometry",
        "all_features",
        "all_features_no_text_len",
    ]

    rows = []

    for fs in feature_sets:
        o = get_best(old, fs)
        n = get_best(new, fs)

        if o is None and n is None:
            continue

        rows.append({
            "feature_set": fs,
            "old": o,
            "new": n,
            "roc_auc_delta": None if not o or not n else (n.get("roc_auc", 0) - o.get("roc_auc", 0)),
            "pr_auc_delta": None if not o or not n else (n.get("pr_auc", 0) - o.get("pr_auc", 0)),
            "top20_precision_delta": None if not o or not n else (
                n.get("top20_precision", 0) - o.get("top20_precision", 0)
            ),
        })

    result = {
        "old_h3_json": args.old_h3_json,
        "new_h3_json": args.new_h3_json,
        "rows": rows,
    }

    lines = []
    lines.append("# H3 before/after school foreground v3")
    lines.append("")
    lines.append("## 1. Feature-set comparison")
    lines.append("")
    lines.append("| feature set | old group | old ROC | new group | new ROC | ΔROC | old PR | new PR | ΔPR | old top20 | new top20 | Δtop20 |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        o = r["old"] or {}
        n = r["new"] or {}

        lines.append(
            f"| `{r['feature_set']}` | "
            f"`{o.get('group', 'n/a')}` | {fmt(o.get('roc_auc'))} | "
            f"`{n.get('group', 'n/a')}` | {fmt(n.get('roc_auc'))} | "
            f"{fmt(r['roc_auc_delta'])} | "
            f"{fmt(o.get('pr_auc'))} | {fmt(n.get('pr_auc'))} | "
            f"{fmt(r['pr_auc_delta'])} | "
            f"{fmt(o.get('top20_precision'))} | {fmt(n.get('top20_precision'))} | "
            f"{fmt(r['top20_precision_delta'])} |"
        )

    lines.append("")
    lines.append("## 2. Interpretation rule")
    lines.append("")
    lines.append(
        "If H3 improves mainly for school-notebooks groups, foreground v3 repaired the diagnostic graph signal for that subset. "
        "If global H3 changes little, that is acceptable: this was a preprocessing repair, not a model retraining step."
    )

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()