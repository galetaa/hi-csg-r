from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def clean(x: Any) -> str:
    s = str(x or "").strip()
    return s if s else "missing"


def rate(rows: list[dict[str, Any]], key: str, positive: str = "1") -> float:
    vals = [clean(r.get(key)) for r in rows if clean(r.get(key)) != "missing"]
    if not vals:
        return 0.0
    return sum(v == positive for v in vals) / len(vals)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    vals = [as_float(r.get(key)) for r in rows if clean(r.get(key)) != "missing"]
    return float(np.mean(vals)) if vals else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "usable_rate": rate(rows, "audit_usable"),
        "critical_topology_error_rate": rate(rows, "critical_topology_error"),
        "skeleton_follows_ink_rate": rate(rows, "skeleton_follows_ink"),
        "border_artifact_rate": rate(rows, "border_artifact"),
        "mean_graph_quality_0_3": mean_float(rows, "graph_quality_0_3"),
        "mean_cer": mean_float(rows, "cer"),
        "mean_structural_risk_score": mean_float(rows, "structural_risk_score"),
        "failure_stage_counts": dict(Counter(clean(r.get("failure_stage")) for r in rows)),
        "quality_counts": dict(Counter(clean(r.get("graph_quality_0_3")) for r in rows)),
        "critical_counts": dict(Counter(clean(r.get("critical_topology_error")) for r in rows)),
        "border_artifact_counts": dict(Counter(clean(r.get("border_artifact")) for r in rows)),
        "exclusion_reason_counts": dict(Counter(clean(r.get("exclusion_reason")) for r in rows)),
    }


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out = defaultdict(list)
    for r in rows:
        out[clean(r.get(key))].append(r)
    return dict(out)


def make_md(summary: dict[str, Any], out_path: Path) -> None:
    lines = []

    lines.append("# H2 manual audit summary — v2")
    lines.append("")
    lines.append("## 1. Overall")
    lines.append("")
    o = summary["overall"]

    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| n | {o['n']} |")
    lines.append(f"| usable rate | {o['usable_rate']:.3f} |")
    lines.append(f"| critical topology error rate | {o['critical_topology_error_rate']:.3f} |")
    lines.append(f"| skeleton follows ink rate | {o['skeleton_follows_ink_rate']:.3f} |")
    lines.append(f"| border artifact rate | {o['border_artifact_rate']:.3f} |")
    lines.append(f"| mean graph quality 0–3 | {o['mean_graph_quality_0_3']:.3f} |")
    lines.append("")

    lines.append("## 2. Failure stages")
    lines.append("")
    lines.append("| stage | n |")
    lines.append("|---|---:|")
    for stage, n in sorted(o["failure_stage_counts"].items()):
        lines.append(f"| `{stage}` | {n} |")
    lines.append("")

    lines.append("## 3. By dataset")
    lines.append("")
    lines.append("| dataset | n | usable | critical | follows ink | border artifact | mean quality | failure stages |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")

    for dataset, s in summary["by_dataset"].items():
        stages = ", ".join(f"{k}:{v}" for k, v in sorted(s["failure_stage_counts"].items()))
        lines.append(
            f"| `{dataset}` | {s['n']} | {s['usable_rate']:.3f} | "
            f"{s['critical_topology_error_rate']:.3f} | "
            f"{s['skeleton_follows_ink_rate']:.3f} | "
            f"{s['border_artifact_rate']:.3f} | "
            f"{s['mean_graph_quality_0_3']:.3f} | {stages} |"
        )

    lines.append("")
    lines.append("## 4. By audit cell")
    lines.append("")
    lines.append("| cell | n | usable | critical | follows ink | border artifact | mean quality | mean CER | mean risk |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for cell, s in summary["by_audit_cell"].items():
        lines.append(
            f"| `{cell}` | {s['n']} | {s['usable_rate']:.3f} | "
            f"{s['critical_topology_error_rate']:.3f} | "
            f"{s['skeleton_follows_ink_rate']:.3f} | "
            f"{s['border_artifact_rate']:.3f} | "
            f"{s['mean_graph_quality_0_3']:.3f} | "
            f"{s['mean_cer']:.3f} | "
            f"{s['mean_structural_risk_score']:.3f} |"
        )

    lines.append("")
    lines.append("## 5. Strict interpretation")
    lines.append("")

    school = summary["by_dataset"].get("school_notebooks_clean", {})
    school_border = school.get("border_artifact_rate", 0.0)
    school_crit = school.get("critical_topology_error_rate", 0.0)

    if school_border >= 0.5 or school_crit >= 0.5:
        lines.append(
            "The school-notebooks subset is dominated by upstream crop/binarization artifacts. "
            "These samples should not be interpreted as pure graph-topology failures."
        )
    else:
        lines.append(
            "School-notebooks does not appear dominated by border artifacts after manual staging."
        )

    lines.append("")
    lines.append(
        "The current structural risk score should be interpreted as a hard-sample indicator, "
        "not as a direct graph-quality score. Manual audit separates extraction failures "
        "from crop and binarization failures."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.annotations_csv))

    summary = {
        "annotations_csv": args.annotations_csv,
        "overall": summarize(rows),
        "by_dataset": {
            k: summarize(v)
            for k, v in sorted(group_by(rows, "dataset").items())
        },
        "by_audit_cell": {
            k: summarize(v)
            for k, v in sorted(group_by(rows, "audit_cell").items())
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_md(summary, out_md)

    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("wrote:", out_json)
    print("wrote:", out_md)


if __name__ == "__main__":
    main()