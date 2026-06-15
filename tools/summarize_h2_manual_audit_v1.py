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


def as_int_str(x: Any) -> str:
    s = str(x).strip()
    return s if s else "missing"


def rate(rows: list[dict[str, Any]], key: str, positive: str = "1") -> float:
    vals = [str(r.get(key, "")).strip() for r in rows]
    vals = [v for v in vals if v != ""]
    if not vals:
        return 0.0
    return sum(v == positive for v in vals) / len(vals)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    vals = [as_float(r.get(key)) for r in rows if str(r.get(key, "")).strip() != ""]
    return float(np.mean(vals)) if vals else 0.0


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "usable_rate": rate(rows, "audit_usable"),
        "critical_topology_error_rate": rate(rows, "critical_topology_error"),
        "skeleton_follows_ink_rate": rate(rows, "skeleton_follows_ink"),
        "missed_visible_stroke_rate": rate(rows, "missed_visible_stroke"),
        "spurious_stroke_rate": rate(rows, "spurious_stroke"),
        "endpoint_error_rate": rate(rows, "endpoint_error"),
        "junction_error_rate": rate(rows, "junction_error"),
        "loop_error_rate": rate(rows, "loop_error"),
        "mean_graph_quality_0_3": mean_float(rows, "graph_quality_0_3"),
        "mean_cer": mean_float(rows, "cer"),
        "mean_structural_risk_score": mean_float(rows, "structural_risk_score"),
        "quality_counts": dict(Counter(as_int_str(r.get("graph_quality_0_3")) for r in rows)),
        "usable_counts": dict(Counter(as_int_str(r.get("audit_usable")) for r in rows)),
        "critical_counts": dict(Counter(as_int_str(r.get("critical_topology_error")) for r in rows)),
        "exclusion_reason_counts": dict(Counter(as_int_str(r.get("exclusion_reason")) for r in rows)),
    }


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[str(r.get(key, "unknown") or "unknown")].append(r)
    return dict(out)


def make_report_md(summary: dict[str, Any], out_path: Path) -> None:
    lines = []

    lines.append("# H2 manual audit summary — v1")
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
    lines.append(f"| mean graph quality 0–3 | {o['mean_graph_quality_0_3']:.3f} |")
    lines.append("")
    lines.append("## 2. By audit cell")
    lines.append("")
    lines.append("| cell | n | usable | critical | follows ink | mean quality | mean CER | mean risk |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for cell, s in summary["by_audit_cell"].items():
        lines.append(
            f"| `{cell}` | {s['n']} | {s['usable_rate']:.3f} | "
            f"{s['critical_topology_error_rate']:.3f} | "
            f"{s['skeleton_follows_ink_rate']:.3f} | "
            f"{s['mean_graph_quality_0_3']:.3f} | "
            f"{s['mean_cer']:.3f} | "
            f"{s['mean_structural_risk_score']:.3f} |"
        )

    lines.append("")
    lines.append("## 3. By dataset")
    lines.append("")
    lines.append("| dataset | n | usable | critical | follows ink | mean quality | mean CER |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for dataset, s in summary["by_dataset"].items():
        lines.append(
            f"| `{dataset}` | {s['n']} | {s['usable_rate']:.3f} | "
            f"{s['critical_topology_error_rate']:.3f} | "
            f"{s['skeleton_follows_ink_rate']:.3f} | "
            f"{s['mean_graph_quality_0_3']:.3f} | "
            f"{s['mean_cer']:.3f} |"
        )

    lines.append("")
    lines.append("## 4. Strict interpretation")
    lines.append("")

    a = summary["by_audit_cell"].get("A_highCER_highRisk", {})
    c = summary["by_audit_cell"].get("C_lowCER_highRisk", {})
    d = summary["by_audit_cell"].get("D_lowCER_lowRisk", {})

    a_crit = a.get("critical_topology_error_rate", 0.0)
    c_quality = c.get("mean_graph_quality_0_3", 0.0)
    d_quality = d.get("mean_graph_quality_0_3", 0.0)

    if a_crit < 0.35:
        lines.append(
            "High structural risk does not strongly correspond to visible critical graph failures. "
            "It appears to capture sample difficulty or structural complexity more than extraction failure."
        )
    else:
        lines.append(
            "High structural risk is visibly associated with critical graph failures in a substantial fraction of A samples."
        )

    if c_quality >= 2.5:
        lines.append(
            "Many C samples are false high-risk cases: recognition succeeds and the graph is visually acceptable."
        )

    if d_quality >= 2.8:
        lines.append(
            "D samples behave as clean controls: low CER, low risk, and high graph quality."
        )

    lines.append("")
    lines.append("## 5. Recommended conclusion")
    lines.append("")
    lines.append(
        "Use this audit to separate two claims: graph structural descriptors can help identify hard samples, "
        "but the current scalar risk score should not be presented as a direct graph-quality measure."
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
        "overall": summarize_group(rows),
        "by_audit_cell": {
            k: summarize_group(v)
            for k, v in sorted(group_by(rows, "audit_cell").items())
        },
        "by_dataset": {
            k: summarize_group(v)
            for k, v in sorted(group_by(rows, "dataset").items())
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report_md(summary, out_md)

    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("wrote:", out_json)
    print("wrote:", out_md)


if __name__ == "__main__":
    main()