from __future__ import annotations

import json
from pathlib import Path


OUT = Path("outputs/htr_graph_v1")
REPORT_MD = OUT / "stage4_graph_ablation_report.md"
REPORT_JSON = OUT / "stage4_graph_ablation_report.json"


ROWS = [
    {
        "model": "image-only old trainer",
        "mixed_val": 0.10460753240310658,
        "cyr_test": 0.19319943785182367,
        "hkr_test": 0.09562620560768798,
        "school_test": 0.15749263653013648,
        "note": "Original tri10k image-only baseline.",
    },
    {
        "model": "graph-vector v2 lowcap-all",
        "mixed_val": 0.09912381438475606,
        "cyr_test": 0.1984,
        "hkr_test": 0.0879,
        "school_test": 0.1580,
        "note": "Best global graph-vector fusion. Improves mixed validation and HKR, not robust on Cyrillic/School.",
    },
    {
        "model": "local gray",
        "mixed_val": 0.1025,
        "cyr_test": 0.2015,
        "hkr_test": 0.0913,
        "school_test": 0.1666,
        "note": "Control run using local-channel trainer with grayscale only.",
    },
    {
        "model": "local gray+fg",
        "mixed_val": 0.1068,
        "cyr_test": 0.2056,
        "hkr_test": 0.0871,
        "school_test": 0.1742,
        "note": "Foreground helps HKR but hurts Cyrillic and School.",
    },
    {
        "model": "local gray+skel",
        "mixed_val": 0.1054,
        "cyr_test": 0.2076,
        "hkr_test": 0.0872,
        "school_test": 0.1678,
        "note": "Skeleton helps HKR but is not robust.",
    },
    {
        "model": "local gray+dist",
        "mixed_val": 0.1027,
        "cyr_test": 0.1995,
        "hkr_test": 0.0874,
        "school_test": 0.1681,
        "note": "Distance channel is the safest local graph channel.",
    },
    {
        "model": "local gray+fg+skel",
        "mixed_val": 0.1064,
        "cyr_test": 0.2043,
        "hkr_test": 0.0934,
        "school_test": 0.1744,
        "note": "Combining fg+skel hurts most domains.",
    },
    {
        "model": "local gray+fg+skel+dist",
        "mixed_val": 0.1025,
        "cyr_test": 0.2003,
        "hkr_test": 0.0853,
        "school_test": 0.1683,
        "note": "Best HKR result among local-channel variants.",
    },
]


def fmt(x: float) -> str:
    return f"{x:.4f}"


def rel(base: float, new: float) -> float:
    return (base - new) / base if base else 0.0


def main() -> None:
    data = {"rows": ROWS}

    gray = next(r for r in ROWS if r["model"] == "local gray")

    for r in ROWS:
        r["vs_local_gray"] = {
            "mixed_val": rel(gray["mixed_val"], r["mixed_val"]),
            "cyr_test": rel(gray["cyr_test"], r["cyr_test"]),
            "hkr_test": rel(gray["hkr_test"], r["hkr_test"]),
            "school_test": rel(gray["school_test"], r["school_test"]),
        }

    lines = []
    lines.append("# Stage 4 graph-aware ablation report\n")

    lines.append("## 1. Purpose\n")
    lines.append(
        "This report evaluates whether graph-derived information improves HTR on the tri10k mixed Cyrillic subset. "
        "Two graph-aware families were tested: global graph-vector fusion and local graph-derived image channels.\n"
    )

    lines.append("## 2. Main results\n")
    lines.append("| model | mixed val CER | Cyrillic test CER | HKR test CER | School test CER |")
    lines.append("|---|---:|---:|---:|---:|")

    for r in ROWS:
        lines.append(
            f"| {r['model']} | {fmt(r['mixed_val'])} | {fmt(r['cyr_test'])} | "
            f"{fmt(r['hkr_test'])} | {fmt(r['school_test'])} |"
        )

    lines.append("")
    lines.append("## 3. Local-channel comparison against local gray control\n")
    lines.append("| model | mixed val Δ | Cyrillic Δ | HKR Δ | School Δ |")
    lines.append("|---|---:|---:|---:|---:|")

    for r in ROWS:
        if not r["model"].startswith("local "):
            continue
        v = r["vs_local_gray"]
        lines.append(
            f"| {r['model']} | "
            f"{100*v['mixed_val']:.1f}% | "
            f"{100*v['cyr_test']:.1f}% | "
            f"{100*v['hkr_test']:.1f}% | "
            f"{100*v['school_test']:.1f}% |"
        )

    lines.append("")
    lines.append("## 4. Interpretation\n")
    lines.append(
        "The global graph-vector low-capacity fusion gives the best mixed validation CER, but the gain is not stable across all test domains. "
        "It improves HKR Words but does not improve Cyrillic Handwriting or School Notebooks consistently.\n"
    )
    lines.append(
        "The local-channel ablation shows that graph-derived channels are useful for HKR Words. "
        "The full local channel variant improves HKR from 0.0913 to 0.0853 CER relative to the local gray control. "
        "However, the same channels hurt School Notebooks and do not reliably improve Cyrillic Handwriting.\n"
    )
    lines.append(
        "Therefore, naive early fusion of graph-derived channels is not robust enough. "
        "The next architecture should use gated or residual graph injection so that the model can suppress graph channels when they are harmful.\n"
    )

    lines.append("## 5. Stage 4 conclusion\n")
    lines.append("```text")
    lines.append("[x] graph feature extraction completed")
    lines.append("[x] global graph-vector fusion tested")
    lines.append("[x] local graph-channel fusion tested")
    lines.append("[x] channel ablation completed")
    lines.append("[!] graph signal is useful but domain-dependent")
    lines.append("[next] gated/residual graph-aware fusion")
    lines.append("```")

    REPORT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("wrote:", REPORT_MD)
    print("wrote:", REPORT_JSON)


if __name__ == "__main__":
    main()