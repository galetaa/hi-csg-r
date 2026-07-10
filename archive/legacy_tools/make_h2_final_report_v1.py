from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2_summary", required=True)
    parser.add_argument("--border_report", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    h2 = load(args.h2_summary)

    overall = h2["overall"]
    by_dataset = h2["by_dataset"]

    hkr = by_dataset.get("hkr_words", {})
    cyr = by_dataset.get("cyrillic_handwriting", {})
    school = by_dataset.get("school_notebooks_clean", {})

    hkr_cyr_n = hkr.get("n", 0) + cyr.get("n", 0)

    hkr_cyr_critical = (
        (
            hkr.get("critical_topology_error_rate", 0.0) * hkr.get("n", 0)
            + cyr.get("critical_topology_error_rate", 0.0) * cyr.get("n", 0)
        )
        / max(hkr_cyr_n, 1)
    )

    hkr_cyr_quality = (
        (
            hkr.get("mean_graph_quality_0_3", 0.0) * hkr.get("n", 0)
            + cyr.get("mean_graph_quality_0_3", 0.0) * cyr.get("n", 0)
        )
        / max(hkr_cyr_n, 1)
    )

    hkr_cyr_follows = (
        (
            hkr.get("skeleton_follows_ink_rate", 0.0) * hkr.get("n", 0)
            + cyr.get("skeleton_follows_ink_rate", 0.0) * cyr.get("n", 0)
        )
        / max(hkr_cyr_n, 1)
    )

    final = {
        "h2_v1_verdict": "partial_support_with_preprocessing_exception",
        "overall": overall,
        "by_dataset": by_dataset,
        "hkr_plus_cyrillic": {
            "n": hkr_cyr_n,
            "critical_topology_error_rate": hkr_cyr_critical,
            "skeleton_follows_ink_rate": hkr_cyr_follows,
            "mean_graph_quality_0_3": hkr_cyr_quality,
        },
        "school_notebooks_decision": {
            "include_in_graph_topology_claim": False,
            "reason": "Dominated by upstream crop/border/binarization artifacts.",
            "border_suppression_v1": "rejected",
        },
    }

    lines: list[str] = []

    lines.append("# H2 final report — v1")
    lines.append("")
    lines.append("## 1. Verdict")
    lines.append("")
    lines.append("```text")
    lines.append("H2-v1: partial support with preprocessing exception")
    lines.append("```")
    lines.append("")
    lines.append(
        "The graph extraction pipeline preserves visible stroke structure reasonably well "
        "on HKR and Cyrillic samples in the manual audit. However, the school-notebooks "
        "subset is dominated by crop/border artifacts that are binarized as foreground. "
        "Those failures are upstream preprocessing failures and should not be interpreted "
        "as pure graph-topology failures."
    )
    lines.append("")

    lines.append("## 2. Overall manual audit")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| n | {overall['n']} |")
    lines.append(f"| usable rate | {fmt(overall['usable_rate'])} |")
    lines.append(f"| critical topology error rate | {fmt(overall['critical_topology_error_rate'])} |")
    lines.append(f"| skeleton follows ink rate | {fmt(overall['skeleton_follows_ink_rate'])} |")
    lines.append(f"| border artifact rate | {fmt(overall['border_artifact_rate'])} |")
    lines.append(f"| mean graph quality 0–3 | {fmt(overall['mean_graph_quality_0_3'])} |")
    lines.append("")

    lines.append("## 3. Dataset split")
    lines.append("")
    lines.append("| dataset | n | critical | follows ink | border artifact | mean quality | failure stages |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    for dataset, s in by_dataset.items():
        stages = ", ".join(f"{k}:{v}" for k, v in sorted(s["failure_stage_counts"].items()))
        lines.append(
            f"| `{dataset}` | {s['n']} | "
            f"{fmt(s['critical_topology_error_rate'])} | "
            f"{fmt(s['skeleton_follows_ink_rate'])} | "
            f"{fmt(s['border_artifact_rate'])} | "
            f"{fmt(s['mean_graph_quality_0_3'])} | "
            f"{stages} |"
        )

    lines.append("")
    lines.append("## 4. HKR + Cyrillic structural preservation")
    lines.append("")
    lines.append("| subset | n | critical | follows ink | mean quality |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| `hkr_words + cyrillic_handwriting` | {hkr_cyr_n} | "
        f"{fmt(hkr_cyr_critical)} | "
        f"{fmt(hkr_cyr_follows)} | "
        f"{fmt(hkr_cyr_quality)} |"
    )
    lines.append("")
    lines.append(
        "This is the subset that should be used for the H2-v1 graph-topology preservation claim."
    )
    lines.append("")

    lines.append("## 5. School notebooks exception")
    lines.append("")
    lines.append(
        "The school-notebooks subset should be reported separately. Manual staging found "
        "that all audited school-notebooks samples were affected by binarization-stage "
        "failure with border artifacts. The issue is not the canonical graph abstraction itself, "
        "but the upstream conversion from cropped image to foreground mask."
    )
    lines.append("")
    lines.append("```text")
    lines.append("school_notebooks_clean:")
    lines.append(f"  n = {school.get('n')}")
    lines.append(f"  critical topology error rate = {fmt(school.get('critical_topology_error_rate', 0.0))}")
    lines.append(f"  skeleton follows ink rate = {fmt(school.get('skeleton_follows_ink_rate', 0.0))}")
    lines.append(f"  border artifact rate = {fmt(school.get('border_artifact_rate', 0.0))}")
    lines.append("  failure stage = binarization")
    lines.append("```")
    lines.append("")

    lines.append("## 6. Border suppression v1")
    lines.append("")
    lines.append(
        "A simple border-connected component suppression check was attempted and rejected. "
        "Visual inspection showed that it either failed to remove the artifact or removed "
        "handwriting together with the border. Therefore it is not integrated."
    )
    lines.append("")
    lines.append(f"See: `{args.border_report}`")
    lines.append("")

    lines.append("## 7. Consequence")
    lines.append("")
    lines.append("For the thesis/report:")
    lines.append("")
    lines.append("- do not claim that H2 is fully solved across all datasets;")
    lines.append("- do claim partial H2 support on HKR/Cyrillic audited samples;")
    lines.append("- report school-notebooks as a preprocessing limitation and failure mode;")
    lines.append("- do not aggregate school-notebooks into pure graph-topology error statistics;")
    lines.append("- do not tune HTR architecture based on this finding.")
    lines.append("")

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()
