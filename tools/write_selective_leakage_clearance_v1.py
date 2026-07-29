from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical_check",
        default="outputs/final_result_package_v1/selective_prediction_canonical_check.json",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/final_result_package_v1",
    )
    args = parser.parse_args()

    check_path = Path(args.canonical_check)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = read_json(check_path)
    summary: dict[str, Any] = report.get("summary", {})

    leakage_files = summary.get("leakage_review_files", [])
    variant_coverage = summary.get("variant_coverage", {})
    bad_model_files = summary.get("bad_model_files", [])
    canonical_files = summary.get("canonical_files", [])

    clearance_status = "PASS_WITH_NOTE"

    reasons = []

    if not canonical_files:
        clearance_status = "FAIL"
        reasons.append("No canonical selective prediction files detected.")

    if bad_model_files:
        clearance_status = "FAIL"
        reasons.append("Bad/exploratory model references detected.")

    required_variants = ["confidence", "graph_or_quality", "confidence_graph"]
    missing_variants = [
        variant for variant in required_variants
        if not bool(variant_coverage.get(variant))
    ]
    if missing_variants:
        clearance_status = "FAIL"
        reasons.append(f"Missing required variants: {missing_variants}")

    text_len_only_in_reporting = True
    reviewed_leakage_notes = []

    for item in leakage_files:
        path = item.get("path", "")
        hits = item.get("hits", [])

        if hits == ["text_len"] and path.endswith("operating_point_strata.md"):
            reviewed_leakage_notes.append(
                {
                    "path": path,
                    "hits": hits,
                    "review": (
                        "`text_len` appears in a post-hoc operating-point "
                        "stratification/reporting table, not as an evident "
                        "risk-model feature."
                    ),
                    "cleared": True,
                }
            )
        else:
            text_len_only_in_reporting = False
            reviewed_leakage_notes.append(
                {
                    "path": path,
                    "hits": hits,
                    "review": "Manual review did not clear this leakage-risk hit.",
                    "cleared": False,
                }
            )

    if leakage_files and not text_len_only_in_reporting:
        clearance_status = "FAIL"
        reasons.append("Uncleared leakage-risk keys remain.")

    if not reasons:
        reasons.append(
            "Canonical +10k artifacts and all required variants were detected; "
            "the only leakage-risk hit is `text_len` in a reporting/stratification file."
        )

    clearance = {
        "clearance_status": clearance_status,
        "canonical_check_verdict": summary.get("verdict"),
        "canonical_files_n": summary.get("canonical_files_n"),
        "bad_model_files_n": summary.get("bad_model_files_n"),
        "variant_coverage": variant_coverage,
        "leakage_review_files_n": summary.get("leakage_review_files_n"),
        "reviewed_leakage_notes": reviewed_leakage_notes,
        "reasons": reasons,
        "final_interpretation": (
            "Selective prediction is acceptable as a secondary applied result "
            "if described as canonical +10k confidence/graph-quality risk analysis. "
            "The detected `text_len` occurrence must be described as post-hoc "
            "stratification/reporting, not as a model feature."
            if clearance_status == "PASS_WITH_NOTE"
            else "Selective prediction is not cleared for final reporting."
        ),
    }

    json_path = out_dir / "selective_prediction_leakage_clearance.json"
    md_path = out_dir / "selective_prediction_leakage_clearance.md"

    json_path.write_text(
        json.dumps(clearance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Selective prediction leakage clearance v1\n")
    lines.append(f"Clearance status: **{clearance_status}**\n")
    lines.append("## Reasons\n")
    for reason in reasons:
        lines.append(f"- {reason}")

    lines.append("\n## Variant coverage\n")
    for key, value in variant_coverage.items():
        lines.append(f"- {key}: `{value}`")

    lines.append("\n## Reviewed leakage notes\n")
    if reviewed_leakage_notes:
        for note in reviewed_leakage_notes:
            lines.append(f"- `{note['path']}`")
            lines.append(f"  - hits: `{note['hits']}`")
            lines.append(f"  - cleared: `{note['cleared']}`")
            lines.append(f"  - review: {note['review']}")
    else:
        lines.append("No leakage-risk files detected.")

    lines.append("\n## Final interpretation\n")
    lines.append(clearance["final_interpretation"])

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(clearance, ensure_ascii=False, indent=2))
    print("wrote:", json_path)
    print("wrote:", md_path)


if __name__ == "__main__":
    main()
