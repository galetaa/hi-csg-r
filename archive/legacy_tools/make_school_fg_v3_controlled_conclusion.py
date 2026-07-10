from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(x: Any, digits: int = 5) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def rel_delta(new: Any, old: Any) -> float | None:
    try:
        old_f = float(old)
        new_f = float(new)
        return (new_f - old_f) / old_f
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def grouped_cer(run: dict[str, Any], dataset: str) -> Any:
    return run.get("grouped", {}).get(dataset, {}).get("cer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross_eval_json", required=True)
    parser.add_argument("--h2_update_json", required=True)
    parser.add_argument("--h3_after_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    cross = load(args.cross_eval_json)
    h2 = load(args.h2_update_json)
    h3 = load(args.h3_after_json)

    runs = cross["runs"]

    old_old = runs["old_model_old_features"]
    old_new = runs["old_model_new_features"]
    new_old = runs["new_model_old_features"]
    new_new = runs["new_model_new_features"]

    old_feature_delta = float(old_new["cer"]) - float(old_old["cer"])
    new_feature_delta = float(new_new["cer"]) - float(new_old["cer"])

    training_delta_old = float(new_old["cer"]) - float(old_old["cer"])
    training_delta_new = float(new_new["cer"]) - float(old_new["cer"])

    school_old_delta = (
        float(grouped_cer(old_new, "school_notebooks_clean"))
        - float(grouped_cer(old_old, "school_notebooks_clean"))
    )
    school_new_delta = (
        float(grouped_cer(new_new, "school_notebooks_clean"))
        - float(grouped_cer(new_old, "school_notebooks_clean"))
    )

    h3_structural = (
        h3.get("best_by_feature_set", {}).get("structural_core", {})
    )

    result = {
        "verdict": {
            "school_dark_auto": "accepted_for_graph_extraction",
            "graph_fusion_v3_checkpoint": "rejected",
            "graph_fusion_v2_checkpoint": "retained",
            "h1_historical_results": "unchanged",
            "domain_specific_normalization": "not_justified",
        },
        "controlled_effects": {
            "v3_features_on_old_checkpoint_cer_delta": old_feature_delta,
            "v3_features_on_new_checkpoint_cer_delta": new_feature_delta,
            "new_checkpoint_on_old_features_cer_delta": training_delta_old,
            "new_checkpoint_on_v3_features_cer_delta": training_delta_new,
            "school_v3_features_on_old_checkpoint_cer_delta": school_old_delta,
            "school_v3_features_on_new_checkpoint_cer_delta": school_new_delta,
        },
        "h2_update": h2,
        "h3_after_v3": h3_structural,
    }

    lines: list[str] = []
    lines.append("# School foreground v3 — controlled conclusion")
    lines.append("")
    lines.append("## 1. Final verdict")
    lines.append("")
    lines.append("| component | decision |")
    lines.append("|---|---|")
    lines.append("| `school_dark_auto` | accepted for future School Notebooks graph extraction |")
    lines.append("| graph-fusion v3 checkpoint | rejected |")
    lines.append("| graph-fusion v2 checkpoint | retained as final graph-vector model |")
    lines.append("| historical H1 results | unchanged |")
    lines.append("| domain-specific normalization | not justified by current evidence |")
    lines.append("")

    lines.append("## 2. Controlled cross-evaluation")
    lines.append("")
    lines.append("| checkpoint | old features CER | v3 features CER | feature ΔCER |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| graph-v2 | {fmt(old_old['cer'])} | {fmt(old_new['cer'])} | "
        f"{old_feature_delta:+.5f} |"
    )
    lines.append(
        f"| graph-v3 | {fmt(new_old['cer'])} | {fmt(new_new['cer'])} | "
        f"{new_feature_delta:+.5f} |"
    )
    lines.append("")

    lines.append("## 3. Training-run effect")
    lines.append("")
    lines.append("| shared manifest | graph-v3 − graph-v2 CER |")
    lines.append("|---|---:|")
    lines.append(f"| old features | {training_delta_old:+.5f} |")
    lines.append(f"| v3 features | {training_delta_new:+.5f} |")
    lines.append("")
    lines.append(
        "The new checkpoint is worse under both manifests. Therefore its degradation "
        "cannot be attributed to foreground v3."
    )
    lines.append("")

    lines.append("## 4. Dataset-local feature effect")
    lines.append("")
    lines.append("| checkpoint | School Notebooks ΔCER from v3 features |")
    lines.append("|---|---:|")
    lines.append(f"| graph-v2 | {school_old_delta:+.5f} |")
    lines.append(f"| graph-v3 | {school_new_delta:+.5f} |")
    lines.append("")
    lines.append(
        "HKR and Cyrillic CER remain unchanged under inference-time feature replacement. "
        "The small benefit is localized to School Notebooks, the only dataset whose "
        "foreground extraction was changed."
    )
    lines.append("")

    lines.append("## 5. H2 result")
    lines.append("")
    v3 = h2["foreground_v3"]
    lines.append(f"- audited samples: {v3['n']}")
    lines.append(f"- good fix rate: {v3['good_fix_rate']:.3f}")
    lines.append(f"- partial fix rate: {v3['partial_fix_rate']:.3f}")
    lines.append(f"- erased-ink rate: {v3['real_ink_erased_rate']:.3f}")
    lines.append(
        f"- skeleton-follows-ink rate after repair: "
        f"{v3['skeleton_follows_ink_after_rate']:.3f}"
    )
    lines.append("")

    lines.append("## 6. H3 result")
    lines.append("")
    if h3_structural:
        lines.append(
            f"The best structural-core result remains localized to "
            f"`{h3_structural.get('group', 'unknown')}` with "
            f"ROC-AUC {fmt(h3_structural.get('roc_auc'), 4)}, "
            f"PR-AUC {fmt(h3_structural.get('pr_auc'), 4)}, and "
            f"top-20% precision {fmt(h3_structural.get('top20_precision'), 4)}."
        )
    lines.append("")
    lines.append(
        "Foreground repair therefore does not materially change the main H3 conclusion."
    )
    lines.append("")

    lines.append("## 7. Scientific conclusion")
    lines.append("")
    lines.append(
        "Foreground v3 substantially improves the visible structural representation "
        "of audited School Notebooks samples. Cross-evaluation shows that the repaired "
        "features are compatible with the existing graph-fusion model and provide a "
        "small dataset-local CER improvement. A newly trained graph-fusion checkpoint "
        "nevertheless performs worse under both old and repaired feature manifests. "
        "Thus, improved visible graph quality does not automatically translate into "
        "improved HTR accuracy."
    )
    lines.append("")

    lines.append("## 8. Next validation requirement")
    lines.append("")
    lines.append(
        "Because the original 23 samples came from a diagnostic CER/risk-quadrant subset, "
        "foreground v3 must next be validated on an independently sampled random School "
        "Notebooks subset before population-level repair rates are reported."
    )

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()