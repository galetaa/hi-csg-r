from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = [
    "hkr_words",
    "cyrillic_handwriting",
    "school_notebooks_clean",
]


def load(path: str) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj.get("metrics", obj)


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return "n/a"


def diff(a: Any, b: Any) -> str:
    try:
        return f"{float(a) - float(b):+.5f}"
    except (TypeError, ValueError):
        return "n/a"


def grouped_cer(summary: dict[str, Any], dataset: str) -> Any:
    return summary.get("grouped", {}).get(dataset, {}).get("cer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_old", required=True)
    parser.add_argument("--old_new", required=True)
    parser.add_argument("--new_old", required=True)
    parser.add_argument("--new_new", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    runs = {
        "old_model_old_features": load(args.old_old),
        "old_model_new_features": load(args.old_new),
        "new_model_old_features": load(args.new_old),
        "new_model_new_features": load(args.new_new),
    }

    old_old = runs["old_model_old_features"]
    old_new = runs["old_model_new_features"]
    new_old = runs["new_model_old_features"]
    new_new = runs["new_model_new_features"]

    result = {
        "runs": runs,
        "contrasts": {
            "feature_swap_on_old_model": {
                "overall_cer_delta": (
                    float(old_new["cer"]) - float(old_old["cer"])
                ),
                "meaning": "Effect of replacing old features by v3 features at inference for the old model.",
            },
            "feature_swap_on_new_model": {
                "overall_cer_delta": (
                    float(new_new["cer"]) - float(new_old["cer"])
                ),
                "meaning": "Dependence of the new model on v3 rather than old features.",
            },
            "training_change_on_old_features": {
                "overall_cer_delta": (
                    float(new_old["cer"]) - float(old_old["cer"])
                ),
                "meaning": "Difference between training runs while both are evaluated with old features.",
            },
            "training_change_on_new_features": {
                "overall_cer_delta": (
                    float(new_new["cer"]) - float(old_new["cer"])
                ),
                "meaning": "Difference between training runs while both are evaluated with v3 features.",
            },
        },
    }

    lines: list[str] = []
    lines.append("# Graph feature cross-evaluation")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| checkpoint | feature manifest | CER | WER | exact |")
    lines.append("|---|---|---:|---:|---:|")

    labels = {
        "old_model_old_features": ("graph-v2", "old"),
        "old_model_new_features": ("graph-v2", "school-fg-v3"),
        "new_model_old_features": ("graph-v3", "old"),
        "new_model_new_features": ("graph-v3", "school-fg-v3"),
    }

    for key, summary in runs.items():
        model, features = labels[key]
        lines.append(
            f"| `{model}` | `{features}` | "
            f"{fmt(summary.get('cer'))} | "
            f"{fmt(summary.get('wer'))} | "
            f"{fmt(summary.get('exact'))} |"
        )

    lines.append("")
    lines.append("## By dataset")
    lines.append("")
    lines.append(
        "| dataset | old model + old fg | old model + v3 fg | "
        "new model + old fg | new model + v3 fg |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    for dataset in DATASETS:
        lines.append(
            f"| `{dataset}` | "
            f"{fmt(grouped_cer(old_old, dataset))} | "
            f"{fmt(grouped_cer(old_new, dataset))} | "
            f"{fmt(grouped_cer(new_old, dataset))} | "
            f"{fmt(grouped_cer(new_new, dataset))} |"
        )

    lines.append("")
    lines.append("## Controlled contrasts")
    lines.append("")
    lines.append("| contrast | ΔCER | interpretation |")
    lines.append("|---|---:|---|")
    lines.append(
        "| v3 features on old checkpoint | "
        f"{diff(old_new.get('cer'), old_old.get('cer'))} | "
        "Inference-time feature-distribution effect. |"
    )
    lines.append(
        "| v3 features on new checkpoint | "
        f"{diff(new_new.get('cer'), new_old.get('cer'))} | "
        "Whether the new model benefits from its matching v3 features. |"
    )
    lines.append(
        "| new vs old checkpoint on old features | "
        f"{diff(new_old.get('cer'), old_old.get('cer'))} | "
        "Training-run/model difference with a shared old manifest. |"
    )
    lines.append(
        "| new vs old checkpoint on v3 features | "
        f"{diff(new_new.get('cer'), old_new.get('cer'))} | "
        "Training-run/model difference with a shared v3 manifest. |"
    )

    lines.append("")
    lines.append("## Decision rules")
    lines.append("")
    lines.append(
        "- If the new checkpoint is worse with both manifests, the main issue is the new training run or optimization, not foreground v3 alone."
    )
    lines.append(
        "- If each checkpoint works best only with its matching manifest, the feature distribution changed and the graph branch is sensitive to that shift."
    )
    lines.append(
        "- If v3 features hurt both checkpoints, visual graph repair does not translate into useful graph-fusion features."
    )
    lines.append(
        "- If v3 features improve school-notebooks but hurt the other datasets, global normalization or shared fusion is causing cross-domain interference."
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