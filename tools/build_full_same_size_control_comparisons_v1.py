from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("outputs/htr_publication_v3/full_same_size_controls")
EVAL_ROOT = ROOT / "eval_fixed_m04"
OUT_ROOT = ROOT / "paired_fixed_m04"
SEEDS = [42, 43, 44]

PAIRS = [
    ("line_vs_base", "tri10k_base", "line_context_10k"),
    ("line_vs_random", "random_crops_10k_control", "line_context_10k"),
    ("line_vs_school_words", "school_words_10k_control", "line_context_10k"),
    ("random_vs_base", "tri10k_base", "random_crops_10k_control"),
    ("school_words_vs_base", "tri10k_base", "school_words_10k_control"),
]


def predictions_path(variant: str, seed: int) -> Path:
    return EVAL_ROOT / f"{variant}_seed{seed}_test" / "predictions.jsonl"


def run_pair(key: str, baseline: str, augmented: str, seed: int) -> dict:
    baseline_predictions = predictions_path(baseline, seed)
    augmented_predictions = predictions_path(augmented, seed)
    out_json = OUT_ROOT / f"{key}_seed{seed}.json"
    out_md = OUT_ROOT / f"{key}_seed{seed}.md"

    status = {
        "key": key,
        "seed": seed,
        "baseline": baseline,
        "augmented": augmented,
        "baseline_predictions": str(baseline_predictions),
        "augmented_predictions": str(augmented_predictions),
        "out_json": str(out_json),
        "out_md": str(out_md),
        "status": "pending",
    }

    if out_json.exists() and out_md.exists():
        status["status"] = "exists"
        return status

    if not baseline_predictions.exists() or not augmented_predictions.exists():
        status["status"] = "missing_predictions"
        return status

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/compare_htr_paired_predictions_iter2.py",
        "--baseline_predictions",
        str(baseline_predictions),
        "--augmented_predictions",
        str(augmented_predictions),
        "--quality_root",
        "data/experiments/school_quality_v1",
        "--out_json",
        str(out_json),
        "--out_md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, check=False)
    status["status"] = "complete" if proc.returncode == 0 else "failed"
    status["returncode"] = proc.returncode
    return status


def main() -> None:
    statuses = [
        run_pair(key, baseline, augmented, seed)
        for seed in SEEDS
        for key, baseline, augmented in PAIRS
    ]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "comparison_status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "out_status": str(OUT_ROOT / "comparison_status.json"),
        "complete": sum(1 for row in statuses if row["status"] in {"complete", "exists"}),
        "missing": sum(1 for row in statuses if row["status"] == "missing_predictions"),
        "failed": sum(1 for row in statuses if row["status"] == "failed"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
