from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


OUT_ROOT = Path("outputs/htr_publication_v3/dose_response_fixed_m04")
BLANK_LOGIT_PENALTY = "-0.4"

RUNS = [
    {
        "key": "baseline_0_lines",
        "manifest_root": Path("data/experiments/htr_graph_v1/graph_ready/tri10k_mixed"),
        "checkpoint": Path("outputs/htr_graph_v1/tri10k_image_only_v1/best.pt"),
        "line_train_n": 0,
    },
    {
        "key": "plus_2k_lines",
        "manifest_root": Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_2k_context_v1"),
        "checkpoint": Path("outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_2k_context_v1/best.pt"),
        "line_train_n": 1998,
    },
    {
        "key": "plus_5k_lines",
        "manifest_root": Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_5k_context_v1"),
        "checkpoint": Path("outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_5k_context_v1/best.pt"),
        "line_train_n": 4999,
    },
    {
        "key": "plus_10k_lines",
        "manifest_root": Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1"),
        "checkpoint": Path("outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1/best.pt"),
        "line_train_n": 9998,
    },
]


def run_eval(run: dict[str, object]) -> dict[str, object]:
    key = str(run["key"])
    manifest_root = Path(run["manifest_root"])
    checkpoint = Path(run["checkpoint"])
    out_dir = OUT_ROOT / f"{key}_test"
    summary_path = out_dir / "summary.json"
    predictions_path = out_dir / "predictions.jsonl"

    status: dict[str, object] = {
        "key": key,
        "manifest": str(manifest_root / "test.jsonl"),
        "vocab": str(manifest_root / "vocab.json"),
        "checkpoint": str(checkpoint),
        "out_dir": str(out_dir),
        "line_train_n": run["line_train_n"],
        "blank_logit_penalty": float(BLANK_LOGIT_PENALTY),
        "status": "pending",
    }

    if summary_path.exists() and predictions_path.exists():
        status["status"] = "exists"
        return status

    missing = [
        str(path)
        for path in [manifest_root / "test.jsonl", manifest_root / "vocab.json", checkpoint]
        if not path.exists()
    ]
    if missing:
        status["status"] = "missing_input"
        status["missing"] = missing
        return status

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        "tools/evaluate_crnn_ctc.py",
        "--manifest",
        str(manifest_root / "test.jsonl"),
        "--vocab",
        str(manifest_root / "vocab.json"),
        "--checkpoint",
        str(checkpoint),
        "--out_dir",
        str(out_dir),
        "--blank_logit_penalty",
        BLANK_LOGIT_PENALTY,
        "--batch_size",
        "64",
        "--num_workers",
        "4",
    ]
    proc = subprocess.run(cmd, check=False)
    status["returncode"] = proc.returncode
    status["status"] = "complete" if proc.returncode == 0 else "failed"
    return status


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    statuses = [run_eval(run) for run in RUNS]
    status_path = OUT_ROOT / "run_status.json"
    status_path.write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "out_status": str(status_path),
        "complete_or_exists": sum(1 for row in statuses if row["status"] in {"complete", "exists"}),
        "failed": [row for row in statuses if row["status"] == "failed"],
        "missing_input": [row for row in statuses if row["status"] == "missing_input"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
