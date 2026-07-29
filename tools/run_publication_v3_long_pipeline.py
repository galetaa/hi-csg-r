from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


OUT_ROOT = Path("outputs/htr_publication_v3")
LONG_ROOT = OUT_ROOT / "long_jobs"
STATUS_PATH = LONG_ROOT / "pipeline_status.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_status(statuses: list[dict[str, Any]]) -> None:
    LONG_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_stage(statuses: list[dict[str, Any]], name: str, cmd: list[str]) -> bool:
    row = {
        "name": name,
        "cmd": cmd,
        "started_at": now(),
        "finished_at": None,
        "returncode": None,
        "status": "running",
    }
    statuses.append(row)
    write_status(statuses)
    print(json.dumps(row, ensure_ascii=False), flush=True)

    proc = subprocess.run(cmd, check=False)
    row["finished_at"] = now()
    row["returncode"] = proc.returncode
    row["status"] = "complete" if proc.returncode == 0 else "failed"
    write_status(statuses)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return proc.returncode == 0


def main() -> None:
    LONG_ROOT.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, Any]] = []
    if STATUS_PATH.exists():
        statuses = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        changed = False
        for row in statuses:
            if row.get("status") == "running":
                row["status"] = "interrupted_on_resume"
                row["finished_at"] = row.get("finished_at") or now()
                row["returncode"] = row.get("returncode")
                changed = True
        if changed:
            write_status(statuses)

    stages = [
        (
            "full_same_size_controls_random_and_school_words",
            [
                sys.executable,
                "tools/run_full_same_size_controls_v1.py",
                "--variants",
                "random_crops_10k_control",
                "school_words_10k_control",
                "--seeds",
                "42",
                "43",
                "44",
            ],
        ),
        (
            "paired_comparisons_after_controls",
            [
                sys.executable,
                "tools/build_full_same_size_control_comparisons_v1.py",
            ],
        ),
        (
            "train_trocr_finetuned_tri10k_base",
            [
                sys.executable,
                "tools/train_trocr_baseline_v1.py",
                "--train_manifest",
                "data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/train.jsonl",
                "--val_manifest",
                "data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/val.jsonl",
                "--out_dir",
                "outputs/htr_publication_v3/trocr_finetuned_tri10k_base",
                "--model_id",
                "microsoft/trocr-base-handwritten",
                "--local_files_only",
                "--epochs",
                "10",
                "--batch_size",
                "1",
                "--grad_accum_steps",
                "16",
                "--lr",
                "5e-5",
                "--weight_decay",
                "0.01",
                "--max_target_length",
                "96",
                "--max_new_tokens",
                "96",
                "--num_beams",
                "1",
                "--seed",
                "42",
                "--fp16",
                "--gradient_checkpointing",
                "--freeze_encoder",
                "--log_every",
                "100",
            ],
        ),
        (
            "eval_trocr_finetuned_tri10k_base_test",
            [
                sys.executable,
                "tools/evaluate_trocr_baseline_v1.py",
                "--manifest",
                "data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/test.jsonl",
                "--model_id",
                "outputs/htr_publication_v3/trocr_finetuned_tri10k_base/best",
                "--out_dir",
                "outputs/htr_publication_v3/external_trocr_finetuned_tri10k_base_test",
                "--local_files_only",
                "--batch_size",
                "4",
                "--num_beams",
                "1",
                "--max_new_tokens",
                "96",
                "--fp16",
            ],
        ),
        (
            "rebuild_publication_v3_status_report",
            [
                sys.executable,
                "tools/build_publication_v3_status_report.py",
            ],
        ),
        (
            "write_publication_v3_repro_snapshot",
            [
                sys.executable,
                "tools/write_publication_repro_snapshot_v1.py",
                "--out_dir",
                "outputs/htr_publication_v3",
            ],
        ),
    ]

    completed_names = {
        row["name"]
        for row in statuses
        if row.get("status") == "complete"
    }

    for name, cmd in stages:
        if name in completed_names:
            print(f"skip complete stage: {name}", flush=True)
            continue
        ok = run_stage(statuses, name, cmd)
        if not ok:
            print(f"stopping after failed stage: {name}", flush=True)
            break

    print("pipeline status:", STATUS_PATH, flush=True)


if __name__ == "__main__":
    main()
