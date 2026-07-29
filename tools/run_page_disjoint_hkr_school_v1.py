from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


DATA_ROOT = Path("data/experiments/htr_publication_v3")
OUT_ROOT = Path("outputs/htr_publication_v3/page_disjoint_hkr_school_v1")
FIXED_TEST_PENALTY = -0.4

VARIANTS = {
    "page_base": DATA_ROOT / "page_disjoint_hkr_school_base_v1",
    "page_line_10k": DATA_ROOT / "page_disjoint_hkr_school_plus_lines_10k_v1",
    "page_random_crops_8k_control": DATA_ROOT / "page_disjoint_hkr_school_random_crops_8k_control_v1",
    "page_school_words_8k_control": DATA_ROOT / "page_disjoint_hkr_school_school_words_8k_control_v1",
}


def run(cmd: list[str], *, dry_run: bool) -> int:
    print(json.dumps({"cmd": cmd, "dry_run": dry_run}, ensure_ascii=False), flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.run(cmd, check=False, env=env)
    return int(proc.returncode)


def checkpoint_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    epoch = ckpt.get("epoch")
    return int(epoch) if epoch is not None else None


def training_config_mismatches(
    *,
    out_dir: Path,
    batch_size: int,
    prefetch_factor: int,
    bucket_by_width: bool,
) -> dict[str, dict[str, Any]]:
    path = out_dir / "config.json"
    if not path.exists() or not (out_dir / "last.pt").exists():
        return {}

    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "batch_size": batch_size,
        "prefetch_factor": prefetch_factor,
        "bucket_by_width": bucket_by_width,
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual_value = config.get(key)
        if actual_value != expected_value:
            mismatches[key] = {
                "actual": actual_value,
                "expected": expected_value,
            }
    return mismatches


def vocab_chars(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return set(obj["char_to_idx"]) - {obj.get("blank_token", "<blank>")}


def manifest_oov_examples(manifest_path: Path, vocab_path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    chars = vocab_chars(vocab_path)
    examples = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(set(str(row.get("text", ""))) - chars)
            if missing:
                examples.append({
                    "line_no": line_no,
                    "sample_id": row.get("sample_id"),
                    "text": row.get("text"),
                    "missing_characters": missing,
                })
                if len(examples) >= limit:
                    break
    return examples


def train_cmd(
    *,
    manifest_root: Path,
    out_dir: Path,
    seed: int,
    epochs: int,
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    bucket_by_width: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "tools/train_crnn_ctc.py",
        "--train_manifest",
        str(manifest_root / "train.jsonl"),
        "--val_manifest",
        str(manifest_root / "val.jsonl"),
        "--vocab",
        str(manifest_root / "vocab.json"),
        "--out_dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--prefetch_factor",
        str(prefetch_factor),
        "--hidden_size",
        "256",
        "--lstm_layers",
        "2",
        "--dropout",
        "0.1",
        "--lr",
        "0.0005",
        "--weight_decay",
        "0.0001",
        "--grad_clip",
        "5.0",
        "--log_every",
        "50",
        "--seed",
        str(seed),
        "--blank_bias_init",
        "-1.0",
        "--height_bins",
        "4",
        "--feature_size",
        "256",
        "--blank_logit_penalty_start",
        "-2.0",
        "--blank_logit_penalty_end",
        "-0.4",
    ]
    if bucket_by_width:
        cmd.append("--bucket_by_width")
    last = out_dir / "last.pt"
    if last.exists():
        cmd.extend(["--resume", str(last)])
    return cmd


def eval_cmd(*, manifest_root: Path, checkpoint: Path, out_dir: Path, num_workers: int) -> list[str]:
    return [
        sys.executable,
            "tools/evaluate_crnn_ctc.py",
        "--manifest",
        str(manifest_root / "test.jsonl"),
        "--vocab",
        str(manifest_root / "vocab.json"),
        "--checkpoint",
        str(checkpoint),
        "--out_dir",
        str(out_dir),
        "--batch_size",
        "16",
        "--num_workers",
        str(num_workers),
        "--blank_logit_penalty",
        str(FIXED_TEST_PENALTY),
    ]


def run_variant_seed(
    *,
    variant: str,
    seed: int,
    epochs: int,
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    bucket_by_width: bool,
    dry_run: bool,
    skip_training: bool,
    skip_eval: bool,
) -> dict[str, Any]:
    manifest_root = VARIANTS[variant]
    ckpt_dir = OUT_ROOT / "checkpoints" / f"{variant}_seed{seed}"
    eval_dir = OUT_ROOT / "eval_fixed_m04" / f"{variant}_seed{seed}_test"
    status: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "manifest_root": str(manifest_root),
        "checkpoint_dir": str(ckpt_dir),
        "eval_dir": str(eval_dir),
        "target_epochs": epochs,
        "batch_size": batch_size,
        "prefetch_factor": prefetch_factor,
        "bucket_by_width": bucket_by_width,
        "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "last_epoch_before": checkpoint_epoch(ckpt_dir / "last.pt"),
        "train_returncode": None,
        "eval_returncode": None,
        "status": "pending",
    }

    if not manifest_root.exists():
        status["status"] = "missing_manifest_root"
        return status

    mismatches = training_config_mismatches(
        out_dir=ckpt_dir,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        bucket_by_width=bucket_by_width,
    )
    if mismatches:
        status["status"] = "checkpoint_config_mismatch"
        status["config_mismatches"] = mismatches
        status["resolution"] = (
            "Archive or remove the incompatible checkpoint directory before restarting, "
            "or rerun with matching memory/training parameters."
        )
        return status

    train_oov = manifest_oov_examples(manifest_root / "train.jsonl", manifest_root / "vocab.json")
    val_oov = manifest_oov_examples(manifest_root / "val.jsonl", manifest_root / "vocab.json")
    test_oov = manifest_oov_examples(manifest_root / "test.jsonl", manifest_root / "vocab.json")
    if train_oov or val_oov or test_oov:
        status["status"] = "manifest_oov"
        status["manifest_oov_examples"] = {
            "train": train_oov,
            "val": val_oov,
            "test": test_oov,
        }
        return status

    if not skip_training:
        last_epoch = checkpoint_epoch(ckpt_dir / "last.pt")
        best_exists = (ckpt_dir / "best.pt").exists()
        if last_epoch is None or last_epoch < epochs or not best_exists:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            status["train_returncode"] = run(
                train_cmd(
                    manifest_root=manifest_root,
                    out_dir=ckpt_dir,
                    seed=seed,
                    epochs=epochs,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    prefetch_factor=prefetch_factor,
                    bucket_by_width=bucket_by_width,
                ),
                dry_run=dry_run,
            )
            if status["train_returncode"] != 0:
                status["status"] = "train_failed"
                return status
        else:
            status["train_returncode"] = 0
            status["train_skipped"] = "complete"

    status["last_epoch_after"] = checkpoint_epoch(ckpt_dir / "last.pt")
    status["best_exists"] = (ckpt_dir / "best.pt").exists()

    if not skip_eval:
        summary = eval_dir / "summary.json"
        if summary.exists():
            status["eval_returncode"] = 0
            status["eval_skipped"] = "summary exists"
        elif not (ckpt_dir / "best.pt").exists():
            status["eval_skipped"] = "missing best.pt"
        else:
            status["eval_returncode"] = run(
                eval_cmd(
                    manifest_root=manifest_root,
                    checkpoint=ckpt_dir / "best.pt",
                    out_dir=eval_dir,
                    num_workers=num_workers,
                ),
                dry_run=dry_run,
            )
            if status["eval_returncode"] != 0:
                status["status"] = "eval_failed"
                return status

    status["status"] = "complete"
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--no_bucket_by_width", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    statuses = []
    status_path = OUT_ROOT / "run_status.json"
    for variant in args.variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        for seed in args.seeds:
            manifest_root = VARIANTS[variant]
            ckpt_dir = OUT_ROOT / "checkpoints" / f"{variant}_seed{seed}"
            eval_dir = OUT_ROOT / "eval_fixed_m04" / f"{variant}_seed{seed}_test"
            statuses.append({
                "variant": variant,
                "seed": seed,
                "manifest_root": str(manifest_root),
                "checkpoint_dir": str(ckpt_dir),
                "eval_dir": str(eval_dir),
                "target_epochs": args.epochs,
                "batch_size": args.batch_size,
                "prefetch_factor": args.prefetch_factor,
                "bucket_by_width": not args.no_bucket_by_width,
                "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                "last_epoch_before": checkpoint_epoch(ckpt_dir / "last.pt"),
                "train_returncode": None,
                "eval_returncode": None,
                "status": "running",
            })
            status_path.write_text(json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8")

            statuses[-1] = run_variant_seed(
                variant=variant,
                seed=seed,
                epochs=args.epochs,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                prefetch_factor=args.prefetch_factor,
                bucket_by_width=not args.no_bucket_by_width,
                dry_run=args.dry_run,
                skip_training=args.skip_training,
                skip_eval=args.skip_eval,
            )
            status_path.write_text(json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "out_root": str(OUT_ROOT),
        "status_path": str(status_path),
        "statuses": statuses,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
