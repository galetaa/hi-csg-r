from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


SEEDS = [42, 43, 44]
EPOCHS = 80
FIXED_TEST_PENALTY = -0.4


VARIANTS = {
    "tri10k_base": {
        "kind": "existing",
        "manifest_root": "data/experiments/htr_graph_v1/graph_ready/tri10k_mixed",
        "checkpoint_dirs": {
            42: "outputs/htr_graph_v1/tri10k_image_only_v1",
            43: "outputs/htr_graph_v1/tri10k_image_only_v1_seed43",
            44: "outputs/htr_graph_v1/tri10k_image_only_v1_seed44",
        },
    },
    "line_context_10k": {
        "kind": "existing",
        "manifest_root": "data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1",
        "checkpoint_dirs": {
            42: "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1",
            43: "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1_seed43",
            44: "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1_seed44",
        },
    },
    "random_crops_10k_control": {
        "kind": "train",
        "manifest_root": "data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_random_crops_10k_control_v1",
    },
    "school_words_10k_control": {
        "kind": "train",
        "manifest_root": "data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_words_10k_control_v1",
    },
}


def run(cmd: list[str], *, dry_run: bool) -> int:
    print(json.dumps({"cmd": cmd, "dry_run": dry_run}, ensure_ascii=False), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def read_checkpoint_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    epoch = ckpt.get("epoch")
    return int(epoch) if epoch is not None else None


def control_checkpoint_dir(out_root: Path, variant: str, seed: int) -> Path:
    return out_root / "checkpoints" / f"{variant}_seed{seed}"


def eval_dir(out_root: Path, variant: str, seed: int) -> Path:
    return out_root / "eval_fixed_m04" / f"{variant}_seed{seed}_test"


def manifest_paths(manifest_root: Path) -> dict[str, Path]:
    return {
        "train": manifest_root / "train.jsonl",
        "val": manifest_root / "val.jsonl",
        "test": manifest_root / "test.jsonl",
        "vocab": manifest_root / "vocab.json",
    }


def train_command(
    *,
    manifest_root: Path,
    out_dir: Path,
    seed: int,
) -> list[str]:
    paths = manifest_paths(manifest_root)
    cmd = [
        sys.executable,
        "tools/train_crnn_ctc.py",
        "--train_manifest",
        str(paths["train"]),
        "--val_manifest",
        str(paths["val"]),
        "--vocab",
        str(paths["vocab"]),
        "--out_dir",
        str(out_dir),
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        "16",
        "--num_workers",
        "4",
        "--prefetch_factor",
        "4",
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
    last = out_dir / "last.pt"
    if last.exists():
        cmd.extend(["--resume", str(last)])
    return cmd


def eval_command(
    *,
    manifest_root: Path,
    checkpoint: Path,
    out_dir: Path,
) -> list[str]:
    paths = manifest_paths(manifest_root)
    return [
        sys.executable,
        "tools/evaluate_crnn_ctc.py",
        "--manifest",
        str(paths["test"]),
        "--vocab",
        str(paths["vocab"]),
        "--checkpoint",
        str(checkpoint),
        "--out_dir",
        str(out_dir),
        "--batch_size",
        "32",
        "--num_workers",
        "4",
        "--blank_logit_penalty",
        str(FIXED_TEST_PENALTY),
    ]


def variant_checkpoint_dir(out_root: Path, variant: str, seed: int) -> Path:
    cfg = VARIANTS[variant]
    if cfg["kind"] == "existing":
        return Path(cfg["checkpoint_dirs"][seed])
    return control_checkpoint_dir(out_root, variant, seed)


def run_variant_seed(
    *,
    out_root: Path,
    variant: str,
    seed: int,
    dry_run: bool,
    train: bool,
    evaluate: bool,
) -> dict[str, Any]:
    cfg = VARIANTS[variant]
    manifest_root = Path(cfg["manifest_root"])
    ckpt_dir = variant_checkpoint_dir(out_root, variant, seed)
    status: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "manifest_root": str(manifest_root),
        "checkpoint_dir": str(ckpt_dir),
        "kind": cfg["kind"],
        "train_returncode": None,
        "eval_returncode": None,
        "last_epoch": None,
        "best_exists": False,
        "eval_dir": str(eval_dir(out_root, variant, seed)),
    }

    if cfg["kind"] == "train" and train:
        last_epoch = read_checkpoint_epoch(ckpt_dir / "last.pt")
        status["last_epoch_before"] = last_epoch
        status["best_exists_before"] = (ckpt_dir / "best.pt").exists()
        if last_epoch is None or last_epoch < EPOCHS or not (ckpt_dir / "best.pt").exists():
            status["train_returncode"] = run(
                train_command(
                    manifest_root=manifest_root,
                    out_dir=ckpt_dir,
                    seed=seed,
                ),
                dry_run=dry_run,
            )
            if status["train_returncode"] != 0:
                return status
        else:
            status["train_returncode"] = 0
            status["train_skipped"] = "complete"

    status["last_epoch"] = read_checkpoint_epoch(ckpt_dir / "last.pt")
    best = ckpt_dir / "best.pt"
    status["best_exists"] = best.exists()

    if evaluate:
        if not best.exists():
            status["eval_skipped"] = "missing best.pt"
        else:
            summary = eval_dir(out_root, variant, seed) / "summary.json"
            if summary.exists():
                status["eval_returncode"] = 0
                status["eval_skipped"] = "summary exists"
            else:
                status["eval_returncode"] = run(
                    eval_command(
                        manifest_root=manifest_root,
                        checkpoint=best,
                        out_dir=eval_dir(out_root, variant, seed),
                    ),
                    dry_run=dry_run,
                )

    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="outputs/htr_publication_v3/full_same_size_controls")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    statuses = []
    for variant in args.variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        for seed in args.seeds:
            statuses.append(
                run_variant_seed(
                    out_root=out_root,
                    variant=variant,
                    seed=seed,
                    dry_run=args.dry_run,
                    train=not args.skip_training,
                    evaluate=not args.skip_eval,
                )
            )
            (out_root / "run_status.json").write_text(
                json.dumps(statuses, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(
        json.dumps({
            "out_root": str(out_root),
            "n_status": len(statuses),
            "statuses": statuses,
        }, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
