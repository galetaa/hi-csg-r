from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

CANONICAL_BASES = {
    42: "outputs/htr_graph_v1/"
    "tri10k_image_only_plus_school_lines_10k_context_v1/best.pt",
    43: "outputs/htr_graph_v1/"
    "tri10k_image_only_plus_school_lines_10k_context_v1_seed43/best.pt",
    44: "outputs/htr_graph_v1/"
    "tri10k_image_only_plus_school_lines_10k_context_v1_seed44/best.pt",
}
MATCHED_BASES = {
    42: "outputs/htr_adapter_v1/m0_ft_seed42/best.pt",
    43: "outputs/htr_adapter_v2/m0_ft_final_seed43/best.pt",
    44: "outputs/htr_adapter_v2/m0_ft_final_seed44/best.pt",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def matched_config(seed: int) -> dict[str, Any]:
    return {
        "experiment_id": f"htr_adapter_v2_m0_ft_final_seed{seed}",
        "protocol": "crnn_ctc_hi_csg_r_late_correction_protocol_v2",
        "stage": "final",
        "mode": "m0_ft",
        "seed": seed,
        "base_checkpoint": CANONICAL_BASES[seed],
        "train_manifest": "data/experiments/htr_adapter_v1/manifests/train.jsonl",
        "val_manifest": "data/experiments/htr_adapter_v1/manifests/val.jsonl",
        "vocab": (
            "data/experiments/htr_baseline_v1_ctc_ready/"
            "tri10k_mixed_plus_school_lines_10k_context_v1/vocab.json"
        ),
        "normalizer": None,
        "warmup_epochs": 0,
        "joint_epochs": 25,
        "blank_logit_penalty": -0.4,
        "batch_size": 16,
        "num_workers": 4,
        "weight_decay": 0.0001,
        "grad_clip": 5.0,
        "lr_rnn": 0.00005,
        "lr_classifier": 0.00005,
        "lr_last_cnn": 0.00001,
        "out_dir": f"outputs/htr_adapter_v2/m0_ft_final_seed{seed}",
    }


def final_config(
    seed: int,
    selected: dict[str, Any],
    selection_path: Path,
    holdout_path: Path,
) -> dict[str, Any]:
    checkpoint = Path(selected["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") or {}
    selected_epoch = max(int(metadata.get("epoch", payload.get("epoch", 8))), 8)
    return {
        "experiment_id": f"htr_adapter_v2_final_seed{seed}",
        "protocol": "crnn_ctc_hi_csg_r_late_correction_protocol_v2",
        "stage": "final",
        "variant": selected["variant"],
        "seed": seed,
        "base_checkpoint": MATCHED_BASES[seed],
        "train_manifest": "data/experiments/htr_adapter_v1/manifests/train.jsonl",
        "val_manifest": "data/experiments/htr_adapter_v1/manifests/val.jsonl",
        "vocab": (
            "data/experiments/htr_baseline_v1_ctc_ready/"
            "tri10k_mixed_plus_school_lines_10k_context_v1/vocab.json"
        ),
        "normalizer": (
            "data/experiments/htr_adapter_v1/normalizer/train_stats.json"
        ),
        "risk_stats": (
            "data/experiments/htr_adapter_v2/normalizer/risk_stats.json"
        ),
        "out_dir": f"outputs/htr_adapter_v2/final_seed{seed}",
        "blank_logit_penalty": -0.4,
        "alpha_max": 0.25,
        "alpha_logit_init": -6.0,
        "lambda_preservation": selected["lambda_preservation"],
        "lambda_alignment": 0.0,
        "preservation_temperature": 1.5,
        "max_epochs": selected_epoch,
        "min_epochs": selected_epoch,
        "early_stopping_patience": selected_epoch + 1,
        "batch_size": 16,
        "num_workers": 4,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "grad_clip": 5.0,
        "domain_balanced": True,
        "selection_artifact": str(selection_path.resolve()),
        "selection_artifact_sha256": sha256(selection_path),
        "holdout_decision": str(holdout_path.resolve()),
        "holdout_decision_sha256": sha256(holdout_path),
        "selected_development_checkpoint": str(checkpoint.resolve()),
        "selected_development_checkpoint_sha256": sha256(checkpoint),
        "final_epoch_budget_source": "selected development checkpoint epoch",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="outputs/htr_adapter_v2/development/selected_candidate.json",
    )
    parser.add_argument(
        "--holdout_decision",
        default="outputs/htr_adapter_v2/holdout/decision/holdout_decision.json",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/htr_adapter_v2/frozen_final_configs",
    )
    args = parser.parse_args()
    selection_path = Path(args.selection)
    holdout_path = Path(args.holdout_decision)
    selection = read_json(selection_path)
    holdout = read_json(holdout_path)
    if selection.get("status") != "PASS" or not selection.get("selected"):
        raise ValueError("Final config resolution requires a passing dev selection")
    if holdout.get("status") != "PASS":
        raise ValueError("Final config resolution requires a positive holdout gate")
    selected = selection["selected"]
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44):
        m0 = matched_config(seed)
        final = final_config(seed, selected, selection_path, holdout_path)
        for name, value in (
            (f"m0_ft_final_seed{seed}.yaml", m0),
            (f"final_seed{seed}.yaml", final),
        ):
            (output / name).write_text(
                yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    matched = {
        "protocol": "crnn_ctc_hi_csg_r_late_correction_protocol_v2",
        "created_at": datetime.now(UTC).isoformat(),
        "seed42_reused_existing_v1_m0_ft": True,
        "checkpoints": {
            str(seed): {
                "path": path,
                "exists": Path(path).exists(),
                "sha256": sha256(path) if Path(path).exists() else None,
            }
            for seed, path in MATCHED_BASES.items()
        },
    }
    (output / "matched_baselines.json").write_text(
        json.dumps(matched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(matched, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
