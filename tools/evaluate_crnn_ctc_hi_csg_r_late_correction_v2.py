from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.adapter_runtime import EpochWidthBatchSampler, summarize_rows, write_jsonl
from src.htr.adapter_runtime_v2 import evaluate_v2_loader
from src.htr.dataset_adapter_v2 import (
    HICSGRLateCorrectionDataset,
    collate_late_correction_batch,
)
from src.htr.model_hi_csg_r_late_correction_v2 import (
    backbone_state_sha256,
    load_frozen_late_correction_model,
)
from src.htr.vocab import CTCVocab
from src.htr.xaligned_hi_csg_r import XAlignedFeatureNormalizer


def group_summary(
    rows: list[dict[str, Any]],
    key_function: Any,
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_function(row))].append(row)
    return {
        key: summarize_rows(values)
        for key, values in sorted(grouped.items())
    }


def length_bucket(row: dict[str, Any]) -> str:
    length = int(row["target_chars"])
    if length <= 5:
        return "01_1-5"
    if length <= 10:
        return "02_6-10"
    if length <= 20:
        return "03_11-20"
    return "04_21+"


def token_type(row: dict[str, Any]) -> str:
    text = str(row["target"])
    if " " in text:
        return "multiword"
    if text.isdigit():
        return "digits"
    if text.isalpha():
        return "alphabetic"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=["correct", "shuffle", "zero"], required=True)
    parser.add_argument("--shuffle_map")
    parser.add_argument("--vocab")
    parser.add_argument("--normalizer")
    parser.add_argument("--risk_stats")
    parser.add_argument("--blank_logit_penalty", type=float)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--max_batches", type=int)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") or {}
    config = metadata.get("config") or checkpoint.get("config") or {}
    vocab_path = args.vocab or metadata.get("vocab") or config.get("vocab")
    normalizer_path = args.normalizer or metadata.get("normalizer") or config.get("normalizer")
    risk_path = args.risk_stats or metadata.get("risk_stats") or config.get("risk_stats")
    base_checkpoint = metadata.get("base_checkpoint") or config.get("base_checkpoint")
    if not all((vocab_path, normalizer_path, risk_path, base_checkpoint)):
        raise ValueError("Checkpoint metadata is missing vocab/normalizer/risk/base paths")
    if args.mode == "shuffle" and not args.shuffle_map:
        raise ValueError("shuffle mode requires --shuffle_map")
    vocab = CTCVocab.from_path(vocab_path)
    normalizer = XAlignedFeatureNormalizer.from_path(normalizer_path)
    risk_stats = json.loads(Path(risk_path).read_text(encoding="utf-8"))
    variant = str(metadata.get("variant") or config["variant"])
    alpha_max = float(metadata.get("alpha_max") or config["alpha_max"])
    model, base_metadata = load_frozen_late_correction_model(
        base_checkpoint,
        num_classes=vocab.num_classes,
        blank_index=vocab.blank_index,
        variant=variant,
        alpha_max=alpha_max,
        risk_q05=risk_stats["q05"],
        risk_q95=risk_stats["q95"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    expected_hash = metadata.get("backbone_state_sha256")
    actual_hash = backbone_state_sha256(model)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("Loaded backbone hash differs from checkpoint provenance")
    dataset = HICSGRLateCorrectionDataset(
        args.manifest,
        vocab,
        normalizer,
        shuffle_map=args.shuffle_map if args.mode == "shuffle" else None,
    )
    sampler = EpochWidthBatchSampler(
        dataset.rows,
        args.batch_size,
        seed=int(metadata.get("seed", config.get("seed", 42))),
        shuffle=False,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_late_correction_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)
    penalty = (
        float(args.blank_logit_penalty)
        if args.blank_logit_penalty is not None
        else float(metadata.get("blank_logit_penalty", config["blank_logit_penalty"]))
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, rows = evaluate_v2_loader(
        model,
        loader,
        vocab,
        device,
        criterion=criterion,
        blank_logit_penalty=penalty,
        preservation_temperature=float(
            config.get("preservation_temperature", 1.5)
        ),
        zero_graph=args.mode == "zero",
        max_batches=args.max_batches,
        frame_output=output / "frame_diagnostics.npz",
    )
    summary.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "base_checkpoint": base_metadata["base_checkpoint_path"],
            "manifest": str(Path(args.manifest).resolve()),
            "mode": args.mode,
            "variant": variant,
            "seed": int(metadata.get("seed", config.get("seed", 42))),
            "blank_logit_penalty": penalty,
            "shuffle_map": (
                str(Path(args.shuffle_map).resolve())
                if args.shuffle_map
                else None
            ),
            "backbone_state_sha256": actual_hash,
        }
    )
    write_jsonl(output / "predictions.jsonl", rows)
    write_jsonl(output / "per_sample_metrics.jsonl", rows)
    artifacts = {
        "summary.json": summary,
        "intervention_summary.json": {
            "alpha": summary["alpha"],
            "gate": summary["gate"],
            "uncertainty": summary["uncertainty"],
            "risk": summary["risk"],
            "correction": summary["correction"],
            "intervention": summary["intervention"],
        },
        "grouped_metrics.json": summary["grouped"],
        "domain_summary.json": group_summary(rows, lambda row: row["core_domain"]),
        "length_bucket_summary.json": group_summary(rows, length_bucket),
        "token_type_summary.json": group_summary(rows, token_type),
    }
    for name, value in artifacts.items():
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

