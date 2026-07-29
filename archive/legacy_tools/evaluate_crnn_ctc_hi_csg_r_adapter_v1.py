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

from src.htr.adapter_runtime import (
    EpochWidthBatchSampler,
    evaluate_loader,
    summarize_rows,
    write_jsonl,
)
from src.htr.dataset_adapter import HICSGRAdapterDataset, collate_adapter_batch
from src.htr.model import CRNNCTC
from src.htr.model_hi_csg_r_adapter import CRNNCTCHICSGRAdapter, baseline_model_config
from src.htr.vocab import CTCVocab
from src.htr.xaligned_hi_csg_r import XAlignedFeatureNormalizer


def load_model(
    checkpoint_path: str | Path,
    vocab: CTCVocab,
) -> tuple[CRNNCTC | CRNNCTCHICSGRAdapter, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Adapter checkpoint has no metadata")
    mode = str(metadata["mode"])
    config = metadata.get("config") or {}
    base_config = config
    base_path = metadata.get("base_checkpoint")
    if base_path and Path(base_path).exists():
        base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
        base_config = base_checkpoint.get("config") or config
    model_config = baseline_model_config({"config": base_config})
    if mode == "m0_ft":
        model: CRNNCTC | CRNNCTCHICSGRAdapter = CRNNCTC(
            num_classes=vocab.num_classes,
            blank_index=vocab.blank_index,
            **model_config,
        )
    else:
        model = CRNNCTCHICSGRAdapter(
            num_classes=vocab.num_classes,
            blank_index=vocab.blank_index,
            gate_bias_init=float(config.get("gate_bias_init", -1.5)),
            **model_config,
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, metadata


def group_summary(
    rows: list[dict[str, Any]],
    key_function: Any,
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_function(row))].append(row)
    return {key: summarize_rows(values) for key, values in sorted(grouped.items())}


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
    parser.add_argument("--vocab")
    parser.add_argument("--normalizer")
    parser.add_argument("--shuffle_map")
    parser.add_argument("--zero_graph", action="store_true")
    parser.add_argument("--blank_logit_penalty", type=float)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = raw_checkpoint.get("metadata") or {}
    vocab_path = args.vocab or metadata.get("vocab")
    if not vocab_path:
        raise ValueError("Vocab path is required")
    vocab = CTCVocab.from_path(vocab_path)
    model, metadata = load_model(args.checkpoint, vocab)
    mode = str(metadata["mode"])
    expected_penalty = float(metadata["blank_logit_penalty"])
    penalty = expected_penalty if args.blank_logit_penalty is None else args.blank_logit_penalty
    if abs(float(penalty) - expected_penalty) > 1e-12:
        raise ValueError(
            f"Evaluation penalty {penalty} differs from frozen checkpoint value "
            f"{expected_penalty}"
        )
    normalizer = None
    if mode != "m0_ft":
        normalizer_path = args.normalizer or metadata.get("normalizer")
        if not normalizer_path:
            raise ValueError("Adapter evaluation requires a normalizer")
        normalizer = XAlignedFeatureNormalizer.from_path(normalizer_path)
    dataset = HICSGRAdapterDataset(
        args.manifest,
        vocab,
        normalizer=normalizer,
        mode=mode,
        shuffle_map=args.shuffle_map,
    )
    sampler = EpochWidthBatchSampler(
        dataset.rows, args.batch_size, seed=int(metadata["seed"]), shuffle=False
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_adapter_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)
    summary, rows = evaluate_loader(
        model,
        loader,
        vocab,
        device,
        mode=mode,
        criterion=criterion,
        blank_logit_penalty=float(penalty),
        graph_enabled=not args.zero_graph,
    )
    summary.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "manifest": str(Path(args.manifest).resolve()),
            "mode": mode,
            "seed": metadata["seed"],
            "blank_logit_penalty": penalty,
            "shuffle_map": (
                str(Path(args.shuffle_map).resolve()) if args.shuffle_map else None
            ),
            "zero_graph": bool(args.zero_graph),
        }
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", rows)
    write_jsonl(output / "per_sample_metrics.jsonl", rows)
    artifacts = {
        "summary.json": summary,
        "domain_summary.json": group_summary(rows, lambda row: row["dataset"]),
        "length_bucket_summary.json": group_summary(rows, length_bucket),
        "token_type_summary.json": group_summary(rows, token_type),
    }
    for name, data in artifacts.items():
        (output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
