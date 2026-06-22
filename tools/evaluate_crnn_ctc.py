from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.htr.ctc_decode import greedy_decode
from src.htr.dataset import HTRDataset, collate_htr_batch
from src.htr.metrics import cer, exact_match, wer
from src.htr.model import CRNNCTC
from src.htr.vocab import CTCVocab


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def apply_blank_logit_penalty(
    log_probs: torch.Tensor,
    blank_index: int,
    penalty: float,
) -> torch.Tensor:
    if penalty == 0.0:
        return log_probs

    adjusted = log_probs.clone()
    adjusted[..., blank_index] += float(penalty)
    adjusted = adjusted - torch.logsumexp(adjusted, dim=-1, keepdim=True)
    return adjusted


def load_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    vocab: CTCVocab,
    device: torch.device,
) -> tuple[CRNNCTC, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})

    model = CRNNCTC(
        num_classes=vocab.num_classes,
        hidden_size=int(config.get("hidden_size", 256)),
        lstm_layers=int(config.get("lstm_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        blank_index=vocab.blank_index,
        blank_bias_init=float(config.get("blank_bias_init", -1.0)),
        height_bins=int(config.get("height_bins", 4)),
        feature_size=int(config.get("feature_size", 256)),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, {
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_cer": ckpt.get("val_cer"),
        "checkpoint_blank_logit_penalty": ckpt.get("blank_logit_penalty"),
        "checkpoint_config": config,
    }


def evaluate(
    *,
    model: CRNNCTC,
    loader: DataLoader,
    vocab: CTCVocab,
    device: torch.device,
    blank_logit_penalty: float,
    max_batches: int | None,
) -> dict[str, Any]:
    cers = []
    wers = []
    exacts = []
    pred_lens = []
    empty_preds = []
    blank_ratios = []

    grouped = defaultdict(lambda: {"cer": [], "wer": [], "exact": []})
    examples = []
    predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            images = batch["images"].to(device)
            widths = batch["widths"].to(device)

            log_probs = model(images)
            log_probs = apply_blank_logit_penalty(
                log_probs,
                blank_index=vocab.blank_index,
                penalty=blank_logit_penalty,
            )

            input_lengths = model.output_lengths(widths).to(device)
            preds = greedy_decode(log_probs, input_lengths, vocab)

            argmax_ids = log_probs.argmax(dim=-1).detach().cpu()
            input_lengths_cpu = input_lengths.detach().cpu().tolist()

            blank_total = 0
            token_total = 0

            for b, length in enumerate(input_lengths_cpu):
                ids = argmax_ids[: int(length), b]
                blank_total += int((ids == vocab.blank_index).sum().item())
                token_total += int(length)

            if token_total > 0:
                blank_ratios.append(blank_total / token_total)

            for pred, target, dataset, level, category, sample_id in zip(
                preds,
                batch["texts"],
                batch["datasets"],
                batch["levels"],
                batch["categories"],
                batch["sample_ids"],
            ):
                c = cer(pred, target)
                w = wer(pred, target)
                e = exact_match(pred, target)

                cers.append(c)
                wers.append(w)
                exacts.append(e)
                pred_lens.append(len(pred))
                empty_preds.append(1.0 if pred == "" else 0.0)

                key = f"{dataset}|{level}|{category}"
                grouped[key]["cer"].append(c)
                grouped[key]["wer"].append(w)
                grouped[key]["exact"].append(e)

                item = {
                    "sample_id": sample_id,
                    "target": target,
                    "pred": pred,
                    "cer": c,
                    "wer": w,
                    "exact": e,
                    "level": level,
                    "category": category,
                }

                predictions.append(item)

                if len(examples) < 50:
                    examples.append(item)

    grouped_out = {
        key: {
            "n": len(vals["cer"]),
            "cer": mean(vals["cer"]),
            "wer": mean(vals["wer"]),
            "exact": mean(vals["exact"]),
        }
        for key, vals in grouped.items()
    }

    return {
        "n": len(cers),
        "cer": mean(cers),
        "wer": mean(wers),
        "exact": mean(exacts),
        "pred_len_mean": mean(pred_lens),
        "pred_empty_ratio": mean(empty_preds),
        "argmax_blank_ratio": mean(blank_ratios),
        "grouped": grouped_out,
        "examples": examples,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--blank_logit_penalty", type=float, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    vocab = CTCVocab.from_path(args.vocab)
    model, ckpt_info = load_model_from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        vocab=vocab,
        device=device,
    )

    ds = HTRDataset(args.manifest, vocab)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_htr_batch,
        pin_memory=torch.cuda.is_available(),
    )

    blank_penalty = (
        args.blank_logit_penalty
        if args.blank_logit_penalty is not None
        else float(ckpt_info.get("checkpoint_blank_logit_penalty") or 0.0)
    )

    metrics = evaluate(
        model=model,
        loader=loader,
        vocab=vocab,
        device=device,
        blank_logit_penalty=blank_penalty,
        max_batches=args.max_batches,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "blank_logit_penalty": blank_penalty,
        **ckpt_info,
        "metrics": {
            k: v
            for k, v in metrics.items()
            if k not in {"predictions"}
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in metrics["predictions"]) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "n": metrics["n"],
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "exact": metrics["exact"],
        "pred_len_mean": metrics["pred_len_mean"],
        "pred_empty_ratio": metrics["pred_empty_ratio"],
        "argmax_blank_ratio": metrics["argmax_blank_ratio"],
        "out_dir": str(out_dir),
        "blank_logit_penalty": blank_penalty,
        "checkpoint_epoch": ckpt_info["checkpoint_epoch"],
        "checkpoint_val_cer": ckpt_info["checkpoint_val_cer"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
