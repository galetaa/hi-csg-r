from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.htr.ctc_decode import greedy_decode
from src.htr.dataset import HTRDataset, collate_htr_batch
from src.htr.metrics import cer, exact_match, wer
from src.htr.model import CRNNCTC
from src.htr.vocab import CTCVocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def apply_blank_logit_penalty(
    log_probs: torch.Tensor,
    blank_index: int,
    penalty: float,
) -> torch.Tensor:
    """
    Applies a fixed blank log-probability penalty and renormalizes.

    penalty < 0 discourages blank.
    This is a debug/training-stabilization tool for CTC blank collapse.
    """
    if penalty == 0.0:
        return log_probs

    adjusted = log_probs.clone()
    adjusted[..., blank_index] += float(penalty)
    adjusted = adjusted - torch.logsumexp(adjusted, dim=-1, keepdim=True)
    return adjusted

def scheduled_blank_penalty(
    *,
    epoch: int,
    total_epochs: int,
    start: float,
    end: float,
) -> float:
    if total_epochs <= 1:
        return end

    alpha = (epoch - 1) / max(total_epochs - 1, 1)
    return start + alpha * (end - start)

def evaluate(
    *,
    model: CRNNCTC,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    vocab: CTCVocab,
    device: torch.device,
    blank_logit_penalty: float = 0.0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()

    losses = []
    cers = []
    wers = []
    exacts = []
    pred_lens = []
    empty_preds = []
    argmax_blank_ratios = []

    grouped = defaultdict(lambda: {"cer": [], "wer": [], "exact": []})
    examples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            widths = batch["widths"].to(device)

            log_probs = model(images)
            log_probs = apply_blank_logit_penalty(
                log_probs,
                blank_index=vocab.blank_index,
                penalty=blank_logit_penalty,
            )
            input_lengths = model.output_lengths(widths).to(device)

            log_probs = apply_blank_logit_penalty(
                log_probs,
                blank_index=vocab.blank_index,
                penalty=args.blank_logit_penalty if "args" in globals() else 0.0,  # pyright: ignore [reportUndefinedVariable]
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            losses.append(float(loss.item()))

            argmax_ids = log_probs.argmax(dim=-1).detach().cpu()  # [T, B]
            input_lengths_cpu = input_lengths.detach().cpu().tolist()

            blank_total = 0
            token_total = 0

            for b, length in enumerate(input_lengths_cpu):
                ids = argmax_ids[: int(length), b]
                blank_total += int((ids == vocab.blank_index).sum().item())
                token_total += int(length)

            preds = greedy_decode(log_probs, input_lengths, vocab)

            for pred, target, dataset, level, category, sample_id in zip(
                preds,
                batch["texts"],
                batch["datasets"],
                batch["levels"],
                batch["categories"],
                batch["sample_ids"],
            ):
                pred_lens.append(len(pred))
                empty_preds.append(1.0 if pred == "" else 0.0)

                c = cer(pred, target)
                w = wer(pred, target)
                e = exact_match(pred, target)

                cers.append(c)
                wers.append(w)
                exacts.append(e)

                key = f"{dataset}|{level}|{category}"
                grouped[key]["cer"].append(c)
                grouped[key]["wer"].append(w)
                grouped[key]["exact"].append(e)

                if len(examples) < 30:
                    examples.append(
                        {
                            "sample_id": sample_id,
                            "target": target,
                            "pred": pred,
                            "cer": c,
                            "wer": w,
                        }
                    )
            if token_total > 0:
                argmax_blank_ratios.append(blank_total / token_total)
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
        "loss": mean(losses),
        "cer": mean(cers),
        "wer": mean(wers),
        "exact": mean(exacts),
        "grouped": grouped_out,
        "examples": examples,
        "pred_len_mean": mean(pred_lens),
        "pred_empty_ratio": mean(empty_preds),
        "argmax_blank_ratio": mean(argmax_blank_ratios),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    vocab = CTCVocab.from_path(args.vocab)

    train_ds = HTRDataset(args.train_manifest, vocab)
    val_ds = HTRDataset(args.val_manifest, vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_htr_batch,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_htr_batch,
        pin_memory=torch.cuda.is_available(),
    )

    model = CRNNCTC(
        num_classes=vocab.num_classes,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        blank_index=vocab.blank_index,
        blank_bias_init=args.blank_bias_init,
        height_bins=args.height_bins,
        feature_size=args.feature_size,
    ).to(device)

    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["device_resolved"] = str(device)
    config["num_classes"] = vocab.num_classes
    config["train_size"] = len(train_ds)
    config["val_size"] = len(val_ds)

    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history = []
    best_cer = float("inf")

    print(json.dumps(config, ensure_ascii=False, indent=2))

    for epoch in range(1, args.epochs + 1):
        if args.blank_logit_penalty is not None:
            current_blank_penalty = args.blank_logit_penalty
        else:
            current_blank_penalty = scheduled_blank_penalty(
                epoch=epoch,
                total_epochs=args.epochs,
                start=args.blank_logit_penalty_start,
                end=args.blank_logit_penalty_end,
            )
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            widths = batch["widths"].to(device)

            log_probs = model(images)
            log_probs = apply_blank_logit_penalty(
                log_probs,
                blank_index=vocab.blank_index,
                penalty=current_blank_penalty,
            )
            input_lengths = model.output_lengths(widths).to(device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_losses.append(float(loss.item()))

            if batch_idx % args.log_every == 0:
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                    f"train_loss={mean(train_losses[-args.log_every:]):.4f}"
                )

            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            vocab=vocab,
            device=device,
            max_batches=args.max_val_batches,
            blank_logit_penalty=current_blank_penalty,
        )

        row = {
            "epoch": epoch,
            "train_loss": mean(train_losses),
            "val": val_metrics,
            "blank_logit_penalty": current_blank_penalty,
        }
        history.append(row)

        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_cer={val_metrics['cer']:.4f} "
            f"val_wer={val_metrics['wer']:.4f} "
            f"val_exact={val_metrics['exact']:.4f} "
            f"pred_len={val_metrics['pred_len_mean']:.2f} "
            f"empty={val_metrics['pred_empty_ratio']:.3f} "
            f"blank={val_metrics['argmax_blank_ratio']:.3f}"
            f"blank_penalty={current_blank_penalty:.3f} "
        )

        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_cer": val_metrics["cer"],
                "blank_logit_penalty": current_blank_penalty,
                "config": config,
            },
            out_dir / "last.pt",
        )

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_cer": val_metrics["cer"],
                    "blank_logit_penalty": current_blank_penalty,
                    "config": config,
                },
                out_dir / "best.pt",
            )
            (out_dir / "best_val_examples.json").write_text(
                json.dumps(val_metrics["examples"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print("best_val_cer:", best_cer)
    print("wrote:", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)

    parser.add_argument("--blank_bias_init", type=float, default=-1.0)

    parser.add_argument("--height_bins", type=int, default=4)
    parser.add_argument("--feature_size", type=int, default=256)

    parser.add_argument("--blank_logit_penalty", type=float, default=None)
    parser.add_argument("--blank_logit_penalty_start", type=float, default=0.0)
    parser.add_argument("--blank_logit_penalty_end", type=float, default=0.0)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()