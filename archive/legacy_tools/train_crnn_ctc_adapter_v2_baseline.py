from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.adapter_runtime import (
    EpochWidthBatchSampler,
    apply_blank_logit_penalty,
    evaluate_loader,
    set_seed,
    summarize_rows,
    write_jsonl,
)
from src.htr.dataset_adapter import HICSGRAdapterDataset, collate_adapter_batch
from src.htr.model import CRNNCTC
from src.htr.vocab import CTCVocab


def scheduled_penalty(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    ratio = (epoch - 1) / (epochs - 1)
    return start + ratio * (end - start)


def domain_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    return {
        key: summarize_rows(values)
        for key, values in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--max_val_batches", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if config.get("mode") != "b0_dev":
        raise ValueError("Baseline config must use mode=b0_dev")
    set_seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vocab = CTCVocab.from_path(config["vocab"])
    train_dataset = HICSGRAdapterDataset(
        config["train_manifest"],
        vocab,
        mode="m0_ft",
    )
    dev_dataset = HICSGRAdapterDataset(
        config["val_manifest"],
        vocab,
        mode="m0_ft",
    )
    train_sampler = EpochWidthBatchSampler(
        train_dataset.rows,
        int(config["batch_size"]),
        seed=int(config["seed"]),
        shuffle=True,
    )
    dev_sampler = EpochWidthBatchSampler(
        dev_dataset.rows,
        int(config["batch_size"]),
        seed=int(config["seed"]),
        shuffle=False,
    )
    loader_options = {
        "collate_fn": collate_adapter_batch,
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_options,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_sampler=dev_sampler,
        **loader_options,
    )
    model = CRNNCTC(
        num_classes=vocab.num_classes,
        hidden_size=int(config["hidden_size"]),
        lstm_layers=int(config["lstm_layers"]),
        dropout=float(config["dropout"]),
        blank_index=vocab.blank_index,
        blank_bias_init=float(config.get("blank_bias_init", -1.0)),
        height_bins=int(config["height_bins"]),
        feature_size=int(config["feature_size"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)
    output = Path(config["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    existing = [
        output / name
        for name in ("best.pt", "last.pt", "history.jsonl", "train_summary.json")
        if (output / name).exists()
    ]
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(f"Baseline artifacts already exist: {existing}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.overwrite:
        for path in existing:
            path.unlink()
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "stderr.log").touch()
    runtime = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "python": sys.version,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "runtime_metadata.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history: list[dict[str, Any]] = []
    best_cer = float("inf")
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        saved_config = checkpoint.get("config") or {}
        for key in ("seed", "train_manifest", "val_manifest"):
            if str(saved_config.get(key)) != str(config.get(key)):
                raise ValueError(f"Resume config mismatch for {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_cer = float(checkpoint.get("best_cer", float("inf")))
        history_path = output / "history.jsonl"
        if history_path.exists():
            history = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if len(history) != start_epoch - 1:
            raise ValueError(
                "Resume history length does not match checkpoint epoch"
            )
    started = time.monotonic()
    epochs = int(config["epochs"])
    for epoch in range(start_epoch, epochs + 1):
        train_sampler.set_epoch(epoch)
        penalty = scheduled_penalty(
            epoch,
            epochs,
            float(config["blank_logit_penalty_start"]),
            float(config["blank_logit_penalty_end"]),
        )
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if args.max_train_batches and batch_index > args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            log_probs = model(batch["images"].to(device, non_blocking=True)).float()
            log_probs = apply_blank_logit_penalty(
                log_probs,
                vocab.blank_index,
                penalty,
            )
            loss = criterion(
                log_probs,
                batch["targets"].to(device, non_blocking=True),
                model.output_lengths(
                    batch["widths"].to(device, non_blocking=True)
                ),
                batch["target_lengths"].to(device, non_blocking=True),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite baseline loss at epoch={epoch}, batch={batch_index}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.item()))
        validation, rows = evaluate_loader(
            model,
            dev_loader,
            vocab,
            device,
            mode="m0_ft",
            criterion=criterion,
            blank_logit_penalty=penalty,
            max_batches=args.max_val_batches,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "blank_logit_penalty": penalty,
            "val_CER": validation["cer"],
            "val_macro_CER": validation["macro_cer"],
            "val_WER": validation["wer"],
            "val_exact": validation["exact"],
            "val_blank_ratio": validation["blank_ratio"],
        }
        history.append(record)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (output / "stdout.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_cer": min(best_cer, float(validation["cer"])),
            "val_cer": float(validation["cer"]),
            "blank_logit_penalty": penalty,
            "config": {
                **config,
                "blank_logit_penalty": penalty,
                "num_classes": vocab.num_classes,
            },
            "metadata": {
                "protocol": "crnn_ctc_hi_csg_r_late_correction_protocol_v2",
                "mode": "b0_dev",
                "seed": int(config["seed"]),
                "train_manifest": str(Path(config["train_manifest"]).resolve()),
                "val_manifest": str(Path(config["val_manifest"]).resolve()),
                "holdout_evaluated": False,
                "selection_metric": "dev_micro_CER",
            },
        }
        torch.save(checkpoint, output / "last.pt")
        if validation["cer"] < best_cer:
            best_cer = float(validation["cer"])
            torch.save(checkpoint, output / "best.pt")
            write_jsonl(output / "dev_predictions.jsonl", rows)
            (output / "dev_summary.json").write_text(
                json.dumps(validation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output / "dev_domain_summary.json").write_text(
                json.dumps(domain_summary(rows), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    summary = {
        "experiment_id": config["experiment_id"],
        "mode": "b0_dev",
        "seed": int(config["seed"]),
        "best_dev_micro_cer": best_cer,
        "epochs_completed": len(history),
        "runtime_seconds": time.monotonic() - started,
        "holdout_evaluated": False,
        **runtime,
    }
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
