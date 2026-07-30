from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
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
    evaluate_loader,
    forward_batch,
    set_seed,
    summarize_rows,
    write_jsonl,
)
from src.htr.dataset_adapter import HICSGRAdapterDataset, collate_adapter_batch
from src.htr.model import CRNNCTC
from src.htr.model_hi_csg_r_adapter import (
    CRNNCTCHICSGRAdapter,
    baseline_model_config,
    configure_image_model_joint_finetuning,
    load_canonical_image_model,
    load_canonical_visual_weights,
    total_parameter_count,
    trainable_parameter_count,
)
from src.htr.vocab import CTCVocab
from src.htr.xaligned_hi_csg_r import (
    FEATURE_VERSION,
    XAlignedFeatureNormalizer,
    verify_normalizer_for_manifest,
)

DEFAULTS: dict[str, Any] = {
    "mode": "m3_full",
    "seed": 42,
    "warmup_epochs": 5,
    "joint_epochs": 25,
    "blank_logit_penalty": -0.4,
    "batch_size": 16,
    "num_workers": 4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "lr_graph": 3e-4,
    "lr_gate": 3e-4,
    "lr_aux": 3e-4,
    "lr_rnn": 5e-5,
    "lr_classifier": 5e-5,
    "lr_last_cnn": 1e-5,
    "aux_loss_weight": 0.15,
    "gate_bias_init": -1.5,
}


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    for key, value in vars(args).items():
        if key not in {"command", "config", "func"} and value is not None:
            config[key] = value
    merged = {**DEFAULTS, **config}
    required = ("base_checkpoint", "train_manifest", "val_manifest", "vocab", "out_dir")
    missing = [key for key in required if not merged.get(key)]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    if merged["mode"] not in {"m0_ft", "m2_geometry", "m3_full"}:
        raise ValueError(f"Unsupported mode: {merged['mode']}")
    if merged["mode"] != "m0_ft" and not merged.get("normalizer"):
        raise ValueError("Adapter modes require --normalizer")
    if merged["mode"] == "m0_ft":
        merged["warmup_epochs"] = 0
    merged.setdefault(
        "experiment_id",
        f"htr_adapter_v1_{merged['mode']}_seed{merged['seed']}",
    )
    return merged


def make_model(
    config: dict[str, Any],
    vocab: CTCVocab,
) -> tuple[CRNNCTC | CRNNCTCHICSGRAdapter, dict[str, Any]]:
    if config["mode"] == "m0_ft":
        model, base_metadata = load_canonical_image_model(
            config["base_checkpoint"],
            num_classes=vocab.num_classes,
            blank_index=vocab.blank_index,
        )
        configure_image_model_joint_finetuning(model)
        return model, base_metadata

    base = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=False)
    model = CRNNCTCHICSGRAdapter(
        num_classes=vocab.num_classes,
        blank_index=vocab.blank_index,
        gate_bias_init=float(config["gate_bias_init"]),
        **baseline_model_config(base),
    )
    base_metadata = load_canonical_visual_weights(model, config["base_checkpoint"])
    if model.adapter_parameter_count() > 400_000:
        raise ValueError(
            f"Adapter parameter budget exceeded: {model.adapter_parameter_count()} > 400000"
        )
    return model, base_metadata


def optimizer_for_stage(
    model: CRNNCTC | CRNNCTCHICSGRAdapter,
    config: dict[str, Any],
    stage: str,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    if isinstance(model, CRNNCTCHICSGRAdapter):
        if stage == "warmup":
            model.configure_warmup()
            groups = [
                {"params": model.graph_adapter.parameters(), "lr": config["lr_graph"]},
                {"params": model.graph_aux_classifier.parameters(), "lr": config["lr_aux"]},
            ]
        else:
            model.configure_joint_finetuning()
            groups = [
                {
                    "params": [
                        *model.graph_adapter.parameters(),
                        *model.fusion_norm.parameters(),
                    ],
                    "lr": config["lr_graph"],
                },
                {"params": model.graph_gate.parameters(), "lr": config["lr_gate"]},
                {"params": model.graph_aux_classifier.parameters(), "lr": config["lr_aux"]},
                {"params": model.rnn.parameters(), "lr": config["lr_rnn"]},
                {"params": model.classifier.parameters(), "lr": config["lr_classifier"]},
                {
                    "params": [
                        parameter
                        for index, layer in enumerate(model.cnn)
                        if index >= 8
                        for parameter in layer.parameters()
                    ],
                    "lr": config["lr_last_cnn"],
                },
            ]
    else:
        configure_image_model_joint_finetuning(model)
        groups = [
            {"params": model.rnn.parameters(), "lr": config["lr_rnn"]},
            {"params": model.classifier.parameters(), "lr": config["lr_classifier"]},
            {
                "params": [
                    parameter
                    for index, layer in enumerate(model.cnn)
                    if index >= 8
                    for parameter in layer.parameters()
                ],
                "lr": config["lr_last_cnn"],
            },
        ]
    groups = [
        {**group, "params": [parameter for parameter in group["params"] if parameter.requires_grad]}
        for group in groups
    ]
    return torch.optim.AdamW(
        groups,
        weight_decay=float(config["weight_decay"]),
    )


def grad_norm(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().item())
    return squared**0.5


def train_epoch(
    model: CRNNCTC | CRNNCTCHICSGRAdapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    config: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    model.train()
    fused_losses: list[float] = []
    aux_losses: list[float] = []
    total_losses: list[float] = []
    adapter_gradients: list[float] = []
    gate_gradients: list[float] = []
    maximum = config.get("max_train_batches")

    for batch_index, batch in enumerate(loader, start=1):
        if maximum and batch_index > int(maximum):
            break
        optimizer.zero_grad(set_to_none=True)
        output = forward_batch(
            model,
            batch,
            device,
            mode=config["mode"],
            blank_logit_penalty=float(config["blank_logit_penalty"]),
        )
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        lengths = output["output_lengths"]
        fused_loss = criterion(output["log_probs"], targets, lengths, target_lengths)
        if "graph_aux_log_probs" in output:
            aux_loss = criterion(
                output["graph_aux_log_probs"],
                targets,
                lengths,
                target_lengths,
            )
        else:
            aux_loss = torch.zeros((), device=device)
        loss = (
            aux_loss
            if stage == "warmup"
            else fused_loss + float(config["aux_loss_weight"]) * aux_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss at batch {batch_index}")
        loss.backward()
        adapter = model.graph_adapter if isinstance(model, CRNNCTCHICSGRAdapter) else None
        gate = model.graph_gate if isinstance(model, CRNNCTCHICSGRAdapter) else None
        adapter_gradients.append(grad_norm(adapter))
        gate_gradients.append(grad_norm(gate))
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["grad_clip"]),
        )
        optimizer.step()
        fused_losses.append(float(fused_loss.item()))
        aux_losses.append(float(aux_loss.item()))
        total_losses.append(float(loss.item()))

    return {
        "train_total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "train_fused_ctc_loss": float(np.mean(fused_losses)) if fused_losses else 0.0,
        "train_graph_aux_ctc_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
        "graph_adapter_grad_norm": (
            float(np.mean(adapter_gradients)) if adapter_gradients else 0.0
        ),
        "gate_grad_norm": float(np.mean(gate_gradients)) if gate_gradients else 0.0,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    base_metadata: dict[str, Any],
    *,
    stage: str,
    epoch: int,
    best_val_cer: float,
    validation: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "experiment_id": config["experiment_id"],
        "mode": config["mode"],
        "seed": int(config["seed"]),
        "base_checkpoint": base_metadata["base_checkpoint_path"],
        "base_checkpoint_sha256": base_metadata["base_checkpoint_sha256"],
        "train_manifest": str(Path(config["train_manifest"]).resolve()),
        "val_manifest": str(Path(config["val_manifest"]).resolve()),
        "vocab": str(Path(config["vocab"]).resolve()),
        "normalizer": (
            str(Path(config["normalizer"]).resolve()) if config.get("normalizer") else None
        ),
        "normalizer_train_manifest": (
            str(Path(config["normalizer_train_manifest"]).resolve())
            if config.get("normalizer_train_manifest")
            else None
        ),
        "feature_version": FEATURE_VERSION,
        "feature_dim": 20,
        "topology_enabled": config["mode"] == "m3_full",
        "blank_logit_penalty": float(config["blank_logit_penalty"]),
        "stage": stage,
        "epoch": epoch,
        "best_val_cer": best_val_cer,
        "config": config,
        "validation": validation,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": int(config["seed"]),
        "mode": config["mode"],
        "best_val_cer": best_val_cer,
        "config": config,
        "metadata": metadata,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def domain_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    return {key: summarize_rows(values) for key, values in sorted(grouped.items())}


def run_train(args: argparse.Namespace) -> None:
    config = load_config(args)
    set_seed(int(config["seed"]))
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = CTCVocab.from_path(config["vocab"])
    normalizer = None
    if config["mode"] != "m0_ft":
        normalizer = XAlignedFeatureNormalizer.from_path(config["normalizer"])
        verify_normalizer_for_manifest(
            normalizer,
            config["train_manifest"],
            normalizer_train_manifest=config.get("normalizer_train_manifest"),
        )

    train_dataset = HICSGRAdapterDataset(
        config["train_manifest"], vocab, normalizer=normalizer, mode=config["mode"]
    )
    val_dataset = HICSGRAdapterDataset(
        config["val_manifest"], vocab, normalizer=normalizer, mode=config["mode"]
    )
    train_sampler = EpochWidthBatchSampler(
        train_dataset.rows,
        int(config["batch_size"]),
        seed=int(config["seed"]),
        shuffle=True,
    )
    val_sampler = EpochWidthBatchSampler(
        val_dataset.rows,
        int(config["batch_size"]),
        seed=int(config["seed"]),
        shuffle=False,
    )
    loader_kwargs = {
        "num_workers": int(config["num_workers"]),
        "collate_fn": collate_adapter_batch,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_kwargs)
    model, base_metadata = make_model(config, vocab)
    if base_metadata.get("base_checkpoint_seed") not in {None, int(config["seed"])}:
        raise ValueError(
            f"Base checkpoint seed {base_metadata['base_checkpoint_seed']} "
            f"does not match run seed {config['seed']}"
        )
    model.to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)
    output_dir = Path(config["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        output_dir / name
        for name in (
            "best.pt",
            "last.pt",
            "history.jsonl",
            "history.json",
            "train_summary.json",
            "stdout.log",
            "stderr.log",
        )
    ]
    existing = [path for path in generated if path.exists()]
    if existing and not config.get("overwrite"):
        raise FileExistsError(
            "Run directory already contains training artifacts; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if config.get("overwrite"):
        for path in existing:
            path.unlink()
    (output_dir / "stderr.log").touch()
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runtime = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "python": sys.version,
        "total_parameters": total_parameter_count(model),
        "adapter_parameters": (
            model.adapter_parameter_count()
            if isinstance(model, CRNNCTCHICSGRAdapter)
            else 0
        ),
        "base_parameters": total_parameter_count(model)
        - (
            model.adapter_parameter_count()
            if isinstance(model, CRNNCTCHICSGRAdapter)
            else 0
        ),
    }
    runtime["relative_parameter_increase_percent"] = (
        100.0 * runtime["adapter_parameters"] / max(runtime["base_parameters"], 1)
    )
    (output_dir / "runtime_metadata.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    best_val_cer = float("inf")
    global_epoch = 0
    started = time.monotonic()
    stages = []
    if int(config["warmup_epochs"]) > 0:
        stages.append(("warmup", int(config["warmup_epochs"])))
    stages.append(("joint", int(config["joint_epochs"])))

    for stage, epochs in stages:
        optimizer = optimizer_for_stage(model, config, stage)
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_sampler.set_epoch(stage_epoch if stage == "joint" else -stage_epoch)
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, config, stage
            )
            val_summary, val_rows = evaluate_loader(
                model,
                val_loader,
                vocab,
                device,
                mode=config["mode"],
                criterion=criterion,
                blank_logit_penalty=float(config["blank_logit_penalty"]),
                max_batches=config.get("max_val_batches"),
            )
            record = {
                "stage": stage,
                "stage_epoch": stage_epoch,
                "global_epoch": global_epoch,
                **train_metrics,
                "val_CER": val_summary["cer"],
                "val_WER": val_summary["wer"],
                "val_exact": val_summary["exact"],
                "val_macro_CER": val_summary["macro_cer"],
                "val_graph_aux_CER": val_summary["graph_aux_cer"],
                "blank_ratio": val_summary["blank_ratio"],
                "pred_len_mean": val_summary["pred_len_mean"],
                "gate": val_summary.get("gate"),
                "trainable_parameters": trainable_parameter_count(model),
            }
            history.append(record)
            with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            with (output_dir / "stdout.log").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)

            selected = stage == "joint" and val_summary["cer"] < best_val_cer
            if selected:
                best_val_cer = float(val_summary["cer"])
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    config,
                    base_metadata,
                    stage=stage,
                    epoch=stage_epoch,
                    best_val_cer=best_val_cer,
                    validation=val_summary,
                )
                save_checkpoint(output_dir / "best.pt", payload)
                write_jsonl(output_dir / "val_predictions.jsonl", val_rows)
                (output_dir / "val_summary.json").write_text(
                    json.dumps(val_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (output_dir / "val_domain_summary.json").write_text(
                    json.dumps(domain_summaries(val_rows), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            payload = checkpoint_payload(
                model,
                optimizer,
                config,
                base_metadata,
                stage=stage,
                epoch=stage_epoch,
                best_val_cer=best_val_cer,
                validation=val_summary,
            )
            save_checkpoint(output_dir / "last.pt", payload)

    summary = {
        "experiment_id": config["experiment_id"],
        "mode": config["mode"],
        "seed": config["seed"],
        "best_val_cer": best_val_cer,
        "epochs_completed": global_epoch,
        "runtime_seconds": time.monotonic() - started,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        **runtime,
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config")
    train.add_argument("--experiment_id")
    train.add_argument("--mode", choices=["m0_ft", "m2_geometry", "m3_full"])
    train.add_argument("--base_checkpoint")
    train.add_argument("--train_manifest")
    train.add_argument("--val_manifest")
    train.add_argument("--vocab")
    train.add_argument("--normalizer")
    train.add_argument("--normalizer_train_manifest")
    train.add_argument("--seed", type=int)
    train.add_argument("--warmup_epochs", type=int)
    train.add_argument("--joint_epochs", type=int)
    train.add_argument("--blank_logit_penalty", type=float)
    train.add_argument("--batch_size", type=int)
    train.add_argument("--num_workers", type=int)
    train.add_argument("--weight_decay", type=float)
    train.add_argument("--grad_clip", type=float)
    train.add_argument("--lr_graph", type=float)
    train.add_argument("--lr_gate", type=float)
    train.add_argument("--lr_aux", type=float)
    train.add_argument("--lr_rnn", type=float)
    train.add_argument("--lr_classifier", type=float)
    train.add_argument("--lr_last_cnn", type=float)
    train.add_argument("--aux_loss_weight", type=float)
    train.add_argument("--gate_bias_init", type=float)
    train.add_argument("--out_dir")
    train.add_argument("--device")
    train.add_argument("--max_train_batches", type=int)
    train.add_argument("--max_val_batches", type=int)
    train.add_argument("--overwrite", action="store_true", default=None)
    train.set_defaults(func=run_train)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except Exception:
        out_dir = getattr(args, "out_dir", None)
        if out_dir is None and getattr(args, "config", None):
            raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            out_dir = raw.get("out_dir")
        if out_dir:
            output = Path(out_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
