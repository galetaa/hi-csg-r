from __future__ import annotations

import argparse
import json
import platform
import random
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
    set_seed,
    summarize_rows,
    write_jsonl,
)
from src.htr.adapter_runtime_v2 import evaluate_v2_loader, forward_v2_batch
from src.htr.dataset_adapter_v2 import (
    DomainBalancedBatchSampler,
    HICSGRLateCorrectionDataset,
    collate_late_correction_batch,
)
from src.htr.losses_adapter_v2 import (
    auxiliary_ctc_weight,
    baseline_preservation_kl,
)
from src.htr.model_hi_csg_r_late_correction_v2 import (
    HI_CSG_R_LateCorrectionCRNNCTC,
    backbone_state_sha256,
    load_frozen_late_correction_model,
)
from src.htr.vocab import CTCVocab
from src.htr.xaligned_hi_csg_r import (
    XAlignedFeatureNormalizer,
    verify_normalizer_for_manifest,
)

DEFAULTS: dict[str, Any] = {
    "variant": "v2_1",
    "seed": 42,
    "blank_logit_penalty": -0.4,
    "alpha_max": 0.25,
    "alpha_logit_init": -6.0,
    "lambda_preservation": 0.05,
    "lambda_alignment": 0.0,
    "preservation_temperature": 1.5,
    "max_epochs": 20,
    "min_epochs": 8,
    "early_stopping_patience": 5,
    "batch_size": 16,
    "num_workers": 4,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "domain_balanced": True,
}


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    for key, value in vars(args).items():
        if key not in {"command", "config", "func"} and value is not None:
            raw[key] = value
    config = {**DEFAULTS, **raw}
    required = (
        "experiment_id",
        "base_checkpoint",
        "train_manifest",
        "val_manifest",
        "vocab",
        "normalizer",
        "risk_stats",
        "out_dir",
    )
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"Missing configuration: {', '.join(missing)}")
    if config["variant"] not in {"v2_1", "v2_2"}:
        raise ValueError("variant must be frozen to v2_1 or v2_2")
    if float(config["lambda_alignment"]) != 0.0:
        raise ValueError("Primary v2 requires lambda_alignment=0.0")
    if float(config["lambda_preservation"]) not in {0.05, 0.10}:
        raise ValueError("lambda_preservation must be 0.05 or 0.10")
    if int(config["max_epochs"]) > 20 or int(config["min_epochs"]) < 8:
        raise ValueError("Development epoch budget violates the frozen protocol")
    return config


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def load_risk_stats(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(data.get("q05", [])) != 3 or len(data.get("q95", [])) != 3:
        raise ValueError("Risk statistics do not contain three q05/q95 values")
    return data


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return total**0.5


def train_epoch(
    model: HI_CSG_R_LateCorrectionCRNNCTC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    model.train()
    total_losses: list[float] = []
    ctc_losses: list[float] = []
    preservation_losses: list[float] = []
    auxiliary_losses: list[float] = []
    gradients: list[float] = []
    empty_max = 0.0
    aux_weight = auxiliary_ctc_weight(epoch)
    for batch_index, batch in enumerate(loader, start=1):
        if config.get("max_train_batches") and batch_index > int(
            config["max_train_batches"]
        ):
            break
        optimizer.zero_grad(set_to_none=True)
        output = forward_v2_batch(
            model,
            batch,
            device,
            blank_logit_penalty=float(config["blank_logit_penalty"]),
        )
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        lengths = output["output_lengths"]
        ctc_loss = criterion(
            output["final_log_probs"],
            targets,
            lengths,
            target_lengths,
        )
        preservation = baseline_preservation_kl(
            output["base_logits"],
            output["final_logits"],
            output["visual_uncertainty"],
            batch["time_mask"].to(device),
            temperature=float(config["preservation_temperature"]),
        )
        auxiliary = criterion(
            output["aux_log_probs"],
            targets,
            lengths,
            target_lengths,
        )
        loss = (
            ctc_loss
            + float(config["lambda_preservation"]) * preservation
            + aux_weight * auxiliary
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
        loss.backward()
        gradients.append(grad_norm(model.graph_adapter))
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["grad_clip"]),
        )
        optimizer.step()
        empty = (
            batch["time_mask"].to(device)
            & ~batch["nonempty_graph_mask"].to(device)
        )
        if bool(empty.any()):
            empty_max = max(
                empty_max,
                float(output["correction_logits"][empty].abs().max().item()),
            )
        total_losses.append(float(loss.item()))
        ctc_losses.append(float(ctc_loss.item()))
        preservation_losses.append(float(preservation.item()))
        auxiliary_losses.append(float(auxiliary.item()))
    return {
        "train_total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "train_ctc_loss": float(np.mean(ctc_losses)) if ctc_losses else 0.0,
        "train_preservation_kl": (
            float(np.mean(preservation_losses)) if preservation_losses else 0.0
        ),
        "train_auxiliary_ctc": (
            float(np.mean(auxiliary_losses)) if auxiliary_losses else 0.0
        ),
        "lambda_auxiliary": aux_weight,
        "graph_adapter_grad_norm": (
            float(np.mean(gradients)) if gradients else 0.0
        ),
        "empty_correction_max": empty_max,
        "alpha": float(model.alpha().item()),
    }


def domain_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["core_domain"])].append(row)
    return {
        key: summarize_rows(values)
        for key, values in sorted(grouped.items())
    }


def save_checkpoint(
    path: Path,
    model: HI_CSG_R_LateCorrectionCRNNCTC,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    base_metadata: dict[str, Any],
    risk_stats: dict[str, Any],
    *,
    epoch: int,
    best_val_cer: float,
    validation: dict[str, Any],
) -> None:
    metadata = {
        "protocol": "crnn_ctc_hi_csg_r_late_correction_protocol_v2",
        "experiment_id": config["experiment_id"],
        "variant": config["variant"],
        "seed": int(config["seed"]),
        "base_checkpoint": base_metadata["base_checkpoint_path"],
        "base_checkpoint_sha256": base_metadata["base_checkpoint_sha256"],
        "backbone_state_sha256": base_metadata["backbone_state_sha256"],
        "train_manifest": str(Path(config["train_manifest"]).resolve()),
        "val_manifest": str(Path(config["val_manifest"]).resolve()),
        "normalizer": str(Path(config["normalizer"]).resolve()),
        "risk_stats": str(Path(config["risk_stats"]).resolve()),
        "blank_logit_penalty": float(config["blank_logit_penalty"]),
        "alpha_max": float(config["alpha_max"]),
        "lambda_preservation": float(config["lambda_preservation"]),
        "lambda_alignment": 0.0,
        "epoch": epoch,
        "best_val_cer": best_val_cer,
        "validation": validation,
        "config": config,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_cer": best_val_cer,
        "config": config,
        "metadata": metadata,
        "risk_stats": risk_stats,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_train(args: argparse.Namespace) -> None:
    config = load_config(args)
    if config.get("stage") == "final":
        holdout_path = Path(str(config.get("holdout_decision", "")))
        selection_path = Path(str(config.get("selection_artifact", "")))
        if not holdout_path.is_file() or not selection_path.is_file():
            raise ValueError(
                "Final training requires frozen selection and holdout artifacts"
            )
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if holdout.get("status") != "PASS" or selection.get("status") != "PASS":
            raise ValueError("Final training is blocked by development/holdout STOP")
        selected = selection.get("selected") or {}
        if (
            config["variant"] != selected.get("variant")
            or float(config["lambda_preservation"])
            != float(selected.get("lambda_preservation", -1.0))
        ):
            raise ValueError("Final config differs from frozen selected candidate")
    set_seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    device = torch.device(
        config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    split_audit = Path("outputs/htr_adapter_v2/split_audit/split_audit.json")
    if split_audit.exists():
        audit = json.loads(split_audit.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise ValueError("Split audit must pass before development training")
    preflight = Path("outputs/htr_adapter_v2/preflight/preflight_report.json")
    if preflight.exists():
        decision = json.loads(preflight.read_text(encoding="utf-8"))["decision"]
        if decision["status"] == "STOP":
            raise ValueError("Preflight STOP blocks v2 training")
        if config["variant"] == "v2_2" and not decision["allow_v2_2"]:
            raise ValueError("Preflight permits V2-1 only; V2-2 is blocked")

    vocab = CTCVocab.from_path(config["vocab"])
    normalizer = XAlignedFeatureNormalizer.from_path(config["normalizer"])
    verify_normalizer_for_manifest(
        normalizer,
        config["train_manifest"],
        normalizer_train_manifest=config.get("normalizer_train_manifest"),
    )
    risk_stats = load_risk_stats(config["risk_stats"])
    train_dataset = HICSGRLateCorrectionDataset(
        config["train_manifest"],
        vocab,
        normalizer,
    )
    val_dataset = HICSGRLateCorrectionDataset(
        config["val_manifest"],
        vocab,
        normalizer,
    )
    if config["domain_balanced"]:
        train_sampler: Any = DomainBalancedBatchSampler(
            train_dataset.rows,
            int(config["batch_size"]),
            seed=int(config["seed"]),
            shuffle=True,
        )
    else:
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
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_options = {
        "collate_fn": collate_late_correction_batch,
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_seed,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        **loader_options,
    )
    model, base_metadata = load_frozen_late_correction_model(
        config["base_checkpoint"],
        num_classes=vocab.num_classes,
        blank_index=vocab.blank_index,
        variant=config["variant"],
        alpha_max=float(config["alpha_max"]),
        risk_q05=risk_stats["q05"],
        risk_q95=risk_stats["q95"],
    )
    model.alpha_logit.data.fill_(float(config["alpha_logit_init"]))
    model.to(device)
    initial_backbone_hash = backbone_state_sha256(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)
    output = Path(config["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    generated = [
        output / name
        for name in (
            "best.pt",
            "last.pt",
            "history.jsonl",
            "history.json",
            "train_summary.json",
        )
    ]
    existing = [path for path in generated if path.exists()]
    if existing and not config.get("overwrite") and not config.get("resume"):
        raise FileExistsError(
            "Run artifacts already exist; use --overwrite or --resume: "
            + ", ".join(str(path) for path in existing)
        )
    if config.get("overwrite") and config.get("resume"):
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if config.get("overwrite"):
        for path in existing:
            path.unlink()
    (output / "stderr.log").touch()
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    runtime = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "python": sys.version,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "v2_module_parameters": model.trainable_module_parameter_count(),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_backbone_sha256": initial_backbone_hash,
    }
    (output / "runtime_metadata.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_cer = float("inf")
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if config.get("resume"):
        resume_path = Path(str(config["resume"]))
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        saved_config = checkpoint.get("config") or {}
        for key in (
            "variant",
            "seed",
            "base_checkpoint",
            "train_manifest",
            "val_manifest",
            "normalizer",
            "risk_stats",
            "alpha_max",
            "lambda_preservation",
        ):
            if str(saved_config.get(key)) != str(config.get(key)):
                raise ValueError(f"Resume config mismatch for {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if backbone_state_sha256(model) != initial_backbone_hash:
            raise ValueError("Resume checkpoint contains a changed backbone")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_cer = float(checkpoint.get("best_val_cer", float("inf")))
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
        if history:
            best_epoch = min(
                range(1, len(history) + 1),
                key=lambda index: float(history[index - 1]["val_CER"]),
            )
            no_improvement = len(history) - best_epoch
    started = time.monotonic()
    for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
        train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config,
            epoch,
        )
        current_hash = backbone_state_sha256(model)
        if current_hash != initial_backbone_hash:
            raise RuntimeError("Frozen backbone hash changed during training")
        validation, validation_rows = evaluate_v2_loader(
            model,
            val_loader,
            vocab,
            device,
            criterion=criterion,
            blank_logit_penalty=float(config["blank_logit_penalty"]),
            preservation_temperature=float(config["preservation_temperature"]),
            max_batches=config.get("max_val_batches"),
        )
        record = {
            "epoch": epoch,
            **train_metrics,
            "val_CER": validation["cer"],
            "val_macro_CER": validation["macro_cer"],
            "val_WER": validation["wer"],
            "val_exact": validation["exact"],
            "val_alpha": validation["alpha"],
            "val_gate": validation["gate"],
            "val_uncertainty": validation["uncertainty"],
            "val_risk": validation["risk"],
            "val_intervention": validation["intervention"],
            "val_correction": validation["correction"],
            "backbone_sha256": current_hash,
        }
        history.append(record)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (output / "stdout.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

        improved = validation["cer"] < best_cer
        if improved:
            best_cer = float(validation["cer"])
            best_epoch = epoch
            no_improvement = 0
            save_checkpoint(
                output / "best.pt",
                model,
                optimizer,
                config,
                base_metadata,
                risk_stats,
                epoch=epoch,
                best_val_cer=best_cer,
                validation=validation,
            )
            write_jsonl(output / "val_predictions.jsonl", validation_rows)
            (output / "val_summary.json").write_text(
                json.dumps(validation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output / "val_domain_summary.json").write_text(
                json.dumps(
                    domain_summaries(validation_rows),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            no_improvement += 1
        save_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            config,
            base_metadata,
            risk_stats,
            epoch=epoch,
            best_val_cer=best_cer,
            validation=validation,
        )
        if (
            epoch >= int(config["min_epochs"])
            and no_improvement >= int(config["early_stopping_patience"])
        ):
            break

    final_hash = backbone_state_sha256(model)
    if final_hash != initial_backbone_hash:
        raise RuntimeError("Frozen backbone hash differs at training completion")
    summary = {
        "experiment_id": config["experiment_id"],
        "variant": config["variant"],
        "seed": int(config["seed"]),
        "best_val_cer": best_cer,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "runtime_seconds": time.monotonic() - started,
        "best_checkpoint": str((output / "best.pt").resolve()),
        "backbone_unchanged": True,
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--device")
    train.add_argument("--num_workers", type=int)
    train.add_argument("--max_train_batches", type=int)
    train.add_argument("--max_val_batches", type=int)
    train.add_argument("--resume")
    train.add_argument("--overwrite", action="store_true", default=None)
    train.set_defaults(func=run_train)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except Exception:
        out_dir = None
        if getattr(args, "config", None):
            raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            out_dir = raw.get("out_dir")
        if out_dir:
            destination = Path(out_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "stderr.log").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
