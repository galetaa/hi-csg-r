from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.metrics import cer, exact_match, wer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def target_text(row: dict[str, Any]) -> str:
    return str(
        row.get("text")
        or row.get("normalized_transcription")
        or row.get("raw_transcription")
        or ""
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class TrocrManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, max_samples: int | None = None) -> None:
        self.rows = read_jsonl(Path(manifest))
        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "image": Image.open(row["image_path"]).convert("RGB"),
            "text": target_text(row),
            "sample_id": row.get("sample_id"),
            "dataset": row.get("dataset"),
            "level": row.get("level"),
            "category": row.get("category"),
        }


class TrocrCollator:
    def __init__(self, processor: TrOCRProcessor, max_target_length: int) -> None:
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        images = [row["image"] for row in rows]
        texts = [row["text"] for row in rows]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "texts": texts,
            "sample_ids": [row["sample_id"] for row in rows],
            "datasets": [row["dataset"] for row in rows],
            "levels": [row["level"] for row in rows],
            "categories": [row["category"] for row in rows],
        }


def configure_model(model: VisionEncoderDecoderModel, processor: TrOCRProcessor) -> None:
    tokenizer = processor.tokenizer
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    model.config.use_cache = False
    if hasattr(model.decoder, "config"):
        model.decoder.config.use_cache = False
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id


def set_trainable_scope(
    model: VisionEncoderDecoderModel,
    *,
    freeze_encoder: bool,
    freeze_decoder_embeddings: bool,
) -> dict[str, int]:
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    if freeze_decoder_embeddings:
        embeddings = model.decoder.get_input_embeddings()
        if embeddings is not None:
            for param in embeddings.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "frozen_parameters": int(total - trainable),
    }


def evaluate_loss(
    *,
    model: VisionEncoderDecoderModel,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
                out = model(pixel_values=pixel_values, labels=labels)
            losses.append(float(out.loss.detach().cpu()))
    return {"loss": mean(losses), "batches": len(losses)}


def evaluate_generation(
    *,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
    max_batches: int | None,
    max_new_tokens: int,
    num_beams: int,
) -> dict[str, Any]:
    model.eval()
    cers: list[float] = []
    wers: list[float] = []
    exacts: list[float] = []
    pred_lens: list[float] = []
    examples: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            if fp16 and device.type == "cuda":
                pixel_values = pixel_values.half()
            generated = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            preds = processor.batch_decode(generated, skip_special_tokens=True)

            for pred, target, sample_id, dataset, level, category in zip(
                preds,
                batch["texts"],
                batch["sample_ids"],
                batch["datasets"],
                batch["levels"],
                batch["categories"],
            ):
                pred = str(pred).strip()
                c = cer(pred, target)
                w = wer(pred, target)
                e = exact_match(pred, target)
                cers.append(c)
                wers.append(w)
                exacts.append(e)
                pred_lens.append(len(pred))
                if len(examples) < 30:
                    examples.append({
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "level": level,
                        "category": category,
                        "target": target,
                        "pred": pred,
                        "cer": c,
                        "wer": w,
                        "exact": e,
                    })

    return {
        "n": len(cers),
        "cer": mean(cers),
        "wer": mean(wers),
        "exact": mean(exacts),
        "pred_len_mean": mean(pred_lens),
        "examples": examples,
    }


def save_checkpoint(
    *,
    out_dir: Path,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    name: str,
) -> str:
    path = out_dir / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_id", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_loss_batches", type=int, default=None)
    parser.add_argument("--max_val_generate_batches", type=int, default=None)
    parser.add_argument("--val_generate_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable transformer gradient checkpointing to lower activation memory.",
    )
    parser.add_argument(
        "--adamw_foreach",
        action="store_true",
        help="Use foreach AdamW kernels. Disabled by default because they need extra peak GPU memory.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Train only the decoder side of TrOCR. Useful on small GPUs when full fine-tuning does not fit.",
    )
    parser.add_argument(
        "--freeze_decoder_embeddings",
        action="store_true",
        help="Freeze decoder token embeddings to reduce trainable parameters.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.json"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    fp16 = bool(args.fp16 and device.type == "cuda")

    resume_dir = out_dir / "last"
    model_source = resume_dir if resume_dir.exists() else args.model_id
    processor = TrOCRProcessor.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    configure_model(model, processor)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    trainable_scope = set_trainable_scope(
        model,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder_embeddings=args.freeze_decoder_embeddings,
    )
    model.to(device)

    train_ds = TrocrManifestDataset(args.train_manifest, max_samples=args.max_train_samples)
    val_ds = TrocrManifestDataset(args.val_manifest, max_samples=args.max_val_samples)
    collator = TrocrCollator(processor, max_target_length=args.max_target_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=args.adamw_foreach,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    history: list[dict[str, Any]] = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    start_epoch = int(history[-1]["epoch"]) + 1 if history else 1
    best_val_cer = min(
        (
            float(row["val_generate"]["cer"])
            for row in history
            if row.get("val_generate") and row["val_generate"].get("cer") is not None
        ),
        default=float("inf"),
    )
    best_val_loss = min(
        (
            float(row["val_loss"]["loss"])
            for row in history
            if row.get("val_loss") and row["val_loss"].get("loss") is not None
        ),
        default=float("inf"),
    )

    config = vars(args).copy()
    config.update({
        "device_resolved": str(device),
        "fp16_enabled": fp16,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "start_epoch": start_epoch,
        "model_source": str(model_source),
        "trainable_scope": trainable_scope,
    })
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and batch_idx > args.max_train_batches:
                break

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=fp16):
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if batch_idx % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(float(out.loss.detach().cpu()))

            if batch_idx % args.log_every == 0:
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                    f"loss={mean(train_losses[-args.log_every:]):.4f}",
                    flush=True,
                )

        if len(train_losses) % args.grad_accum_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            fp16=fp16,
            max_batches=args.max_val_loss_batches,
        )

        val_generate = None
        if args.val_generate_every > 0 and epoch % args.val_generate_every == 0:
            val_generate = evaluate_generation(
                model=model,
                processor=processor,
                loader=val_loader,
                device=device,
                fp16=fp16,
                max_batches=args.max_val_generate_batches,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

        row = {
            "epoch": epoch,
            "train_loss": mean(train_losses),
            "val_loss": val_loss,
            "val_generate": val_generate,
        }
        history.append(row)
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        save_checkpoint(
            out_dir=out_dir,
            model=model,
            processor=processor,
            name="last",
        )

        is_best = False
        if val_generate is not None and val_generate["cer"] < best_val_cer:
            best_val_cer = val_generate["cer"]
            is_best = True
        elif val_generate is None and val_loss["loss"] < best_val_loss:
            best_val_loss = val_loss["loss"]
            is_best = True

        if val_loss["loss"] < best_val_loss:
            best_val_loss = val_loss["loss"]

        if is_best:
            save_checkpoint(
                out_dir=out_dir,
                model=model,
                processor=processor,
                name="best",
            )

        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_loss['loss']:.4f} "
            f"val_cer={(val_generate or {}).get('cer', None)} "
            f"best_val_cer={best_val_cer}",
            flush=True,
        )

    print("wrote:", out_dir, flush=True)


if __name__ == "__main__":
    main()
