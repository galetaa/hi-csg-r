from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tools.train_graph_fusion_ctc import (
    SimpleCTCVocab,
    read_jsonl,
    get_image_path,
    get_text,
    apply_blank_penalty,
    greedy_decode,
    compute_metrics,
    scheduled_penalty,
)


GRAPH_CHANNELS = {
    "fg": ["fg"],
    "skel": ["skel"],
    "dist": ["dist"],
    "fg_skel": ["fg", "skel"],
    "fg_dist": ["fg", "dist"],
    "skel_dist": ["skel", "dist"],
    "fg_skel_dist": ["fg", "skel", "dist"],
}


class GatedLocalFusionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        vocab: SimpleCTCVocab,
        graph_channel_mode: str = "fg_skel_dist",
    ):
        self.manifest = Path(manifest)
        self.rows = read_jsonl(self.manifest)
        self.vocab = vocab
        self.graph_channel_mode = graph_channel_mode

        if graph_channel_mode not in GRAPH_CHANNELS:
            raise ValueError(f"Unknown graph_channel_mode={graph_channel_mode}")

        self.graph_keys = GRAPH_CHANNELS[graph_channel_mode]

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_npz(self, row: dict[str, Any]) -> Path:
        value = row.get("local_graph_npz")
        if not isinstance(value, str) or not value:
            raise KeyError("Row has no local_graph_npz")

        p = Path(value)
        candidates = [p, Path.cwd() / p, self.manifest.parent / p]

        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError(value)

    @staticmethod
    def _resize_map(arr: np.ndarray, size_wh: tuple[int, int], nearest: bool) -> np.ndarray:
        mode = Image.NEAREST if nearest else Image.BILINEAR
        img = Image.fromarray(arr)
        img = img.resize(size_wh, resample=mode)
        return np.asarray(img)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]

        img_path = get_image_path(row, self.manifest)
        gray = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
        gray = (255.0 - gray) / 255.0

        h, w = gray.shape

        npz_path = self._resolve_npz(row)
        z = np.load(npz_path)

        graph_channels = []

        for key in self.graph_keys:
            arr = z[key]

            if key in {"fg", "skel"}:
                arr = arr.astype(np.uint8)
                if arr.shape != gray.shape:
                    arr = self._resize_map(arr, (w, h), nearest=True)
                arr = arr.astype(np.float32)

            elif key == "dist":
                arr = arr.astype(np.float32)
                if arr.shape != gray.shape:
                    arr = self._resize_map(arr, (w, h), nearest=False).astype(np.float32)

            else:
                raise KeyError(key)

            graph_channels.append(arr)

        gray_tensor = torch.from_numpy(gray[None, :, :]).float()
        graph_tensor = torch.from_numpy(np.stack(graph_channels, axis=0)).float()

        text = get_text(row)
        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)

        return {
            "gray": gray_tensor,
            "graph": graph_tensor,
            "width": int(w),
            "height": int(h),
            "target": target,
            "target_length": int(len(target)),
            "text": text,
            "dataset": row.get("dataset", row.get("source_dataset", "")),
            "sample_id": row.get("sample_id", str(idx)),
        }


def collate_gated(batch: list[dict[str, Any]]) -> dict[str, Any]:
    graph_c = batch[0]["graph"].shape[0]
    max_h = max(x["gray"].shape[1] for x in batch)
    max_w = max(x["gray"].shape[2] for x in batch)

    grays = []
    graphs = []
    widths = []
    targets = []
    target_lengths = []
    texts = []
    datasets = []
    sample_ids = []

    for x in batch:
        _, h, w = x["gray"].shape

        gray_pad = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        graph_pad = torch.zeros((graph_c, max_h, max_w), dtype=torch.float32)

        gray_pad[:, :h, :w] = x["gray"]
        graph_pad[:, :h, :w] = x["graph"]

        grays.append(gray_pad)
        graphs.append(graph_pad)
        widths.append(x["width"])
        targets.append(x["target"])
        target_lengths.append(x["target_length"])
        texts.append(x["text"])
        datasets.append(x["dataset"])
        sample_ids.append(x["sample_id"])

    return {
        "grays": torch.stack(grays, dim=0),
        "graphs": torch.stack(graphs, dim=0),
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.cat(targets, dim=0),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
        "datasets": datasets,
        "sample_ids": sample_ids,
    }


class GatedLocalFusionCRNNCTC(nn.Module):
    def __init__(
        self,
        num_classes: int,
        graph_channels: int,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        graph_dropout: float = 0.2,
        blank_index: int = 0,
        blank_bias_init: float = -1.0,
        height_bins: int = 4,
        feature_size: int = 256,
        gate_bias_init: float = -4.0,
    ):
        super().__init__()

        self.blank_index = blank_index
        self.height_bins = height_bins
        self.width_downsample = 4

        self.gray_cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )

        self.graph_cnn = nn.Sequential(
            nn.Conv2d(graph_channels, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

        self.graph_proj = nn.Conv2d(128, 256, kernel_size=1)
        self.graph_dropout = nn.Dropout2d(graph_dropout)

        self.gate_conv = nn.Conv2d(512, 256, kernel_size=1)

        with torch.no_grad():
            self.gate_conv.weight.zero_()
            self.gate_conv.bias.fill_(gate_bias_init)

        visual_dim = 256 * height_bins

        self.input_proj = nn.Sequential(
            nn.Linear(visual_dim, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

        with torch.no_grad():
            self.classifier.bias.zero_()
            self.classifier.bias[blank_index] = blank_bias_init

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(widths // self.width_downsample, min=1)

    def forward(
        self,
        grays: torch.Tensor,
        graphs: torch.Tensor,
        widths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gray_feat = self.gray_cnn(grays)

        graph_feat = self.graph_cnn(graphs)
        graph_feat = self.graph_proj(graph_feat)
        graph_feat = self.graph_dropout(graph_feat)

        gate = torch.sigmoid(self.gate_conv(torch.cat([gray_feat, graph_feat], dim=1)))

        fused = gray_feat + gate * graph_feat

        feat = F.adaptive_avg_pool2d(fused, (self.height_bins, fused.shape[-1]))

        b, c, h, t = feat.shape
        x = feat.permute(0, 3, 1, 2).contiguous().view(b, t, c * h)

        x = self.input_proj(x)
        x, _ = self.lstm(x)
        logits = self.classifier(x)

        out_lens = self.output_lengths(widths)
        out_lens = torch.clamp(out_lens, max=logits.shape[1])

        gate_mean = gate.detach().mean()

        return logits, out_lens, gate_mean


@torch.no_grad()
def evaluate(
    model: GatedLocalFusionCRNNCTC,
    loader: DataLoader,
    vocab: SimpleCTCVocab,
    device: torch.device,
    blank_penalty: float,
) -> dict[str, Any]:
    model.eval()

    all_targets = []
    all_preds = []
    all_datasets = []
    all_sample_ids = []

    blank_argmax = 0
    total_argmax = 0
    gate_means = []

    for batch in loader:
        grays = batch["grays"].to(device)
        graphs = batch["graphs"].to(device)
        widths = batch["widths"].to(device)

        logits, out_lens, gate_mean = model(grays, graphs, widths)
        gate_means.append(float(gate_mean.item()))

        penalized = apply_blank_penalty(logits, vocab.blank_index, blank_penalty)
        argmax = penalized.argmax(dim=-1)

        mask = torch.arange(argmax.shape[1], device=device)[None, :] < out_lens[:, None]
        blank_argmax += int(((argmax == vocab.blank_index) & mask).sum().item())
        total_argmax += int(mask.sum().item())

        preds = greedy_decode(logits, out_lens, vocab, blank_penalty)

        all_targets.extend(batch["texts"])
        all_preds.extend(preds)
        all_datasets.extend(batch["datasets"])
        all_sample_ids.extend(batch["sample_ids"])

    metrics = compute_metrics(all_targets, all_preds)
    metrics["argmax_blank_ratio"] = blank_argmax / max(total_argmax, 1)
    metrics["gate_mean"] = float(np.mean(gate_means)) if gate_means else 0.0

    by_ds = defaultdict(lambda: {"targets": [], "preds": []})
    for ds, t, p in zip(all_datasets, all_targets, all_preds):
        by_ds[ds]["targets"].append(t)
        by_ds[ds]["preds"].append(p)

    metrics["grouped"] = {
        ds: compute_metrics(vals["targets"], vals["preds"])
        for ds, vals in by_ds.items()
    }

    predictions = [
        {
            "sample_id": sid,
            "dataset": ds,
            "target": t,
            "pred": p,
        }
        for sid, ds, t, p in zip(all_sample_ids, all_datasets, all_targets, all_preds)
    ]

    return {
        "metrics": metrics,
        "predictions": predictions,
    }


def train_one_epoch(
    model: GatedLocalFusionCRNNCTC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    blank_penalty: float,
    log_every: int,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    losses = []
    gates = []

    for step, batch in enumerate(loader, 1):
        grays = batch["grays"].to(device)
        graphs = batch["graphs"].to(device)
        widths = batch["widths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, out_lens, gate_mean = model(grays, graphs, widths)

        logits = logits.float()
        logits = apply_blank_penalty(logits, model.blank_index, blank_penalty)

        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

        loss = criterion(
            log_probs,
            targets,
            out_lens,
            target_lengths,
        )

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        losses.append(float(loss.item()))
        gates.append(float(gate_mean.item()))

        if log_every > 0 and step % log_every == 0:
            print(
                f"step {step}/{len(loader)} "
                f"loss={np.mean(losses[-log_every:]):.4f} "
                f"gate={np.mean(gates[-log_every:]):.4f}"
            )

    return (
        float(np.mean(losses)) if losses else 0.0,
        float(np.mean(gates)) if gates else 0.0,
    )


def make_loader(
    manifest: Path,
    vocab: SimpleCTCVocab,
    graph_channel_mode: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    ds = GatedLocalFusionDataset(
        manifest=manifest,
        vocab=vocab,
        graph_channel_mode=graph_channel_mode,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_gated,
        pin_memory=torch.cuda.is_available(),
    )


def cmd_train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = SimpleCTCVocab.from_json(args.vocab)

    graph_channels = len(GRAPH_CHANNELS[args.graph_channel_mode])

    train_loader = make_loader(
        manifest=Path(args.train_manifest),
        vocab=vocab,
        graph_channel_mode=args.graph_channel_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_loader = make_loader(
        manifest=Path(args.val_manifest),
        vocab=vocab,
        graph_channel_mode=args.graph_channel_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = GatedLocalFusionCRNNCTC(
        num_classes=vocab.num_classes,
        graph_channels=graph_channels,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        graph_dropout=args.graph_dropout,
        blank_index=vocab.blank_index,
        blank_bias_init=args.blank_bias_init,
        height_bins=args.height_bins,
        feature_size=args.feature_size,
        gate_bias_init=args.gate_bias_init,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if args.amp and not use_amp:
        print("amp requested but disabled because device is not cuda")

    config = {
        "num_classes": vocab.num_classes,
        "graph_channels": graph_channels,
        "graph_channel_mode": args.graph_channel_mode,
        "hidden_size": args.hidden_size,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "graph_dropout": args.graph_dropout,
        "blank_index": vocab.blank_index,
        "blank_bias_init": args.blank_bias_init,
        "height_bins": args.height_bins,
        "feature_size": args.feature_size,
        "gate_bias_init": args.gate_bias_init,
        "vocab": str(args.vocab),
        "amp": use_amp,
    }

    history = []
    best_cer = math.inf
    best_epoch = None
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        scaler_state = ckpt.get("scaler_state")
        if scaler_state and use_amp:
            scaler.load_state_dict(scaler_state)

        history = ckpt.get("history", [])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

        if history:
            best_row = min(history, key=lambda r: r["val"]["cer"])
            best_cer = best_row["val"]["cer"]
            best_epoch = best_row["epoch"]
        else:
            best_cer = ckpt.get("val", {}).get("cer", math.inf)
            best_epoch = ckpt.get("epoch")

        print(f"resumed from {args.resume}, next_epoch={start_epoch}, best_epoch={best_epoch}, best_cer={best_cer}")

    for epoch in range(start_epoch, args.epochs + 1):
        pen = scheduled_penalty(
            epoch,
            args.epochs,
            args.blank_logit_penalty_start,
            args.blank_logit_penalty_end,
        )

        print(f"\n=== epoch {epoch}/{args.epochs} blank_penalty={pen:.4f} ===")

        train_loss, train_gate = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            blank_penalty=pen,
            log_every=args.log_every,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_result = evaluate(
            model=model,
            loader=val_loader,
            vocab=vocab,
            device=device,
            blank_penalty=pen,
        )

        val_metrics = val_result["metrics"]

        row = {
            "epoch": epoch,
            "blank_logit_penalty": pen,
            "train_loss": train_loss,
            "train_gate_mean": train_gate,
            "val": val_metrics,
        }
        history.append(row)

        print(
            "epoch", epoch,
            "loss", round(train_loss, 4),
            "train_gate", round(train_gate, 4),
            "val_cer", round(val_metrics["cer"], 4),
            "val_wer", round(val_metrics["wer"], 4),
            "exact", round(val_metrics["exact"], 4),
            "pred_len", round(val_metrics["pred_len_mean"], 2),
            "empty", round(val_metrics["pred_empty_ratio"], 4),
            "blank", round(val_metrics["argmax_blank_ratio"], 4),
            "val_gate", round(val_metrics["gate_mean"], 4),
        )

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            best_epoch = epoch

            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "config": config,
                "epoch": epoch,
                "val": val_metrics,
                "blank_logit_penalty": pen,
                "history": history,
                "best_cer": best_cer,
                "best_epoch": best_epoch,
            }

            torch.save(ckpt, out_dir / "best.pt")

            (out_dir / "best_val_examples.json").write_text(
                json.dumps(val_result["predictions"][:100], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"saved best.pt epoch={epoch} val_cer={best_cer:.6f}")

        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        last_ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "config": config,
            "epoch": epoch,
            "val": val_metrics,
            "blank_logit_penalty": pen,
            "history": history,
            "best_cer": best_cer,
            "best_epoch": best_epoch,
        }
        torch.save(last_ckpt, out_dir / "last.pt")

    print("best_epoch:", best_epoch)
    print("best_cer:", best_cer)


def load_checkpoint_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[GatedLocalFusionCRNNCTC, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = GatedLocalFusionCRNNCTC(
        num_classes=cfg["num_classes"],
        graph_channels=cfg["graph_channels"],
        hidden_size=cfg["hidden_size"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
        graph_dropout=cfg["graph_dropout"],
        blank_index=cfg["blank_index"],
        blank_bias_init=cfg["blank_bias_init"],
        height_bins=cfg["height_bins"],
        feature_size=cfg["feature_size"],
        gate_bias_init=cfg["gate_bias_init"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def cmd_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)

    checkpoint = Path(args.checkpoint)
    model, ckpt = load_checkpoint_model(checkpoint, device)

    vocab_path = args.vocab or ckpt["config"]["vocab"]
    vocab = SimpleCTCVocab.from_json(vocab_path)

    loader = make_loader(
        manifest=Path(args.manifest),
        vocab=vocab,
        graph_channel_mode=ckpt["config"]["graph_channel_mode"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    result = evaluate(
        model=model,
        loader=loader,
        vocab=vocab,
        device=device,
        blank_penalty=args.blank_logit_penalty,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = result["metrics"]

    summary = {
        "n": m["n"],
        "cer": m["cer"],
        "wer": m["wer"],
        "exact": m["exact"],
        "pred_len_mean": m["pred_len_mean"],
        "pred_empty_ratio": m["pred_empty_ratio"],
        "argmax_blank_ratio": m["argmax_blank_ratio"],
        "gate_mean": m["gate_mean"],
        "grouped": m.get("grouped", {}),
        "out_dir": str(out_dir),
        "blank_logit_penalty": args.blank_logit_penalty,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_cer": ckpt["val"]["cer"],
        "graph_channel_mode": ckpt["config"]["graph_channel_mode"],
        "graph_channels": ckpt["config"]["graph_channels"],
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in result["predictions"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--train_manifest", required=True)
    train.add_argument("--val_manifest", required=True)
    train.add_argument("--vocab", required=True)
    train.add_argument("--out_dir", required=True)

    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch_size", type=int, default=64)
    train.add_argument("--num_workers", type=int, default=2)
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--graph_dropout", type=float, default=0.2)
    train.add_argument("--hidden_size", type=int, default=256)
    train.add_argument("--lstm_layers", type=int, default=2)
    train.add_argument("--blank_bias_init", type=float, default=-1.0)
    train.add_argument("--blank_logit_penalty_start", type=float, default=-2.0)
    train.add_argument("--blank_logit_penalty_end", type=float, default=-0.4)
    train.add_argument("--height_bins", type=int, default=4)
    train.add_argument("--feature_size", type=int, default=256)
    train.add_argument("--graph_channel_mode", default="fg_skel_dist")
    train.add_argument("--gate_bias_init", type=float, default=-4.0)
    train.add_argument("--log_every", type=int, default=50)
    train.add_argument("--seed", type=int, default=70)
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--resume", default=None)
    train.add_argument("--amp", action="store_true")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--out_dir", required=True)
    ev.add_argument("--vocab", default=None)
    ev.add_argument("--batch_size", type=int, default=64)
    ev.add_argument("--num_workers", type=int, default=2)
    ev.add_argument("--blank_logit_penalty", type=float, required=True)
    ev.add_argument("--cpu", action="store_true")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
