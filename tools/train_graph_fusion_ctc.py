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


IMAGE_KEYS = ["image_path", "path", "crop_path", "file_path", "img_path", "image"]


# -------------------------
# Vocab
# -------------------------

class SimpleCTCVocab:
    def __init__(self, tokens: list[str], blank_index: int = 0):
        self.tokens = tokens
        self.blank_index = blank_index
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(tokens)}

    @classmethod
    def from_json(cls, path: str | Path) -> "SimpleCTCVocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))

        blank_index = int(obj.get("blank_index", 0))

        if "idx_to_token" in obj:
            tokens = list(obj["idx_to_token"])
        elif "itos" in obj:
            tokens = list(obj["itos"])
        elif "idx_to_char" in obj:
            tokens = list(obj["idx_to_char"])
        elif "chars" in obj:
            blank_token = obj.get("blank_token", "<blank>")
            tokens = [blank_token] + list(obj["chars"])
            blank_index = 0
        else:
            raise ValueError(f"Unknown vocab format. Keys: {sorted(obj.keys())}")

        return cls(tokens=tokens, blank_index=blank_index)

    @property
    def num_classes(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch not in self.token_to_idx:
                raise KeyError(f"Character not in vocab: {repr(ch)}")
            idx = self.token_to_idx[ch]
            if idx == self.blank_index:
                raise ValueError("Target text contains blank token")
            ids.append(idx)
        return ids

    def decode_indices(self, indices: list[int]) -> str:
        chars = []
        for idx in indices:
            if idx == self.blank_index:
                continue
            chars.append(self.idx_to_token.get(int(idx), ""))
        return "".join(chars)


# -------------------------
# Data
# -------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_image_path(row: dict[str, Any], manifest_path: Path) -> Path:
    value = None
    for key in IMAGE_KEYS:
        if isinstance(row.get(key), str) and row[key]:
            value = row[key]
            break

    if value is None:
        raise KeyError(f"No image path key in row. Keys: {sorted(row.keys())}")

    p = Path(value)
    candidates = [
        p,
        Path.cwd() / p,
        manifest_path.parent / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(value)


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "label", "transcription", "target"]:
        if key in row:
            return str(row[key])
    return ""


def select_graph_feature_indices(
    feature_names: list[str],
    drop_names: set[str],
) -> tuple[list[int], list[str]]:
    idxs = []
    names = []

    for i, name in enumerate(feature_names):
        if name in drop_names:
            continue
        idxs.append(i)
        names.append(name)

    return idxs, names


def infer_feature_names(manifest: Path) -> list[str]:
    for row in read_jsonl(manifest):
        if "graph_feature_names" in row:
            return list(row["graph_feature_names"])
    raise RuntimeError(f"No graph_feature_names found in {manifest}")


def compute_graph_stats(
    manifest: Path,
    selected_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    rows = read_jsonl(manifest)
    feats = []

    for r in rows:
        if r.get("graph_valid") is False:
            continue
        gf = r.get("graph_features")
        if gf is None:
            raise KeyError("Row has no graph_features")
        feats.append([float(gf[i]) for i in selected_indices])

    if not feats:
        raise RuntimeError(f"No graph_valid rows found in {manifest}")

    arr = np.asarray(feats, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


class GraphHTRDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        vocab: SimpleCTCVocab,
        graph_indices: list[int],
        graph_mean: np.ndarray,
        graph_std: np.ndarray,
        force_zero_graph: bool = False,
    ):
        self.manifest = Path(manifest)
        self.rows = read_jsonl(self.manifest)
        self.vocab = vocab
        self.graph_indices = graph_indices
        self.graph_mean = graph_mean.astype(np.float32)
        self.graph_std = graph_std.astype(np.float32)
        self.force_zero_graph = force_zero_graph

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]

        img_path = get_image_path(row, self.manifest)
        arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)

        # Same convention as previous HTR dataset: black ink -> positive signal.
        arr = (255.0 - arr) / 255.0

        h, w = arr.shape
        image = torch.from_numpy(arr[None, :, :]).float()

        text = get_text(row)
        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)

        gf = row.get("graph_features")
        if gf is None:
            raise KeyError("Row has no graph_features")

        graph_valid = bool(row.get("graph_valid", True))

        if self.force_zero_graph or not graph_valid:
            graph = np.zeros(len(self.graph_indices), dtype=np.float32)
            graph_valid = False
        else:
            graph = np.asarray([float(gf[i]) for i in self.graph_indices], dtype=np.float32)
            graph = (graph - self.graph_mean) / self.graph_std

        graph = torch.from_numpy(graph).float()

        return {
            "image": image,
            "width": int(w),
            "height": int(h),
            "target": target,
            "target_length": int(len(target)),
            "text": text,
            "dataset": row.get("dataset", row.get("source_dataset", "")),
            "sample_id": row.get("sample_id", str(idx)),
            "graph": graph,
            "graph_valid": graph_valid,
        }


def collate_graph_htr(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_h = max(x["image"].shape[1] for x in batch)
    max_w = max(x["image"].shape[2] for x in batch)

    images = []
    widths = []
    targets = []
    target_lengths = []
    texts = []
    datasets = []
    sample_ids = []
    graphs = []
    graph_valids = []

    for x in batch:
        img = x["image"]
        _, h, w = img.shape
        padded = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        padded[:, :h, :w] = img

        images.append(padded)
        widths.append(x["width"])
        targets.append(x["target"])
        target_lengths.append(x["target_length"])
        texts.append(x["text"])
        datasets.append(x["dataset"])
        sample_ids.append(x["sample_id"])
        graphs.append(x["graph"])
        graph_valids.append(x["graph_valid"])

    return {
        "images": torch.stack(images, dim=0),
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.cat(targets, dim=0),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
        "datasets": datasets,
        "sample_ids": sample_ids,
        "graphs": torch.stack(graphs, dim=0),
        "graph_valids": torch.tensor(graph_valids, dtype=torch.bool),
    }


# -------------------------
# Model
# -------------------------

class GraphFusionCRNNCTC(nn.Module):
    def __init__(
        self,
        num_classes: int,
        graph_dim: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        blank_index: int = 0,
        blank_bias_init: float = -1.0,
        height_bins: int = 4,
        feature_size: int = 256,
        graph_hidden_dim: int = 64,
        graph_embed_dim: int = 128,
        graph_dropout: float = 0.1,
    ):
        super().__init__()

        self.blank_index = blank_index
        self.height_bins = height_bins
        self.width_downsample = 4

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
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

        visual_dim = 256 * height_bins

        self.graph_mlp = nn.Sequential(
            nn.Linear(graph_dim, graph_hidden_dim),
            nn.LayerNorm(graph_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(graph_dropout),
            nn.Linear(graph_hidden_dim, graph_embed_dim),
            nn.LayerNorm(graph_embed_dim),
            nn.ReLU(inplace=True),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(visual_dim + graph_embed_dim, feature_size),
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
        images: torch.Tensor,
        widths: torch.Tensor,
        graphs: torch.Tensor,
        graph_valids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.cnn(images)

        # Preserve vertical structure, collapse only to fixed height bins.
        feat = F.adaptive_avg_pool2d(feat, (self.height_bins, feat.shape[-1]))

        b, c, h, t = feat.shape
        visual = feat.permute(0, 3, 1, 2).contiguous().view(b, t, c * h)

        g = self.graph_mlp(graphs)
        if graph_valids is not None:
            g = g * graph_valids.to(dtype=g.dtype)[:, None]
        g = g[:, None, :].expand(b, t, g.shape[-1])

        x = torch.cat([visual, g], dim=-1)
        x = self.input_proj(x)

        x, _ = self.lstm(x)
        logits = self.classifier(x)

        out_lens = self.output_lengths(widths)
        out_lens = torch.clamp(out_lens, max=logits.shape[1])

        return logits, out_lens


# -------------------------
# Metrics / decoding
# -------------------------

def apply_blank_penalty(logits: torch.Tensor, blank_index: int, penalty: float) -> torch.Tensor:
    if penalty == 0:
        return logits
    logits = logits.clone()
    logits[..., blank_index] += penalty
    return logits


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, vocab: SimpleCTCVocab, blank_penalty: float) -> list[str]:
    logits = apply_blank_penalty(logits, vocab.blank_index, blank_penalty)
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    lengths_np = lengths.detach().cpu().numpy()

    texts = []
    for seq, ln in zip(pred, lengths_np):
        collapsed = []
        prev = None
        for idx in seq[: int(ln)]:
            idx = int(idx)
            if idx != prev and idx != vocab.blank_index:
                collapsed.append(idx)
            prev = idx
        texts.append(vocab.decode_indices(collapsed))

    return texts


def edit_distance(a: list[Any], b: list[Any]) -> int:
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur

    return prev[-1]


def compute_metrics(targets: list[str], preds: list[str]) -> dict[str, float]:
    char_dist = 0
    char_total = 0
    word_dist = 0
    word_total = 0
    exact = 0

    for t, p in zip(targets, preds):
        char_dist += edit_distance(list(t), list(p))
        char_total += max(len(t), 1)

        tw = t.split()
        pw = p.split()
        word_dist += edit_distance(tw, pw)
        word_total += max(len(tw), 1)

        exact += int(t == p)

    n = max(len(targets), 1)
    return {
        "n": len(targets),
        "cer": char_dist / max(char_total, 1),
        "wer": word_dist / max(word_total, 1),
        "exact": exact / n,
        "pred_len_mean": float(np.mean([len(p) for p in preds])) if preds else 0.0,
        "pred_empty_ratio": float(np.mean([len(p) == 0 for p in preds])) if preds else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: GraphFusionCRNNCTC,
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

    for batch in loader:
        images = batch["images"].to(device)
        widths = batch["widths"].to(device)
        graphs = batch["graphs"].to(device)
        graph_valids = batch["graph_valids"].to(device)

        logits, out_lens = model(images, widths, graphs, graph_valids)
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

    grouped = {}
    by_ds = defaultdict(lambda: {"targets": [], "preds": []})
    for ds, t, p in zip(all_datasets, all_targets, all_preds):
        by_ds[ds]["targets"].append(t)
        by_ds[ds]["preds"].append(p)

    for ds, vals in by_ds.items():
        grouped[ds] = compute_metrics(vals["targets"], vals["preds"])

    metrics["grouped"] = grouped

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
    model: GraphFusionCRNNCTC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    blank_penalty: float,
    log_every: int,
) -> float:
    model.train()
    losses = []

    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device)
        widths = batch["widths"].to(device)
        graphs = batch["graphs"].to(device)
        graph_valids = batch["graph_valids"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        logits, out_lens = model(images, widths, graphs, graph_valids)
        logits = apply_blank_penalty(logits, model.blank_index, blank_penalty)

        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

        loss = criterion(
            log_probs,
            targets,
            out_lens,
            target_lengths,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        losses.append(float(loss.item()))

        if log_every > 0 and step % log_every == 0:
            print(f"step {step}/{len(loader)} loss={np.mean(losses[-log_every:]):.4f}")

    return float(np.mean(losses)) if losses else 0.0


def scheduled_penalty(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    alpha = (epoch - 1) / (epochs - 1)
    return start + alpha * (end - start)


# -------------------------
# Commands
# -------------------------

def build_loaders(
    train_manifest: Path | None,
    val_manifest: Path | None,
    eval_manifest: Path | None,
    vocab: SimpleCTCVocab,
    graph_indices: list[int],
    graph_mean: np.ndarray,
    graph_std: np.ndarray,
    batch_size: int,
    num_workers: int,
    force_zero_graph: bool = False,
):
    train_loader = None
    val_loader = None
    eval_loader = None

    if train_manifest is not None:
        ds = GraphHTRDataset(
            train_manifest,
            vocab,
            graph_indices,
            graph_mean,
            graph_std,
            force_zero_graph=force_zero_graph,
        )
        train_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_graph_htr,
            pin_memory=torch.cuda.is_available(),
        )

    if val_manifest is not None:
        ds = GraphHTRDataset(
            val_manifest,
            vocab,
            graph_indices,
            graph_mean,
            graph_std,
            force_zero_graph=force_zero_graph,
        )
        val_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_graph_htr,
            pin_memory=torch.cuda.is_available(),
        )

    if eval_manifest is not None:
        ds = GraphHTRDataset(
            eval_manifest,
            vocab,
            graph_indices,
            graph_mean,
            graph_std,
            force_zero_graph=force_zero_graph,
        )
        eval_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_graph_htr,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, eval_loader


def cmd_train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = SimpleCTCVocab.from_json(args.vocab)

    feature_names_all = infer_feature_names(Path(args.train_manifest))
    drop_names = set(args.drop_graph_features.split(",")) if args.drop_graph_features else set()
    graph_indices, graph_feature_names = select_graph_feature_indices(feature_names_all, drop_names)

    if "text_len" in graph_feature_names:
        raise RuntimeError("text_len leakage detected in selected graph features")

    graph_mean, graph_std = compute_graph_stats(Path(args.train_manifest), graph_indices)

    (out_dir / "graph_stats.json").write_text(
        json.dumps(
            {
                "feature_names_all": feature_names_all,
                "selected_feature_names": graph_feature_names,
                "dropped_feature_names": sorted(drop_names),
                "mean": graph_mean.tolist(),
                "std": graph_std.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train_loader, val_loader, _ = build_loaders(
        train_manifest=Path(args.train_manifest),
        val_manifest=Path(args.val_manifest),
        eval_manifest=None,
        vocab=vocab,
        graph_indices=graph_indices,
        graph_mean=graph_mean,
        graph_std=graph_std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = GraphFusionCRNNCTC(
        num_classes=vocab.num_classes,
        graph_dim=len(graph_indices),
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        blank_index=vocab.blank_index,
        blank_bias_init=args.blank_bias_init,
        height_bins=args.height_bins,
        feature_size=args.feature_size,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_embed_dim=args.graph_embed_dim,
        graph_dropout=args.graph_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)

    history = []
    best_cer = math.inf
    best_epoch = None

    config = {
        "num_classes": vocab.num_classes,
        "graph_dim": len(graph_indices),
        "graph_feature_names": graph_feature_names,
        "drop_graph_features": sorted(drop_names),
        "hidden_size": args.hidden_size,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "blank_index": vocab.blank_index,
        "blank_bias_init": args.blank_bias_init,
        "height_bins": args.height_bins,
        "feature_size": args.feature_size,
        "graph_hidden_dim": args.graph_hidden_dim,
        "graph_embed_dim": args.graph_embed_dim,
        "graph_dropout": args.graph_dropout,
        "vocab": str(args.vocab),
    }

    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

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

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            blank_penalty=pen,
            log_every=args.log_every,
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
            "val": val_metrics,
        }
        history.append(row)

        print(
            "epoch", epoch,
            "loss", round(train_loss, 4),
            "val_cer", round(val_metrics["cer"], 4),
            "val_wer", round(val_metrics["wer"], 4),
            "exact", round(val_metrics["exact"], 4),
            "pred_len", round(val_metrics["pred_len_mean"], 2),
            "empty", round(val_metrics["pred_empty_ratio"], 4),
            "blank", round(val_metrics["argmax_blank_ratio"], 4),
        )

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            best_epoch = epoch

            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "graph_mean": graph_mean.tolist(),
                "graph_std": graph_std.tolist(),
                "graph_indices": graph_indices,
                "feature_names_all": feature_names_all,
                "epoch": epoch,
                "val": val_metrics,
                "blank_logit_penalty": pen,
                "history": history,
                "best_cer": best_cer,
                "best_epoch": best_epoch,
            }
            torch.save(ckpt, out_dir / "best.pt")

            examples = val_result["predictions"][:100]
            (out_dir / "best_val_examples.json").write_text(
                json.dumps(examples, ensure_ascii=False, indent=2),
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
            "config": config,
            "graph_mean": graph_mean.tolist(),
            "graph_std": graph_std.tolist(),
            "graph_indices": graph_indices,
            "feature_names_all": feature_names_all,
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
) -> tuple[GraphFusionCRNNCTC, dict[str, Any], np.ndarray, np.ndarray, list[int]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = GraphFusionCRNNCTC(
        num_classes=cfg["num_classes"],
        graph_dim=cfg["graph_dim"],
        hidden_size=cfg["hidden_size"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
        blank_index=cfg["blank_index"],
        blank_bias_init=cfg["blank_bias_init"],
        height_bins=cfg["height_bins"],
        feature_size=cfg["feature_size"],
        graph_hidden_dim=cfg.get("graph_hidden_dim", cfg["graph_embed_dim"]),
        graph_embed_dim=cfg["graph_embed_dim"],
        graph_dropout=cfg["graph_dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    graph_mean = np.asarray(ckpt["graph_mean"], dtype=np.float32)
    graph_std = np.asarray(ckpt["graph_std"], dtype=np.float32)
    graph_indices = list(ckpt["graph_indices"])

    return model, ckpt, graph_mean, graph_std, graph_indices


def cmd_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)

    checkpoint = Path(args.checkpoint)
    model, ckpt, graph_mean, graph_std, graph_indices = load_checkpoint_model(checkpoint, device)

    vocab_path = args.vocab or ckpt["config"]["vocab"]
    vocab = SimpleCTCVocab.from_json(vocab_path)
    blank_penalty = (
        float(args.blank_logit_penalty)
        if args.blank_logit_penalty is not None
        else float(ckpt["blank_logit_penalty"])
    )

    _, _, eval_loader = build_loaders(
        train_manifest=None,
        val_manifest=None,
        eval_manifest=Path(args.manifest),
        vocab=vocab,
        graph_indices=graph_indices,
        graph_mean=graph_mean,
        graph_std=graph_std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_zero_graph=args.zero_graph,
    )

    result = evaluate(
        model=model,
        loader=eval_loader,
        vocab=vocab,
        device=device,
        blank_penalty=blank_penalty,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n": result["metrics"]["n"],
        "cer": result["metrics"]["cer"],
        "wer": result["metrics"]["wer"],
        "exact": result["metrics"]["exact"],
        "pred_len_mean": result["metrics"]["pred_len_mean"],
        "pred_empty_ratio": result["metrics"]["pred_empty_ratio"],
        "argmax_blank_ratio": result["metrics"]["argmax_blank_ratio"],
        "grouped": result["metrics"].get("grouped", {}),
        "out_dir": str(out_dir),
        "blank_logit_penalty": blank_penalty,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_cer": ckpt["val"]["cer"],
        "graph_feature_dim": ckpt["config"]["graph_dim"],
        "graph_feature_names": ckpt["config"]["graph_feature_names"],
        "zero_graph": args.zero_graph,
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
    train.add_argument("--graph_dropout", type=float, default=0.1)
    train.add_argument("--hidden_size", type=int, default=256)
    train.add_argument("--lstm_layers", type=int, default=2)
    train.add_argument("--blank_bias_init", type=float, default=-1.0)
    train.add_argument("--blank_logit_penalty_start", type=float, default=-2.0)
    train.add_argument("--blank_logit_penalty_end", type=float, default=-0.4)
    train.add_argument("--height_bins", type=int, default=4)
    train.add_argument("--feature_size", type=int, default=256)
    train.add_argument("--graph_hidden_dim", type=int, default=64)
    train.add_argument("--graph_embed_dim", type=int, default=128)
    train.add_argument("--drop_graph_features", default="text_len")
    train.add_argument("--log_every", type=int, default=50)
    train.add_argument("--seed", type=int, default=49)
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--resume", default=None)
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--out_dir", required=True)
    ev.add_argument("--vocab", default=None)
    ev.add_argument("--batch_size", type=int, default=64)
    ev.add_argument("--num_workers", type=int, default=2)
    ev.add_argument("--blank_logit_penalty", type=float, default=None)
    ev.add_argument("--zero_graph", action="store_true")
    ev.add_argument("--cpu", action="store_true")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
