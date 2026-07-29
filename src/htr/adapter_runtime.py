from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from src.htr.ctc_decode import greedy_decode
from src.htr.metrics import edit_distance
from src.htr.model import CRNNCTC
from src.htr.model_hi_csg_r_adapter import CRNNCTCHICSGRAdapter
from src.htr.vocab import CTCVocab
from torch import nn
from torch.utils.data import Sampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EpochWidthBatchSampler(Sampler[list[int]]):
    """Width-sorted batches with deterministic epoch-level batch shuffling."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        batch_size: int,
        *,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        indices = list(range(len(rows)))
        indices.sort(key=lambda index: _row_width(rows[index]))
        self.batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self.batches)


def _row_width(row: dict[str, Any]) -> int:
    info = row.get("image_info")
    if isinstance(info, dict) and info.get("width") is not None:
        return int(info["width"])
    return int(row.get("width", 0) or 0)


def apply_blank_logit_penalty(
    log_probs: torch.Tensor,
    blank_index: int,
    penalty: float,
) -> torch.Tensor:
    if penalty == 0.0:
        return log_probs
    adjusted = log_probs.clone()
    adjusted[..., blank_index] += float(penalty)
    return adjusted - torch.logsumexp(adjusted, dim=-1, keepdim=True)


def forward_batch(
    model: CRNNCTC | CRNNCTCHICSGRAdapter,
    batch: dict[str, Any],
    device: torch.device,
    *,
    mode: str,
    blank_logit_penalty: float,
    graph_enabled: bool = True,
) -> dict[str, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    widths = batch["widths"].to(device, non_blocking=True)
    if mode == "m0_ft":
        log_probs = model(images)
        assert isinstance(model, CRNNCTC)
        return {
            "log_probs": apply_blank_logit_penalty(
                log_probs.float(), model.blank_index, blank_logit_penalty
            ),
            "output_lengths": model.output_lengths(widths),
        }
    assert isinstance(model, CRNNCTCHICSGRAdapter)
    output = model(
        images,
        widths,
        batch["graph_features"].to(device, non_blocking=True),
        batch["graph_quality"].to(device, non_blocking=True),
        batch["graph_mask"].to(device, non_blocking=True),
        graph_enabled=graph_enabled,
    )
    output["log_probs"] = apply_blank_logit_penalty(
        output["log_probs"].float(), model.blank_index, blank_logit_penalty
    )
    output["graph_aux_log_probs"] = apply_blank_logit_penalty(
        output["graph_aux_log_probs"].float(), model.blank_index, blank_logit_penalty
    )
    return output


def sample_metric_row(
    sample_id: str,
    graph_sample_id: str,
    dataset: str,
    target: str,
    prediction: str,
    *,
    level: Any = None,
    category: Any = None,
) -> dict[str, Any]:
    char_edits = edit_distance(prediction, target)
    word_edits = edit_distance(prediction.split(), target.split())
    return {
        "sample_id": sample_id,
        "graph_sample_id": graph_sample_id,
        "dataset": dataset,
        "level": level,
        "category": category,
        "target": target,
        "prediction": prediction,
        "char_edits": char_edits,
        "target_chars": len(target),
        "sample_cer": char_edits / max(len(target), 1),
        "word_edits": word_edits,
        "target_words": len(target.split()),
        "sample_wer": word_edits / max(len(target.split()), 1),
        "exact": prediction == target,
        "prediction_length": len(prediction),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    char_edits = sum(int(row["char_edits"]) for row in rows)
    target_chars = sum(int(row["target_chars"]) for row in rows)
    word_edits = sum(int(row["word_edits"]) for row in rows)
    target_words = sum(int(row["target_words"]) for row in rows)
    return {
        "samples": len(rows),
        "cer": char_edits / max(target_chars, 1),
        "macro_cer": float(np.mean([row["sample_cer"] for row in rows])) if rows else 0.0,
        "wer": word_edits / max(target_words, 1),
        "exact": float(np.mean([row["exact"] for row in rows])) if rows else 0.0,
        "char_edits": char_edits,
        "target_chars": target_chars,
        "word_edits": word_edits,
        "target_words": target_words,
        "pred_len_mean": (
            float(np.mean([row["prediction_length"] for row in rows])) if rows else 0.0
        ),
    }


@torch.no_grad()
def evaluate_loader(
    model: CRNNCTC | CRNNCTCHICSGRAdapter,
    loader: Any,
    vocab: CTCVocab,
    device: torch.device,
    *,
    mode: str,
    criterion: nn.CTCLoss,
    blank_logit_penalty: float,
    graph_enabled: bool = True,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    aux_losses: list[float] = []
    blank_count = 0
    valid_steps = 0
    gates: list[np.ndarray] = []
    gate_empty: list[np.ndarray] = []
    gate_nonempty: list[np.ndarray] = []
    gate_by_dataset: defaultdict[str, list[float]] = defaultdict(list)
    aux_char_edits = 0
    aux_target_chars = 0
    had_auxiliary_output = False

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        output = forward_batch(
            model,
            batch,
            device,
            mode=mode,
            blank_logit_penalty=blank_logit_penalty,
            graph_enabled=graph_enabled,
        )
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        lengths = output["output_lengths"]
        losses.append(
            float(criterion(output["log_probs"], targets, lengths, target_lengths).item())
        )
        if "graph_aux_log_probs" in output:
            aux_losses.append(
                float(
                    criterion(
                        output["graph_aux_log_probs"],
                        targets,
                        lengths,
                        target_lengths,
                    ).item()
                )
            )
        ids = output["log_probs"].argmax(dim=-1).cpu()
        for batch_index, length in enumerate(lengths.cpu().tolist()):
            values = ids[: int(length), batch_index]
            blank_count += int((values == vocab.blank_index).sum())
            valid_steps += int(length)
        predictions = greedy_decode(output["log_probs"], lengths, vocab)
        aux_predictions = (
            greedy_decode(output["graph_aux_log_probs"], lengths, vocab)
            if "graph_aux_log_probs" in output
            else None
        )
        had_auxiliary_output = aux_predictions is not None
        row_start = len(rows)
        for index, prediction in enumerate(predictions):
            rows.append(
                sample_metric_row(
                    batch["sample_ids"][index],
                    batch["graph_sample_ids"][index],
                    batch["datasets"][index],
                    batch["texts"][index],
                    prediction,
                    level=batch["levels"][index],
                    category=batch["categories"][index],
                )
            )
            if aux_predictions is not None:
                rows[-1]["graph_aux_prediction"] = aux_predictions[index]
                aux_char_edits += edit_distance(aux_predictions[index], batch["texts"][index])
                aux_target_chars += len(batch["texts"][index])
            length = int(lengths[index].item())
            raw_graph = batch["graph_raw_features"][index, :length]
            rows[-1].update(
                {
                    "short_branch_fraction_mean": float(raw_graph[:, 15].mean().item()),
                    "ambiguous_edge_fraction_mean": float(raw_graph[:, 17].mean().item()),
                    "graph_occupancy_mean": float(raw_graph[:, 18].mean().item()),
                    "warning_density_mean": float(raw_graph[:, 19].mean().item()),
                }
            )

        if "gate" in output:
            gate = output["gate"].detach().cpu().numpy()[..., 0]
            mask = batch["graph_mask"].numpy()
            occupancy = batch["graph_quality"].numpy()[..., 1] > 0
            gates.append(gate[mask])
            gate_empty.append(gate[mask & ~occupancy])
            gate_nonempty.append(gate[mask & occupancy])
            for index, dataset in enumerate(batch["datasets"]):
                gate_by_dataset[str(dataset)].extend(gate[index][mask[index]].tolist())
                row = rows[row_start + index]
                length = int(lengths[index].item())
                row["gate_curve"] = gate[index, :length].tolist()

    summary = summarize_rows(rows)
    summary.update(
        {
            "ctc_loss": float(np.mean(losses)) if losses else 0.0,
            "graph_aux_ctc_loss": float(np.mean(aux_losses)) if aux_losses else None,
            "graph_aux_cer": (
                aux_char_edits / max(aux_target_chars, 1)
                if had_auxiliary_output
                else None
            ),
            "blank_ratio": blank_count / max(valid_steps, 1),
        }
    )
    if gates:
        all_gates = np.concatenate(gates)
        summary["gate"] = {
            "mean": float(np.mean(all_gates)),
            "std": float(np.std(all_gates)),
            "p10": float(np.quantile(all_gates, 0.10)),
            "p50": float(np.quantile(all_gates, 0.50)),
            "p90": float(np.quantile(all_gates, 0.90)),
            "empty_bin": _concat_mean(gate_empty),
            "nonempty_bin": _concat_mean(gate_nonempty),
            "by_dataset": {
                key: float(np.mean(values)) for key, values in sorted(gate_by_dataset.items())
            },
        }
    return summary, rows


def _concat_mean(values: list[np.ndarray]) -> float | None:
    nonempty = [value for value in values if value.size]
    return float(np.mean(np.concatenate(nonempty))) if nonempty else None


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
