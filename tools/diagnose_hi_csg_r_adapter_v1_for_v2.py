from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.adapter_runtime import (
    EpochWidthBatchSampler,
    apply_blank_logit_penalty,
    sample_metric_row,
    summarize_rows,
)
from src.htr.ctc_decode import greedy_decode
from src.htr.dataset_adapter import HICSGRAdapterDataset, collate_adapter_batch
from src.htr.model import CRNNCTC
from src.htr.model_hi_csg_r_adapter import CRNNCTCHICSGRAdapter
from src.htr.vocab import CTCVocab
from src.htr.xaligned_hi_csg_r import XAlignedFeatureNormalizer

from tools.evaluate_crnn_ctc_hi_csg_r_adapter_v1 import load_model

PENALTIES = (-0.8, -0.6, -0.5, -0.4, -0.3, -0.2, 0.0)
GRAPH_SCALES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)


def add_predictions(
    target: list[dict[str, Any]],
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    batch: dict[str, Any],
    vocab: CTCVocab,
) -> None:
    predictions = greedy_decode(log_probs, lengths, vocab)
    for index, prediction in enumerate(predictions):
        target.append(
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


def domain_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    return {name: summarize_rows(values) for name, values in sorted(grouped.items())}


def graph_forward(
    model: CRNNCTCHICSGRAdapter,
    batch: dict[str, Any],
    device: torch.device,
    *,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    features = batch["graph_features"].to(device, non_blocking=True)
    quality = batch["graph_quality"].to(device, non_blocking=True)
    visual = model.visual_sequence(images)
    graph = model.graph_adapter(features, mask)
    gate = model.graph_gate(visual, graph, quality, mask)
    residual = gate * graph
    return visual, graph, gate, residual


def logits_for_residual(
    model: CRNNCTCHICSGRAdapter,
    visual: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    fused = model._baseline_preserving_fusion(visual, float(scale) * residual)
    sequence, _ = model.rnn(fused.transpose(0, 1))
    return model.classifier(sequence).log_softmax(dim=-1)


@torch.no_grad()
def evaluate_preflight(
    m0: CRNNCTC,
    m3: CRNNCTCHICSGRAdapter,
    loader: DataLoader,
    vocab: CTCVocab,
    device: torch.device,
) -> dict[str, Any]:
    m0.eval()
    m3.eval()
    penalty_rows = {
        (name, penalty): []
        for name in ("m0_ft", "m3")
        for penalty in PENALTIES
    }
    scale_rows = {scale: [] for scale in GRAPH_SCALES}
    strict_rows: list[dict[str, Any]] = []
    gate_original: list[np.ndarray] = []
    gate_strict: list[np.ndarray] = []
    original_empty_norm: list[np.ndarray] = []
    original_nonempty_norm: list[np.ndarray] = []
    strict_empty_norm: list[np.ndarray] = []
    strict_nonempty_norm: list[np.ndarray] = []

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        widths = batch["widths"].to(device, non_blocking=True)
        lengths = m0.output_lengths(widths)
        m0_log_probs = m0(images)
        valid = batch["graph_mask"].to(device, non_blocking=True)
        raw = batch["graph_raw_features"].to(device, non_blocking=True)
        nonempty = (
            (raw[..., 0] > 0)
            | (raw[..., 1] > 0)
            | (raw[..., 2] > 0)
            | (raw[..., 18] > 0)
        ) & valid
        visual, _, gate, residual = graph_forward(m3, batch, device, mask=valid)
        original_log_probs = logits_for_residual(m3, visual, residual, 1.0)

        for penalty in PENALTIES:
            add_predictions(
                penalty_rows[("m0_ft", penalty)],
                apply_blank_logit_penalty(m0_log_probs, vocab.blank_index, penalty),
                lengths,
                batch,
                vocab,
            )
            add_predictions(
                penalty_rows[("m3", penalty)],
                apply_blank_logit_penalty(
                    original_log_probs,
                    vocab.blank_index,
                    penalty,
                ),
                lengths,
                batch,
                vocab,
            )
        for scale in GRAPH_SCALES:
            log_probs = logits_for_residual(m3, visual, residual, scale)
            add_predictions(
                scale_rows[scale],
                apply_blank_logit_penalty(log_probs, vocab.blank_index, -0.4),
                lengths,
                batch,
                vocab,
            )

        strict_visual, _, strict_gate, strict_residual = graph_forward(
            m3,
            batch,
            device,
            mask=nonempty,
        )
        strict_log_probs = logits_for_residual(
            m3,
            strict_visual,
            strict_residual,
            1.0,
        )
        add_predictions(
            strict_rows,
            apply_blank_logit_penalty(strict_log_probs, vocab.blank_index, -0.4),
            lengths,
            batch,
            vocab,
        )
        gate_original.append(gate[valid].detach().cpu().numpy())
        gate_strict.append(strict_gate[valid].detach().cpu().numpy())
        residual_norm = residual.norm(dim=-1)
        strict_norm = strict_residual.norm(dim=-1)
        original_empty_norm.append(residual_norm[valid & ~nonempty].cpu().numpy())
        original_nonempty_norm.append(residual_norm[nonempty].cpu().numpy())
        strict_empty_norm.append(strict_norm[valid & ~nonempty].cpu().numpy())
        strict_nonempty_norm.append(strict_norm[nonempty].cpu().numpy())

    penalty = {
        name: {
            str(value): summarize_rows(penalty_rows[(name, value)])
            for value in PENALTIES
        }
        for name in ("m0_ft", "m3")
    }
    scale = {
        str(value): {
            **summarize_rows(scale_rows[value]),
            "domain": domain_summary(scale_rows[value]),
        }
        for value in GRAPH_SCALES
    }
    strict = {
        "summary": summarize_rows(strict_rows),
        "domain": domain_summary(strict_rows),
        "gate": {
            "original_mean": float(np.mean(np.concatenate(gate_original))),
            "original_std": float(np.std(np.concatenate(gate_original))),
            "strict_mean": float(np.mean(np.concatenate(gate_strict))),
            "strict_std": float(np.std(np.concatenate(gate_strict))),
        },
        "contribution_norm": {
            "original_empty": concat_mean(original_empty_norm),
            "original_nonempty": concat_mean(original_nonempty_norm),
            "strict_empty": concat_mean(strict_empty_norm),
            "strict_nonempty": concat_mean(strict_nonempty_norm),
        },
    }
    return {"blank_penalty": penalty, "graph_scale": scale, "strict_mask": strict}


def concat_mean(values: list[np.ndarray]) -> float:
    nonempty = [value for value in values if value.size]
    return float(np.mean(np.concatenate(nonempty))) if nonempty else 0.0


@torch.no_grad()
def evaluate_strict_shuffle(
    model: CRNNCTCHICSGRAdapter,
    loader: DataLoader,
    vocab: CTCVocab,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        widths = batch["widths"].to(device, non_blocking=True)
        valid = batch["graph_mask"].to(device, non_blocking=True)
        raw = batch["graph_raw_features"].to(device, non_blocking=True)
        nonempty = (
            (raw[..., 0] > 0)
            | (raw[..., 1] > 0)
            | (raw[..., 2] > 0)
            | (raw[..., 18] > 0)
        ) & valid
        visual, _, _, residual = graph_forward(model, batch, device, mask=nonempty)
        log_probs = logits_for_residual(model, visual, residual, 1.0)
        add_predictions(
            rows,
            apply_blank_logit_penalty(log_probs, vocab.blank_index, -0.4),
            model.output_lengths(widths),
            batch,
            vocab,
        )
    return {"summary": summarize_rows(rows), "domain": domain_summary(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0_checkpoint", required=True)
    parser.add_argument("--m3_checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shuffle_map", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = CTCVocab.from_path(args.vocab)
    normalizer = XAlignedFeatureNormalizer.from_path(args.normalizer)
    m0, _ = load_model(args.m0_checkpoint, vocab)
    m3, _ = load_model(args.m3_checkpoint, vocab)
    if not isinstance(m0, CRNNCTC) or not isinstance(m3, CRNNCTCHICSGRAdapter):
        raise TypeError("Preflight requires an M0-FT checkpoint and an M3 checkpoint")
    dataset = HICSGRAdapterDataset(
        args.manifest,
        vocab,
        normalizer=normalizer,
        mode="m3_full",
    )
    shuffled = HICSGRAdapterDataset(
        args.manifest,
        vocab,
        normalizer=normalizer,
        mode="m3_full",
        shuffle_map=args.shuffle_map,
    )
    sampler = EpochWidthBatchSampler(
        dataset.rows,
        args.batch_size,
        seed=42,
        shuffle=False,
    )
    shuffled_sampler = EpochWidthBatchSampler(
        shuffled.rows,
        args.batch_size,
        seed=42,
        shuffle=False,
    )
    loader_options = {
        "collate_fn": collate_adapter_batch,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    loader = DataLoader(dataset, batch_sampler=sampler, **loader_options)
    shuffled_loader = DataLoader(
        shuffled,
        batch_sampler=shuffled_sampler,
        **loader_options,
    )
    m0.to(device)
    m3.to(device)
    diagnostics = evaluate_preflight(m0, m3, loader, vocab, device)
    diagnostics["strict_mask"]["shuffle"] = evaluate_strict_shuffle(
        m3,
        shuffled_loader,
        vocab,
        device,
    )

    penalty_results = diagnostics["blank_penalty"]["m0_ft"]
    selected_penalty = min(
        PENALTIES,
        key=lambda value: (
            penalty_results[str(value)]["cer"],
            abs(value + 0.4),
        ),
    )
    scale_results = diagnostics["graph_scale"]
    original_cer = scale_results["1.0"]["cer"]
    best_scale = min(
        GRAPH_SCALES,
        key=lambda value: (scale_results[str(value)]["cer"], abs(value - 0.25)),
    )
    d2_gain = original_cer - scale_results[str(best_scale)]["cer"]
    strict_cer = diagnostics["strict_mask"]["summary"]["cer"]
    d3_gain = original_cer - strict_cer
    strict_shuffle_cer = diagnostics["strict_mask"]["shuffle"]["summary"]["cer"]
    correct_better_shuffle = strict_cer < strict_shuffle_cer
    branch_active = diagnostics["strict_mask"]["gate"]["original_std"] > 1e-6
    preflight_effect = max(d2_gain, d3_gain)
    status = (
        "CONTINUE_FULL"
        if correct_better_shuffle and branch_active and preflight_effect >= 0.0005
        else "CONTINUE_V2_1_ONLY"
        if correct_better_shuffle and branch_active
        else "STOP"
    )
    decision = {
        "status": status,
        "selected_blank_logit_penalty": selected_penalty,
        "selected_alpha_max": 0.25,
        "best_v1_graph_scale": best_scale,
        "d2_absolute_cer_gain": d2_gain,
        "d3_absolute_cer_gain": d3_gain,
        "correct_better_strict_shuffle": correct_better_shuffle,
        "branch_technically_active": branch_active,
        "allow_v2_2": status == "CONTINUE_FULL",
    }
    report = {"decision": decision, **diagnostics}
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "blank_penalty_sweep.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("model", "penalty", "cer", "wer", "exact"),
        )
        writer.writeheader()
        for name in ("m0_ft", "m3"):
            for penalty in PENALTIES:
                summary = diagnostics["blank_penalty"][name][str(penalty)]
                writer.writerow(
                    {
                        "model": name,
                        "penalty": penalty,
                        "cer": summary["cer"],
                        "wer": summary["wer"],
                        "exact": summary["exact"],
                    }
                )
    markdown = [
        "# HI-CSG-R adapter v2 preflight",
        "",
        f"**Decision:** `{status}`",
        "",
        "## Frozen choices",
        "",
        f"- blank logit penalty: `{selected_penalty}`",
        "- alpha_max: `0.25`",
        f"- best v1 graph scale: `{best_scale}`",
        f"- D2 CER gain: `{d2_gain:.6f}`",
        f"- D3 CER gain: `{d3_gain:.6f}`",
        f"- strict correct CER: `{strict_cer:.6f}`",
        f"- strict shuffle CER: `{strict_shuffle_cer:.6f}`",
        "",
        "The v1 conclusion remains unchanged. These diagnostics only determine "
        "whether the separately versioned v2 protocol may proceed.",
    ]
    (output / "preflight_report.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    (output / "blank_penalty_sweep.md").write_text(
        "# Blank penalty sweep\n\n"
        + "\n".join(
            f"- {name} `{penalty}`: CER "
            f"`{diagnostics['blank_penalty'][name][str(penalty)]['cer']:.6f}`"
            for name in ("m0_ft", "m3")
            for penalty in PENALTIES
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if status == "STOP":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

