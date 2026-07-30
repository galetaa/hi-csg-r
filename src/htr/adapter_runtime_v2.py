from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from src.htr.adapter_runtime import (
    apply_blank_logit_penalty,
    sample_metric_row,
    summarize_rows,
)
from src.htr.ctc_decode import greedy_decode
from src.htr.losses_adapter_v2 import baseline_preservation_kl
from src.htr.metrics import edit_distance
from src.htr.model_hi_csg_r_late_correction_v2 import (
    HI_CSG_R_LateCorrectionCRNNCTC,
)
from src.htr.vocab import CTCVocab
from torch import nn


def forward_v2_batch(
    model: HI_CSG_R_LateCorrectionCRNNCTC,
    batch: dict[str, Any],
    device: torch.device,
    *,
    blank_logit_penalty: float,
    zero_graph: bool = False,
) -> dict[str, torch.Tensor]:
    graph = batch["normalized_graph_features"].to(device, non_blocking=True)
    risk = batch["structural_risk_raw"].to(device, non_blocking=True)
    time_mask = batch["time_mask"].to(device, non_blocking=True)
    nonempty = batch["nonempty_graph_mask"].to(device, non_blocking=True)
    if zero_graph:
        graph = torch.zeros_like(graph)
        risk = torch.zeros_like(risk)
        nonempty = torch.zeros_like(nonempty)
    output = model(
        batch["images"].to(device, non_blocking=True),
        batch["widths"].to(device, non_blocking=True),
        graph,
        risk,
        time_mask,
        nonempty,
    )
    base = output["base_logits"].transpose(0, 1).float().log_softmax(dim=-1)
    final = output["final_logits"].transpose(0, 1).float().log_softmax(dim=-1)
    aux = output["aux_logits"].transpose(0, 1).float().log_softmax(dim=-1)
    output["base_log_probs"] = apply_blank_logit_penalty(
        base,
        model.blank_index,
        blank_logit_penalty,
    )
    output["final_log_probs"] = apply_blank_logit_penalty(
        final,
        model.blank_index,
        blank_logit_penalty,
    )
    output["aux_log_probs"] = apply_blank_logit_penalty(
        aux,
        model.blank_index,
        blank_logit_penalty,
    )
    return output


def _summary_by(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: summarize_rows(values)
        for key, values in sorted(grouped.items())
    }


def _stats(values: list[np.ndarray]) -> dict[str, float | None]:
    arrays = [value.reshape(-1) for value in values if value.size]
    if not arrays:
        return {
            "mean": None,
            "std": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    merged = np.concatenate(arrays)
    return {
        "mean": float(merged.mean()),
        "std": float(merged.std()),
        "p10": float(np.quantile(merged, 0.10)),
        "p50": float(np.quantile(merged, 0.50)),
        "p90": float(np.quantile(merged, 0.90)),
        "max": float(merged.max()),
    }


@torch.no_grad()
def evaluate_v2_loader(
    model: HI_CSG_R_LateCorrectionCRNNCTC,
    loader: Any,
    vocab: CTCVocab,
    device: torch.device,
    *,
    criterion: nn.CTCLoss,
    blank_logit_penalty: float,
    preservation_temperature: float = 1.5,
    zero_graph: bool = False,
    max_batches: int | None = None,
    frame_output: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    final_losses: list[float] = []
    base_losses: list[float] = []
    aux_losses: list[float] = []
    preservation_losses: list[float] = []
    gate_values: list[np.ndarray] = []
    gate_nonempty: list[np.ndarray] = []
    uncertainty_values: list[np.ndarray] = []
    uncertainty_blank: list[np.ndarray] = []
    uncertainty_nonblank: list[np.ndarray] = []
    risk_values: list[np.ndarray] = []
    correction_norms: list[np.ndarray] = []
    correction_abs: list[np.ndarray] = []
    base_norms: list[np.ndarray] = []
    empty_corrections: list[np.ndarray] = []
    gate_by_domain: defaultdict[str, list[float]] = defaultdict(list)
    frame_store: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    changed_frames = 0
    valid_frames = 0
    intervention_frames = 0
    strong_intervention_frames = 0

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        output = forward_v2_batch(
            model,
            batch,
            device,
            blank_logit_penalty=blank_logit_penalty,
            zero_graph=zero_graph,
        )
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        lengths = output["output_lengths"]
        final_losses.append(
            float(
                criterion(
                    output["final_log_probs"],
                    targets,
                    lengths,
                    target_lengths,
                ).item()
            )
        )
        base_losses.append(
            float(
                criterion(
                    output["base_log_probs"],
                    targets,
                    lengths,
                    target_lengths,
                ).item()
            )
        )
        aux_losses.append(
            float(
                criterion(
                    output["aux_log_probs"],
                    targets,
                    lengths,
                    target_lengths,
                ).item()
            )
        )
        preservation_losses.append(
            float(
                baseline_preservation_kl(
                    output["base_logits"],
                    output["final_logits"],
                    output["visual_uncertainty"],
                    batch["time_mask"].to(device),
                    temperature=preservation_temperature,
                ).item()
            )
        )
        final_predictions = greedy_decode(
            output["final_log_probs"],
            lengths,
            vocab,
        )
        base_predictions = greedy_decode(
            output["base_log_probs"],
            lengths,
            vocab,
        )
        gate = output["gate"][..., 0].cpu().numpy()
        uncertainty = output["visual_uncertainty"][..., 0].cpu().numpy()
        risk = output["risk"][..., 0].cpu().numpy()
        correction = output["correction_logits"]
        correction_norm = correction.norm(dim=-1).cpu().numpy()
        correction_absolute = correction.abs().amax(dim=-1).cpu().numpy()
        base_norm = output["base_logits"].norm(dim=-1).cpu().numpy()
        time_mask = batch["time_mask"].numpy()
        nonempty = batch["nonempty_graph_mask"].numpy() & time_mask
        empty = time_mask & ~nonempty
        base_argmax = output["base_logits"].argmax(dim=-1).cpu().numpy()
        final_argmax = output["final_logits"].argmax(dim=-1).cpu().numpy()
        changed_frames += int(((base_argmax != final_argmax) & time_mask).sum())
        valid_frames += int(time_mask.sum())
        intervention_frames += int(((gate > 0.05) & time_mask).sum())
        strong_intervention_frames += int(((gate > 0.15) & time_mask).sum())
        gate_values.append(gate[time_mask])
        gate_nonempty.append(gate[nonempty])
        uncertainty_values.append(uncertainty[time_mask])
        uncertainty_blank.append(
            uncertainty[time_mask & (base_argmax == vocab.blank_index)]
        )
        uncertainty_nonblank.append(
            uncertainty[time_mask & (base_argmax != vocab.blank_index)]
        )
        risk_values.append(risk[time_mask])
        correction_norms.append(correction_norm[time_mask])
        correction_abs.append(correction_absolute[time_mask])
        base_norms.append(base_norm[time_mask])
        empty_corrections.append(correction_absolute[empty])
        for index, domain in enumerate(batch["core_domains"]):
            gate_by_domain[str(domain)].extend(gate[index][time_mask[index]].tolist())
        if frame_output:
            frame_store["gate"].append(gate[time_mask])
            frame_store["uncertainty"].append(uncertainty[time_mask])
            frame_store["risk"].append(risk[time_mask])
            frame_store["correction_norm"].append(correction_norm[time_mask])
            frame_store["base_norm"].append(base_norm[time_mask])
            frame_store["nonempty"].append(nonempty[time_mask].astype(np.uint8))
            frame_store["top1_changed"].append(
                (base_argmax[time_mask] != final_argmax[time_mask]).astype(np.uint8)
            )

        raw_graph = batch["raw_graph_features"].numpy()
        for index, (prediction, base_prediction) in enumerate(
            zip(final_predictions, base_predictions, strict=True)
        ):
            row = sample_metric_row(
                batch["sample_ids"][index],
                batch["graph_sample_ids"][index],
                batch["datasets"][index],
                batch["texts"][index],
                prediction,
                level=batch["levels"][index],
                category=batch["categories"][index],
            )
            length = int(lengths[index].item())
            target = batch["texts"][index]
            base_edits = edit_distance(base_prediction, target)
            row.update(
                {
                    "core_domain": batch["core_domains"][index],
                    "baseline_prediction": base_prediction,
                    "baseline_char_edits": base_edits,
                    "baseline_exact": base_prediction == target,
                    "prediction_changed": prediction != base_prediction,
                    "edit_delta_vs_baseline": row["char_edits"] - base_edits,
                    "visual_uncertainty_mean": float(
                        uncertainty[index, :length].mean()
                    ),
                    "gate_mean": float(gate[index, :length].mean()),
                    "correction_norm_mean": float(
                        correction_norm[index, :length].mean()
                    ),
                    "risk_mean": float(risk[index, :length].mean()),
                    "graph_occupancy_mean": float(
                        raw_graph[index, :length, 18].mean()
                    ),
                    "short_branch_fraction_mean": float(
                        raw_graph[index, :length, 15].mean()
                    ),
                    "warning_density_mean": float(
                        raw_graph[index, :length, 19].mean()
                    ),
                    "component_count_norm_mean": float(
                        raw_graph[index, :length, 14].mean()
                    ),
                }
            )
            rows.append(row)

    summary = summarize_rows(rows)
    base_rows = [
        {
            **row,
            "prediction": row["baseline_prediction"],
            "char_edits": row["baseline_char_edits"],
            "sample_cer": row["baseline_char_edits"] / max(row["target_chars"], 1),
            "word_edits": edit_distance(
                row["baseline_prediction"].split(),
                row["target"].split(),
            ),
            "sample_wer": edit_distance(
                row["baseline_prediction"].split(),
                row["target"].split(),
            )
            / max(row["target_words"], 1),
            "exact": row["baseline_exact"],
            "prediction_length": len(row["baseline_prediction"]),
        }
        for row in rows
    ]
    changed = [row for row in rows if row["prediction_changed"]]
    unchanged = [row for row in rows if not row["prediction_changed"]]
    improved = [row for row in changed if row["edit_delta_vs_baseline"] < 0]
    summary.update(
        {
            "baseline": summarize_rows(base_rows),
            "ctc_loss": float(np.mean(final_losses)) if final_losses else 0.0,
            "baseline_ctc_loss": float(np.mean(base_losses)) if base_losses else 0.0,
            "auxiliary_ctc_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "preservation_kl": (
                float(np.mean(preservation_losses)) if preservation_losses else 0.0
            ),
            "alpha": float(model.alpha().item()),
            "gate": {
                **_stats(gate_values),
                "empty": 0.0,
                "nonempty": _stats(gate_nonempty)["mean"],
                "by_domain": {
                    key: float(np.mean(values))
                    for key, values in sorted(gate_by_domain.items())
                },
            },
            "uncertainty": {
                **_stats(uncertainty_values),
                "blank_argmax": _stats(uncertainty_blank)["mean"],
                "nonblank_argmax": _stats(uncertainty_nonblank)["mean"],
            },
            "risk": _stats(risk_values),
            "correction": {
                "absolute": _stats(correction_abs),
                "l2": _stats(correction_norms),
                "base_l2": _stats(base_norms),
                "l2_ratio": (
                    (_stats(correction_norms)["mean"] or 0.0)
                    / max(_stats(base_norms)["mean"] or 0.0, 1e-12)
                ),
                "empty_max": _stats(empty_corrections)["max"] or 0.0,
            },
            "intervention": {
                "top1_frame_change_rate": changed_frames / max(valid_frames, 1),
                "prediction_change_rate": len(changed) / max(len(rows), 1),
                "gate_gt_005_rate": intervention_frames / max(valid_frames, 1),
                "gate_gt_015_rate": strong_intervention_frames / max(valid_frames, 1),
                "changed_summary": summarize_rows(changed),
                "unchanged_summary": summarize_rows(unchanged),
                "precision": len(improved) / max(len(changed), 1),
                "improved_samples": len(improved),
                "hurt_samples": sum(
                    row["edit_delta_vs_baseline"] > 0 for row in changed
                ),
            },
        }
    )
    if frame_output:
        destination = Path(frame_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            **{
                key: np.concatenate(values) if values else np.asarray([])
                for key, values in frame_store.items()
            },
        )
    grouped = {
        "domain": _summary_by(rows, "core_domain"),
        "dataset": _summary_by(rows, "dataset"),
        "baseline_domain": _summary_by(base_rows, "core_domain"),
        "baseline_dataset": _summary_by(base_rows, "dataset"),
    }
    uncertainty_correct = [
        np.asarray([row["visual_uncertainty_mean"]])
        for row in rows
        if row["baseline_exact"]
    ]
    uncertainty_error = [
        np.asarray([row["visual_uncertainty_mean"]])
        for row in rows
        if not row["baseline_exact"]
    ]
    summary["uncertainty"]["baseline_correct_samples"] = _stats(
        uncertainty_correct
    )["mean"]
    summary["uncertainty"]["baseline_error_samples"] = _stats(
        uncertainty_error
    )["mean"]
    return {**summary, "grouped": grouped}, rows
