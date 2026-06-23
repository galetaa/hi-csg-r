from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.htr.dataset import HTRDataset, collate_htr_batch
from src.htr.metrics import cer, exact_match, wer
from src.htr.vocab import CTCVocab
from tools.evaluate_crnn_ctc import (
    apply_blank_logit_penalty,
    load_model_from_checkpoint,
)


MODELS = {
    "baseline": {
        "checkpoint": "outputs/htr_graph_v1/tri10k_image_only_v1/best.pt",
        "vocab": "data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/vocab.json",
    },
    "plus_5k_context": {
        "checkpoint": "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_5k_context_v1/best.pt",
        "vocab": "data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_5k_context_v1/vocab.json",
    },
    "plus_10k_context": {
        "checkpoint": "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1/best.pt",
        "vocab": "data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/vocab.json",
    },
}

FINAL_BLANK_LOGIT_PENALTIES = {
    # Baseline final eval selected this penalty in the validation sweep.
    "baseline": -0.8,
}

CONFIDENCE_FEATURES = [
    "mean_max_prob",
    "mean_entropy",
    "argmax_blank_ratio",
    "decoded_len_per_frame",
    "avg_decoded_char_confidence",
    "mean_frame_margin",
    "sequence_score_proxy",
]

GRAPH_FEATURES = [
    "fg_fraction",
    "skel_fraction",
    "cc_count",
    "dir_h_frac",
    "stroke_width_mean",
    "aspect_ratio",
    "bbox_area_frac",
    "branchpoint_per_100_skel",
    "ruling_response_mean",
]

DATASETS = [
    "hkr_words",
    "cyrillic_handwriting",
    "school_notebooks_clean",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def dataset_from_sample_id(sample_id: str) -> str:
    if sample_id.startswith("hkr_") or sample_id.startswith("hkr_words"):
        return "hkr_words"
    if sample_id.startswith("cyr_") or sample_id.startswith("cyrillic_"):
        return "cyrillic_handwriting"
    if sample_id.startswith("school_") or sample_id.startswith("school_notebooks"):
        return "school_notebooks_clean"
    return "unknown"


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    names = row.get("graph_feature_names") or []
    values = row.get("graph_features") or []
    return {
        str(name): float(value)
        for name, value in zip(names, values)
    }


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        row = dict(row)
        features = feature_dict(row)
        diagnostics = row.get("school_foreground_diagnostics") or {}
        for key, value in diagnostics.items():
            try:
                features[str(key)] = float(value)
            except Exception:
                pass
        row["_features"] = features
        out[str(row["sample_id"])] = row
    return out


def overlay_quality(
    manifests: dict[str, dict[str, dict[str, Any]]],
    quality_root: Path,
) -> None:
    for split in ["val", "test"]:
        manifest = manifests[split]
        for bucket in ["clean_core", "hard_real", "invalid_or_review"]:
            path = quality_root / f"{split}.{bucket}.jsonl"
            if not path.exists():
                continue
            for row in read_jsonl(path):
                sample_id = str(row["sample_id"])
                if sample_id not in manifest:
                    continue
                target = manifest[sample_id]
                target["iter2_quality_bucket"] = row.get("iter2_quality_bucket", bucket)
                target["iter2_quality_reasons"] = row.get("iter2_quality_reasons", [])
                diagnostics = row.get("school_foreground_diagnostics") or {}
                for key, value in diagnostics.items():
                    try:
                        target["_features"][str(key)] = float(value)
                    except Exception:
                        pass


def decode_with_confidence(
    probs: torch.Tensor,
    vocab: CTCVocab,
) -> tuple[str, list[float]]:
    ids = probs.argmax(dim=-1).detach().cpu().tolist()
    max_probs = probs.max(dim=-1).values.detach().cpu().tolist()

    chars = []
    confs = []
    prev = None
    for idx, conf in zip(ids, max_probs):
        idx = int(idx)
        if idx == vocab.blank_index:
            prev = idx
            continue
        if idx == prev:
            prev = idx
            continue
        chars.append(vocab.idx_to_char.get(idx, ""))
        confs.append(float(conf))
        prev = idx
    return "".join(chars), confs


def confidence_features_for_sample(
    log_probs_sample: torch.Tensor,
    vocab: CTCVocab,
) -> tuple[str, dict[str, float]]:
    probs = log_probs_sample.exp()
    max_probs, _ = probs.max(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    entropy = -(probs * torch.clamp(log_probs_sample, min=-80.0)).sum(dim=-1)
    entropy = entropy / math.log(float(vocab.num_classes))

    pred, char_confs = decode_with_confidence(probs, vocab)
    frame_count = max(1, int(probs.shape[0]))
    argmax_ids = probs.argmax(dim=-1)
    blank_ratio = float((argmax_ids == vocab.blank_index).float().mean().item())

    emitted = np.asarray(char_confs, dtype=np.float64)
    avg_char_conf = float(emitted.mean()) if emitted.size else 0.0
    seq_score = float(np.log(np.clip(emitted, 1e-6, 1.0)).mean()) if emitted.size else -20.0

    return pred, {
        "mean_max_prob": float(max_probs.mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "argmax_blank_ratio": blank_ratio,
        "decoded_len_per_frame": float(len(pred) / frame_count),
        "avg_decoded_char_confidence": avg_char_conf,
        "mean_frame_margin": float((top2[:, 0] - top2[:, 1]).mean().item()),
        "sequence_score_proxy": seq_score,
    }


def compute_or_load_confidence_predictions(
    *,
    model_name: str,
    split: str,
    manifest_path: Path,
    out_root: Path,
    batch_size: int,
    num_workers: int,
    device_arg: str | None,
) -> list[dict[str, Any]]:
    cache_name = f"{model_name}.{split}.jsonl"
    if model_name in FINAL_BLANK_LOGIT_PENALTIES:
        safe_penalty = str(FINAL_BLANK_LOGIT_PENALTIES[model_name]).replace("-", "m").replace(".", "p")
        cache_name = f"{model_name}.{split}.penalty_{safe_penalty}.jsonl"

    cache_path = out_root / "confidence_predictions" / cache_name
    if cache_path.exists():
        return read_jsonl(cache_path)

    config = MODELS[model_name]
    device = torch.device(device_arg if device_arg else ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = CTCVocab.from_path(config["vocab"])
    model, ckpt_info = load_model_from_checkpoint(
        checkpoint_path=Path(config["checkpoint"]),
        vocab=vocab,
        device=device,
    )
    blank_penalty = float(
        FINAL_BLANK_LOGIT_PENALTIES.get(
            model_name,
            ckpt_info.get("checkpoint_blank_logit_penalty") or 0.0,
        )
    )

    ds = HTRDataset(manifest_path, vocab)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_htr_batch,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["images"].to(device)
            widths = batch["widths"].to(device)
            log_probs = model(images)
            log_probs = apply_blank_logit_penalty(
                log_probs,
                blank_index=vocab.blank_index,
                penalty=blank_penalty,
            )
            input_lengths = model.output_lengths(widths).detach().cpu().tolist()

            for b, length in enumerate(input_lengths):
                sample_log_probs = log_probs[: int(length), b, :].detach().cpu()
                pred, conf = confidence_features_for_sample(sample_log_probs, vocab)
                target = str(batch["texts"][b])
                row = {
                    "sample_id": str(batch["sample_ids"][b]),
                    "dataset": dataset_from_sample_id(str(batch["sample_ids"][b])),
                    "target": target,
                    "pred": pred,
                    "cer": cer(pred, target),
                    "wer": wer(pred, target),
                    "exact": exact_match(pred, target),
                    "exact_error": 1.0 - exact_match(pred, target),
                    "frame_count": int(length),
                    "blank_logit_penalty": blank_penalty,
                    **conf,
                }
                rows.append(row)

            if batch_idx % 50 == 0:
                print(f"[{model_name} {split}] batch {batch_idx}/{len(loader)}")

    write_jsonl(rows, cache_path)
    return rows


def model_matrix(
    rows: list[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        if mode in {"model_confidence", "confidence_graph"}:
            values.extend(float(row[key]) for key in CONFIDENCE_FEATURES)
        if mode == "confidence_graph":
            features = manifests[str(row["sample_id"])]["_features"]
            values.extend(float(features.get(key, 0.0)) for key in GRAPH_FEATURES)
            quality = manifests[str(row["sample_id"])].get("iter2_quality_bucket", "")
            values.append(1.0 if quality == "hard_real" else 0.0)
            values.append(1.0 if quality == "invalid_or_review" else 0.0)
        if mode == "feature_only":
            features = manifests[str(row["sample_id"])]["_features"]
            values.extend(float(features.get(key, 0.0)) for key in GRAPH_FEATURES)
            quality = manifests[str(row["sample_id"])].get("iter2_quality_bucket", "")
            values.append(1.0 if quality == "hard_real" else 0.0)
            values.append(1.0 if quality == "invalid_or_review" else 0.0)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def fit_risk_model(
    val_rows: list[dict[str, Any]],
    val_manifest: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> tuple[StandardScaler, LogisticRegression]:
    x = model_matrix(val_rows, val_manifest, mode=mode)
    y = np.asarray([float(row["exact_error"]) for row in val_rows], dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(x_scaled, y)
    return scaler, clf


def predict_risk(
    rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    scaler: StandardScaler,
    clf: LogisticRegression,
    *,
    mode: str,
) -> np.ndarray:
    x = model_matrix(rows, manifest, mode=mode)
    return clf.predict_proba(scaler.transform(x))[:, 1]


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "cer": mean([float(row["cer"]) for row in rows]),
        "wer": mean([float(row["wer"]) for row in rows]),
        "exact": mean([float(row["exact"]) for row in rows]),
    }


def ece(rows: list[dict[str, Any]], *, n_bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"ece": None, "bins": []}
    bins = []
    total = len(rows)
    ece_value = 0.0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        bucket = [
            row for row in rows
            if (low <= float(row["risk"]) < high)
            or (i == n_bins - 1 and float(row["risk"]) == 1.0)
        ]
        if not bucket:
            bins.append({
                "bin_low": low,
                "bin_high": high,
                "n": 0,
                "mean_risk": None,
                "error_rate": None,
            })
            continue
        mean_risk = mean([float(row["risk"]) for row in bucket])
        error_rate = mean([float(row["exact_error"]) for row in bucket])
        ece_value += (len(bucket) / total) * abs(error_rate - mean_risk)
        bins.append({
            "bin_low": low,
            "bin_high": high,
            "n": len(bucket),
            "mean_risk": mean_risk,
            "error_rate": error_rate,
        })
    return {"ece": ece_value, "bins": bins}


def auc(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    y = np.asarray([float(row["exact_error"]) for row in rows], dtype=np.int64)
    if len(np.unique(y)) < 2:
        return None
    scores = np.asarray([float(row["risk"]) for row in rows], dtype=np.float64)
    return float(roc_auc_score(y, scores))


def coverage_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["risk"]), str(row["sample_id"])))
    out = []
    for coverage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n = max(1, int(round(len(ordered) * coverage)))
        kept = ordered[:n]
        m = metrics(kept)
        out.append({
            "coverage": coverage,
            "n": n,
            "cer": m["cer"],
            "wer": m["wer"],
            "exact": m["exact"],
            "mean_risk": mean([float(row["risk"]) for row in kept]),
        })
    return out


def bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for dataset in DATASETS:
        out[f"dataset:{dataset}"] = metrics([row for row in rows if row["dataset"] == dataset])
    school = [row for row in rows if row["dataset"] == "school_notebooks_clean"]
    for bucket in ["clean_core", "hard_real", "invalid_or_review"]:
        out[f"school_quality:{bucket}"] = metrics([
            row for row in school
            if row.get("school_quality_bucket") == bucket
        ])
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--school_quality_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    manifests = {
        "val": load_manifest(Path(args.val_manifest)),
        "test": load_manifest(Path(args.test_manifest)),
    }
    overlay_quality(manifests, Path(args.school_quality_root))

    summary: dict[str, Any] = {
        "models": {},
        "methods": ["feature_only", "model_confidence", "confidence_graph"],
    }
    curve_rows = []
    risk_table_rows = []
    calibration_rows = []

    for model_name in MODELS:
        val_rows = compute_or_load_confidence_predictions(
            model_name=model_name,
            split="val",
            manifest_path=Path(args.val_manifest),
            out_root=out_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device,
        )
        test_rows = compute_or_load_confidence_predictions(
            model_name=model_name,
            split="test",
            manifest_path=Path(args.test_manifest),
            out_root=out_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device,
        )

        summary["models"][model_name] = {
            "overall": metrics(test_rows),
            "risk_methods": {},
        }

        for mode in ["feature_only", "model_confidence", "confidence_graph"]:
            scaler, clf = fit_risk_model(
                val_rows,
                manifests["val"],
                mode=mode,
            )
            risks = predict_risk(
                test_rows,
                manifests["test"],
                scaler,
                clf,
                mode=mode,
            )

            enriched = []
            for row, risk in zip(test_rows, risks):
                item = dict(row)
                meta = manifests["test"].get(str(row["sample_id"]), {})
                item["risk"] = float(risk)
                item["school_quality_bucket"] = str(meta.get("iter2_quality_bucket", ""))
                enriched.append(item)

            method_summary = {
                "overall": metrics(enriched),
                "risk_auc_exact_error_all": auc(enriched),
                "risk_auc_exact_error_school": auc([
                    row for row in enriched
                    if row["dataset"] == "school_notebooks_clean"
                ]),
                "ece_all": ece(enriched)["ece"],
                "ece_school": ece([
                    row for row in enriched
                    if row["dataset"] == "school_notebooks_clean"
                ])["ece"],
                "bucket_metrics": bucket_metrics(enriched),
            }
            summary["models"][model_name]["risk_methods"][mode] = method_summary

            for scope, scoped in [
                ("all", enriched),
                ("school", [
                    row for row in enriched
                    if row["dataset"] == "school_notebooks_clean"
                ]),
            ]:
                for row in coverage_curve(scoped):
                    curve_rows.append({
                        "model": model_name,
                        "risk_method": mode,
                        "scope": scope,
                        **row,
                    })

            for bucket_key, bucket_value in method_summary["bucket_metrics"].items():
                bucket_type, bucket = bucket_key.split(":", 1)
                risk_table_rows.append({
                    "model": model_name,
                    "risk_method": mode,
                    "bucket_type": bucket_type,
                    "bucket": bucket,
                    **bucket_value,
                })

            for scope, scoped in [
                ("all", enriched),
                ("school", [
                    row for row in enriched
                    if row["dataset"] == "school_notebooks_clean"
                ]),
            ]:
                cal = ece(scoped)
                for row in cal["bins"]:
                    calibration_rows.append({
                        "model": model_name,
                        "risk_method": mode,
                        "scope": scope,
                        **row,
                    })

    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "selective_summary.json", summary)

    with (out_root / "coverage_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    with (out_root / "risk_table_by_bucket.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(risk_table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(risk_table_rows)

    with (out_root / "calibration_bins.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calibration_rows)

    lines = [
        "# Confidence-Aware Selective Prediction - Iteration 2",
        "",
        "Risk models are fit on val exact-error labels and evaluated on test.",
        "",
        "## Risk Quality",
        "",
        "| model | risk method | AUC all | AUC School | ECE all | ECE School |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        for mode in ["feature_only", "model_confidence", "confidence_graph"]:
            row = summary["models"][model_name]["risk_methods"][mode]
            lines.append(
                f"| `{model_name}` | `{mode}` | "
                f"{fmt(row['risk_auc_exact_error_all'])} | {fmt(row['risk_auc_exact_error_school'])} | "
                f"{fmt(row['ece_all'])} | {fmt(row['ece_school'])} |"
            )

    lines.extend([
        "",
        "## Selective Coverage",
        "",
        "| model | risk method | scope | coverage | CER | WER | exact |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for model_name in MODELS:
        for mode in ["feature_only", "model_confidence", "confidence_graph"]:
            for scope in ["all", "school"]:
                for coverage in [0.5, 0.7, 0.8, 0.9, 1.0]:
                    match = next(
                        row for row in curve_rows
                        if row["model"] == model_name
                        and row["risk_method"] == mode
                        and row["scope"] == scope
                        and abs(float(row["coverage"]) - coverage) < 1e-9
                    )
                    lines.append(
                        f"| `{model_name}` | `{mode}` | `{scope}` | {fmt(coverage)} | "
                        f"{fmt(match['cer'])} | {fmt(match['wer'])} | {fmt(match['exact'])} |"
                    )

    lines.extend([
        "",
        "## School clean_core vs hard_real",
        "",
        "| model | risk method | bucket | n | CER | WER | exact |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for model_name in MODELS:
        for mode in ["feature_only", "model_confidence", "confidence_graph"]:
            buckets = summary["models"][model_name]["risk_methods"][mode]["bucket_metrics"]
            for bucket in ["clean_core", "hard_real"]:
                row = buckets[f"school_quality:{bucket}"]
                lines.append(
                    f"| `{model_name}` | `{mode}` | `{bucket}` | {row['n']} | "
                    f"{fmt(row['cer'])} | {fmt(row['wer'])} | {fmt(row['exact'])} |"
                )

    lines.extend([
        "",
        "## Files",
        "",
        "- `selective_summary.json`",
        "- `coverage_curves.csv`",
        "- `risk_table_by_bucket.csv`",
        "- `calibration_bins.csv`",
        "- `confidence_predictions/*.jsonl`",
    ])
    (out_root / "selective_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "out_root": str(out_root),
        "models": {
            model: {
                mode: {
                    "auc_all": summary["models"][model]["risk_methods"][mode]["risk_auc_exact_error_all"],
                    "auc_school": summary["models"][model]["risk_methods"][mode]["risk_auc_exact_error_school"],
                    "ece_all": summary["models"][model]["risk_methods"][mode]["ece_all"],
                }
                for mode in ["feature_only", "model_confidence", "confidence_graph"]
            }
            for model in MODELS
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
