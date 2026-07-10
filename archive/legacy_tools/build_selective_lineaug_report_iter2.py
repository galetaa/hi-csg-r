from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAMES = [
    "baseline",
    "plus_5k_context",
    "plus_10k_context",
]

DATASETS = [
    "hkr_words",
    "cyrillic_handwriting",
    "school_notebooks_clean",
]

TEXT_LEN_BUCKETS = [
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-10", 7, 10),
    ("11+", 11, 10**9),
]

RISK_FEATURES = [
    "fg_fraction",
    "skel_fraction",
    "cc_count",
    "dir_h_frac",
    "stroke_width_mean",
    "aspect_ratio",
    "bbox_area_frac",
    "branchpoint_per_100_skel",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def text_len_bucket(text: str) -> str:
    n = len(text)
    for name, low, high in TEXT_LEN_BUCKETS:
        if low <= n <= high:
            return name
    return "0"


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    names = row.get("graph_feature_names") or []
    values = row.get("graph_features") or []
    return {
        str(name): float(value)
        for name, value in zip(names, values)
    }


def load_predictions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row["dataset"] = dataset_from_sample_id(str(row["sample_id"]))
            out[str(row["sample_id"])] = row
    return out


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


def overlay_school_quality(
    manifest: dict[str, dict[str, Any]],
    quality_root: Path | None,
) -> None:
    if quality_root is None:
        return

    for bucket in ["clean_core", "hard_real", "invalid_or_review"]:
        path = quality_root / f"test.{bucket}.jsonl"
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


def percentile_lookup(values: list[float], value: float) -> float:
    if not values:
        return 0.5
    arr = np.asarray(values, dtype=np.float64)
    return float(np.searchsorted(np.sort(arr), value, side="right") / arr.size)


def build_feature_distributions(
    manifest: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in manifest.values():
        dataset = str(row.get("dataset", "unknown"))
        features = row["_features"]
        for key in RISK_FEATURES:
            if key in features:
                out[dataset][key].append(float(features[key]))
    return out


def feature_extremeness(
    *,
    dataset: str,
    features: dict[str, float],
    distributions: dict[str, dict[str, list[float]]],
) -> float:
    scores = []
    for key in RISK_FEATURES:
        if key not in features:
            continue
        percentile = percentile_lookup(
            distributions.get(dataset, {}).get(key, []),
            float(features[key]),
        )
        scores.append(abs(percentile - 0.5) * 2.0)
    return float(np.mean(scores)) if scores else 0.0


def school_rule_risk(row: dict[str, Any]) -> float:
    features = row["_features"]
    score = 0.0

    fg = float(features.get("fg_fraction", 0.0))
    skel = float(features.get("skel_fraction", 0.0))
    cc = float(features.get("cc_count", 0.0))
    dir_h = float(features.get("dir_h_frac", 0.0))
    stroke = float(features.get("stroke_width_mean", 0.0))
    ruling = float(features.get("ruling_response_mean", 0.0))
    warnings = float(features.get("warning_count", row.get("graph_warning_count", 0.0)))

    if fg < 0.055:
        score += min(1.0, (0.055 - fg) / 0.055)
    if fg > 0.175:
        score += min(1.0, (fg - 0.175) / 0.105)
    if skel < 0.015:
        score += min(1.0, (0.015 - skel) / 0.015)
    if skel > 0.065:
        score += min(1.0, (skel - 0.065) / 0.06)
    if cc > 12:
        score += min(1.0, (cc - 12.0) / 20.0)
    if dir_h > 0.50:
        score += min(1.0, (dir_h - 0.50) / 0.35)
    if stroke > 5.0:
        score += min(1.0, (stroke - 5.0) / 3.0)
    if ruling > 22.0:
        score += min(1.0, (ruling - 22.0) / 30.0)
    if warnings > 0:
        score += min(1.0, warnings / 3.0)

    return score / 9.0


def risk_score(
    row: dict[str, Any],
    distributions: dict[str, dict[str, list[float]]],
) -> float:
    dataset = str(row.get("dataset", "unknown"))
    features = row["_features"]
    text_len = int(features.get("text_len", row.get("text_len", 0) or 0))

    risk = 0.60 * feature_extremeness(
        dataset=dataset,
        features=features,
        distributions=distributions,
    )

    if dataset == "school_notebooks_clean":
        risk += 0.60 * school_rule_risk(row)
        if row.get("iter2_quality_bucket") == "hard_real":
            risk += 0.20
        elif row.get("iter2_quality_bucket") == "invalid_or_review":
            risk += 0.50

    if text_len <= 3:
        risk += 0.25

    return float(risk)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def metrics(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return {
        "n": len(rows),
        "cer": mean([float(row[f"{model}_cer"]) for row in rows]),
        "wer": mean([float(row[f"{model}_wer"]) for row in rows]),
        "exact": mean([float(row[f"{model}_exact"]) for row in rows]),
        "mean_risk_score": mean([float(row["risk_score"]) for row in rows]),
    }


def auc_for_error(rows: list[dict[str, Any]], model: str) -> float | None:
    positives = [row for row in rows if float(row[f"{model}_exact"]) < 1.0]
    negatives = [row for row in rows if float(row[f"{model}_exact"]) >= 1.0]
    if not positives or not negatives:
        return None

    pos_scores = np.asarray([row["risk_score"] for row in positives], dtype=np.float64)
    neg_scores = np.asarray([row["risk_score"] for row in negatives], dtype=np.float64)
    wins = 0.0
    total = float(pos_scores.size * neg_scores.size)
    for score in pos_scores:
        wins += float(np.sum(score > neg_scores))
        wins += 0.5 * float(np.sum(score == neg_scores))
    return wins / total


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(y) < 3:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    x_rank = np.argsort(np.argsort(x_arr)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y_arr)).astype(np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def bucket_rows(rows: list[dict[str, Any]], bucket_type: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bucket_type == "dataset":
            key = row["dataset"]
        elif bucket_type == "text_len":
            key = row["text_len_bucket"]
        elif bucket_type == "school_quality":
            if row["dataset"] != "school_notebooks_clean":
                continue
            key = row.get("school_quality_bucket") or "unknown"
        elif bucket_type == "school_ruling":
            if row["dataset"] != "school_notebooks_clean":
                continue
            ruling = float(row.get("ruling_response_mean", 0.0))
            if ruling <= 10:
                key = "ruling_low_<=10"
            elif ruling <= 22:
                key = "ruling_mid_10_22"
            else:
                key = "ruling_high_>22"
        elif bucket_type == "risk_quintile":
            key = row["risk_quintile"]
        else:
            raise ValueError(bucket_type)
        out[str(key)].append(row)
    return out


def assign_risk_quintiles(rows: list[dict[str, Any]]) -> None:
    scores = np.asarray([row["risk_score"] for row in rows], dtype=np.float64)
    cuts = [float(np.quantile(scores, q)) for q in [0.2, 0.4, 0.6, 0.8]]
    for row in rows:
        score = float(row["risk_score"])
        idx = sum(score > cut for cut in cuts) + 1
        row["risk_quintile"] = f"q{idx}"


def coverage_curve(
    rows: list[dict[str, Any]],
    *,
    model: str,
    scope: str,
) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: (row["risk_score"], row["sample_id"]))
    out = []
    for coverage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        keep_n = max(1, int(round(len(sorted_rows) * coverage)))
        kept = sorted_rows[:keep_n]
        m = metrics(kept, model)
        out.append({
            "model": model,
            "scope": scope,
            "coverage": coverage,
            "n": keep_n,
            "cer": m["cer"],
            "wer": m["wer"],
            "exact": m["exact"],
            "mean_risk_score": m["mean_risk_score"],
        })
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline_predictions", nargs="+", required=True)
    parser.add_argument("--plus5_predictions", required=True)
    parser.add_argument("--plus10_predictions", required=True)
    parser.add_argument("--school_quality_root", default=None)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    overlay_school_quality(
        manifest,
        Path(args.school_quality_root) if args.school_quality_root else None,
    )
    distributions = build_feature_distributions(manifest)

    predictions = {
        "baseline": load_predictions([Path(p) for p in args.baseline_predictions]),
        "plus_5k_context": load_predictions([Path(args.plus5_predictions)]),
        "plus_10k_context": load_predictions([Path(args.plus10_predictions)]),
    }

    common = sorted(set(manifest) & set.intersection(*(set(p) for p in predictions.values())))
    rows = []
    for sample_id in common:
        meta = manifest[sample_id]
        features = meta["_features"]
        text = str(meta.get("text") or meta.get("normalized_transcription") or predictions["baseline"][sample_id].get("target", ""))
        row = {
            "sample_id": sample_id,
            "dataset": str(meta.get("dataset", predictions["baseline"][sample_id]["dataset"])),
            "text": text,
            "text_len": len(text),
            "text_len_bucket": text_len_bucket(text),
            "school_quality_bucket": str(meta.get("iter2_quality_bucket", "")),
            "school_quality_reasons": ";".join(meta.get("iter2_quality_reasons") or []),
            "ruling_response_mean": float(features.get("ruling_response_mean", 0.0)),
            "ruling_response_p95": float(features.get("ruling_response_p95", 0.0)),
            "risk_score": risk_score(meta, distributions),
        }
        for key in [
            "fg_fraction",
            "skel_fraction",
            "cc_count",
            "dir_h_frac",
            "stroke_width_mean",
            "warning_count",
        ]:
            row[key] = float(features.get(key, 0.0))

        for model in MODEL_NAMES:
            pred = predictions[model][sample_id]
            row[f"{model}_cer"] = float(pred["cer"])
            row[f"{model}_wer"] = float(pred["wer"])
            row[f"{model}_exact"] = float(pred.get("exact", 0.0))
            row[f"{model}_pred"] = str(pred.get("pred", ""))
        rows.append(row)

    assign_risk_quintiles(rows)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    risk_table_rows = []
    for bucket_type in [
        "dataset",
        "text_len",
        "school_quality",
        "school_ruling",
        "risk_quintile",
    ]:
        for bucket, bucketed in sorted(bucket_rows(rows, bucket_type).items()):
            for model in MODEL_NAMES:
                m = metrics(bucketed, model)
                baseline_m = metrics(bucketed, "baseline")
                risk_table_rows.append({
                    "bucket_type": bucket_type,
                    "bucket": bucket,
                    "model": model,
                    "n": m["n"],
                    "cer": m["cer"],
                    "wer": m["wer"],
                    "exact": m["exact"],
                    "delta_cer_vs_baseline": m["cer"] - baseline_m["cer"],
                    "delta_wer_vs_baseline": m["wer"] - baseline_m["wer"],
                    "delta_exact_vs_baseline": m["exact"] - baseline_m["exact"],
                    "mean_risk_score": m["mean_risk_score"],
                })

    with (out_root / "risk_table_by_bucket.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(risk_table_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(risk_table_rows)

    curve_rows = []
    for model in MODEL_NAMES:
        curve_rows.extend(coverage_curve(rows, model=model, scope="all"))
        school_rows = [row for row in rows if row["dataset"] == "school_notebooks_clean"]
        curve_rows.extend(coverage_curve(school_rows, model=model, scope="school"))

    with (out_root / "coverage_curves.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(curve_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)

    auc = {
        model: {
            "all": auc_for_error(rows, model),
            "school": auc_for_error(
                [row for row in rows if row["dataset"] == "school_notebooks_clean"],
                model,
            ),
        }
        for model in MODEL_NAMES
    }

    school_rows = [row for row in rows if row["dataset"] == "school_notebooks_clean"]
    feature_correlations = {
        model: {
            key: spearman(
                [float(row[key]) for row in school_rows],
                [float(row[f"{model}_cer"]) for row in school_rows],
            )
            for key in [
                "risk_score",
                "fg_fraction",
                "skel_fraction",
                "cc_count",
                "dir_h_frac",
                "stroke_width_mean",
                "ruling_response_mean",
            ]
        }
        for model in MODEL_NAMES
    }

    summary = {
        "n_common": len(rows),
        "models": MODEL_NAMES,
        "overall": {
            model: metrics(rows, model)
            for model in MODEL_NAMES
        },
        "school_clean_vs_hard": {
            model: {
                bucket: metrics([
                    row for row in school_rows
                    if row["school_quality_bucket"] == bucket
                ], model)
                for bucket in ["clean_core", "hard_real", "invalid_or_review"]
            }
            for model in MODEL_NAMES
        },
        "risk_auc_for_exact_error": auc,
        "school_feature_spearman_with_cer": feature_correlations,
    }

    write_json(out_root / "selective_summary.json", summary)

    lines = [
        "# Selective Prediction - Iteration 2 Line Augmentation",
        "",
        "Risk score is feature-only: graph/foreground extremeness, School quality rules, graph warnings, and short-text risk. Predictions are not used to compute the risk score.",
        "",
        "## Overall",
        "",
        "| model | n | CER | WER | exact | risk AUC all | risk AUC School |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        m = summary["overall"][model]
        lines.append(
            f"| `{model}` | {m['n']} | {fmt(m['cer'])} | {fmt(m['wer'])} | {fmt(m['exact'])} | "
            f"{fmt(auc[model]['all'])} | {fmt(auc[model]['school'])} |"
        )

    lines.extend([
        "",
        "## School Clean vs Hard",
        "",
        "| model | bucket | n | CER | WER | exact |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for model in MODEL_NAMES:
        for bucket in ["clean_core", "hard_real", "invalid_or_review"]:
            m = summary["school_clean_vs_hard"][model][bucket]
            lines.append(
                f"| `{model}` | `{bucket}` | {m['n']} | {fmt(m['cer'])} | {fmt(m['wer'])} | {fmt(m['exact'])} |"
            )

    def curve_lookup(model: str, scope: str, coverage: float) -> dict[str, Any]:
        for row in curve_rows:
            if row["model"] == model and row["scope"] == scope and abs(float(row["coverage"]) - coverage) < 1e-9:
                return row
        raise KeyError((model, scope, coverage))

    lines.extend([
        "",
        "## Selective Coverage",
        "",
        "| model | scope | coverage | CER | WER | exact |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for model in MODEL_NAMES:
        for scope in ["all", "school"]:
            for coverage in [0.5, 0.7, 0.8, 0.9, 1.0]:
                row = curve_lookup(model, scope, coverage)
                lines.append(
                    f"| `{model}` | `{scope}` | {fmt(coverage)} | {fmt(row['cer'])} | {fmt(row['wer'])} | {fmt(row['exact'])} |"
                )

    lines.extend([
        "",
        "## School Feature Signal",
        "",
        "Spearman correlation with per-sample CER on School test samples.",
        "",
        "| model | feature | rho |",
        "|---|---|---:|",
    ])
    for model in MODEL_NAMES:
        for key, value in feature_correlations[model].items():
            lines.append(f"| `{model}` | `{key}` | {fmt(value)} |")

    lines.extend([
        "",
        "## Files",
        "",
        "- `risk_table_by_bucket.csv`",
        "- `coverage_curves.csv`",
        "- `selective_summary.json`",
    ])

    (out_root / "selective_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "out_root": str(out_root),
        "n_common": len(rows),
        "overall": summary["overall"],
        "risk_auc_for_exact_error": auc,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
