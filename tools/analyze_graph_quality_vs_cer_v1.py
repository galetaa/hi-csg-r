from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


LEAKY_OR_TARGET_FEATURES = {
    "text_len",
    "target_len",
    "label_len",
    "num_chars",
    "char_count",
    "word_count",
}


CORE_GRAPH_FEATURES = [
    "fg_fraction",
    "bbox_area_frac",
    "cc_count",
    "cc_area_mean",
    "cc_area_median",
    "cc_area_max_frac",
    "skel_pixels",
    "skel_fraction",
    "skel_components",
    "graph_nodes",
    "graph_edges_8n",
    "graph_avg_degree",
    "graph_endpoint_count",
    "graph_branchpoint_count",
    "graph_isolated_count",
    "endpoint_per_100_skel",
    "branchpoint_per_100_skel",
    "degree_hist_0",
    "degree_hist_1",
    "degree_hist_2",
    "degree_hist_3",
    "degree_hist_4",
    "degree_hist_5plus",
    "dir_h_frac",
    "dir_v_frac",
    "dir_diag_down_frac",
    "dir_diag_up_frac",
    "stroke_width_mean",
    "stroke_width_std",
    "stroke_width_p50",
    "stroke_width_p90",
    "warning_count",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def edit_distance(a: list[Any], b: list[Any]) -> int:
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur

    return prev[-1]


def cer(pred: str, target: str) -> float:
    return edit_distance(list(pred), list(target)) / max(len(target), 1)


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def sample_key(row: dict[str, Any]) -> str:
    for key in ["sample_id", "clean_sample_id", "id"]:
        v = row.get(key)
        if v is not None:
            return str(v)
    raise KeyError(f"No sample_id-like key. Keys: {sorted(row.keys())}")


def flatten_numeric_dict(prefix: str, d: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in d.items():
        fv = safe_float(v)
        if fv is not None:
            out[f"{prefix}{k}"] = fv
    return out


def extract_graph_features(row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}

    gf = row.get("graph_features")
    names = row.get("graph_feature_names")

    if isinstance(gf, dict):
        for k, v in gf.items():
            fv = safe_float(v)
            if fv is not None:
                features[str(k)] = fv

    elif isinstance(gf, list):
        if isinstance(names, list) and len(names) == len(gf):
            for k, v in zip(names, gf):
                fv = safe_float(v)
                if fv is not None:
                    features[str(k)] = fv
        else:
            for i, v in enumerate(gf):
                fv = safe_float(v)
                if fv is not None:
                    features[f"graph_feature_{i:03d}"] = fv

    quality = row.get("quality")
    if isinstance(quality, dict):
        features.update(flatten_numeric_dict("quality_", quality))

    graph_quality = row.get("graph_quality")
    if isinstance(graph_quality, dict):
        features.update(flatten_numeric_dict("graph_quality_", graph_quality))

    warnings = row.get("warnings")
    if isinstance(warnings, list):
        features["warning_list_count"] = float(len(warnings))

    cleaned = {}
    for k, v in features.items():
        if k in LEAKY_OR_TARGET_FEATURES:
            continue
        if k.lower() in LEAKY_OR_TARGET_FEATURES:
            continue
        cleaned[k] = float(v)

    return cleaned


def load_manifest_features(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out = {}

    for row in rows:
        key = sample_key(row)
        feats = extract_graph_features(row)

        meta = {
            "sample_id": key,
            "dataset": row.get("source_dataset") or row.get("dataset"),
            "level": row.get("level"),
            "category": row.get("category"),
            "image_path": row.get("image_path"),
            "features": feats,
        }
        out[key] = meta

    return out


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out = {}

    for row in rows:
        key = sample_key(row)
        target = str(row.get("target", row.get("text", "")))
        pred = str(row.get("pred", row.get("prediction", "")))

        c = safe_float(row.get("cer"))
        if c is None:
            c = cer(pred, target)

        out[key] = {
            "sample_id": key,
            "target": target,
            "pred": pred,
            "cer": float(c),
            "wer": safe_float(row.get("wer")),
            "exact": row.get("exact"),
            "dataset": row.get("dataset"),
            "level": row.get("level"),
            "category": row.get("category"),
        }

    return out


def join_predictions_and_features(
    *,
    manifest_features: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    joined = []

    for sid, pred in predictions.items():
        m = manifest_features.get(sid)
        if m is None:
            continue

        row = {
            "sample_id": sid,
            "target": pred["target"],
            "pred": pred["pred"],
            "cer": pred["cer"],
            "target_len": len(pred["target"]),
            "pred_len": len(pred["pred"]),
            "dataset": pred.get("dataset") or m.get("dataset"),
            "level": pred.get("level") or m.get("level"),
            "category": pred.get("category") or m.get("category"),
            "image_path": m.get("image_path"),
        }

        for k, v in m["features"].items():
            row[k] = v

        joined.append(row)

    return joined


def finite_xy(rows: list[dict[str, Any]], feature: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for r in rows:
        x = safe_float(r.get(feature))
        y = safe_float(r.get("cer"))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def correlation_rows(rows: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
    out = []

    for feature in features:
        x, y = finite_xy(rows, feature)
        if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
            continue

        sp = spearmanr(x, y)
        pr = pearsonr(x, y)

        out.append(
            {
                "feature": feature,
                "n": int(len(x)),
                "spearman_r": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "abs_spearman_r": abs(float(sp.statistic)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }
        )

    out.sort(key=lambda r: (-r["abs_spearman_r"], r["feature"]))
    return out


def high_error_rows(
    rows: list[dict[str, Any]],
    features: list[str],
    high_error_quantile: float,
) -> tuple[list[dict[str, Any]], float]:
    cers = np.asarray([float(r["cer"]) for r in rows], dtype=np.float64)
    threshold = float(np.quantile(cers, high_error_quantile))

    y = np.asarray([1 if float(r["cer"]) >= threshold else 0 for r in rows], dtype=np.int32)

    out = []
    for feature in features:
        xs = []
        yy = []

        for r, label in zip(rows, y):
            x = safe_float(r.get(feature))
            if x is None:
                continue
            xs.append(float(x))
            yy.append(int(label))

        if len(xs) < 20 or len(set(yy)) < 2:
            continue

        x = np.asarray(xs, dtype=np.float64)
        yy_arr = np.asarray(yy, dtype=np.int32)

        if np.std(x) == 0:
            continue

        auc_raw = roc_auc_score(yy_arr, x)
        if auc_raw >= 0.5:
            score = x
            direction = "higher_feature_higher_error"
            roc_auc = auc_raw
        else:
            score = -x
            direction = "lower_feature_higher_error"
            roc_auc = roc_auc_score(yy_arr, score)

        pr_auc = average_precision_score(yy_arr, score)

        # Simple diagnostic threshold: top 20% most suspicious by score.
        score_threshold = float(np.quantile(score, 0.80))
        pred_high = (score >= score_threshold).astype(np.int32)

        precision, recall, f1, _ = precision_recall_fscore_support(
            yy_arr,
            pred_high,
            average="binary",
            zero_division=0,
        )

        out.append(
            {
                "feature": feature,
                "n": int(len(score)),
                "high_error_threshold_cer": threshold,
                "high_error_rate": float(np.mean(yy_arr)),
                "roc_auc_direction_invariant": float(roc_auc),
                "pr_auc": float(pr_auc),
                "direction": direction,
                "top20_precision": float(precision),
                "top20_recall": float(recall),
                "top20_f1": float(f1),
                "feature_mean": float(np.mean(x)),
                "feature_std": float(np.std(x)),
            }
        )

    out.sort(key=lambda r: (-r["roc_auc_direction_invariant"], -r["pr_auc"], r["feature"]))
    return out, threshold


def dataset_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        key = str(r.get("dataset") or "unknown")
        groups[key].append(r)

    out = []
    for dataset, group in sorted(groups.items()):
        cers = [float(r["cer"]) for r in group]
        out.append(
            {
                "dataset": dataset,
                "n": len(group),
                "mean_cer": float(np.mean(cers)),
                "median_cer": float(np.median(cers)),
                "p90_cer": float(np.quantile(cers, 0.90)),
                "exact_rate": float(np.mean([1.0 if float(r["cer"]) == 0.0 else 0.0 for r in group])),
            }
        )

    return out


def worst_samples(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: float(r["cer"]), reverse=True)

    out = []
    for r in sorted_rows[:n]:
        item = {
            "sample_id": r["sample_id"],
            "dataset": r.get("dataset"),
            "level": r.get("level"),
            "category": r.get("category"),
            "cer": float(r["cer"]),
            "target": r.get("target"),
            "pred": r.get("pred"),
            "image_path": r.get("image_path"),
        }

        for k in [
            "warning_count",
            "warning_list_count",
            "fg_fraction",
            "skel_fraction",
            "graph_endpoint_count",
            "graph_branchpoint_count",
            "endpoint_per_100_skel",
            "branchpoint_per_100_skel",
            "stroke_width_mean",
            "stroke_width_std",
        ]:
            if k in r:
                item[k] = r[k]

        out.append(item)

    return out


def select_features(rows: list[dict[str, Any]]) -> list[str]:
    candidates = set()

    for r in rows:
        for k, v in r.items():
            if k in {
                "sample_id",
                "target",
                "pred",
                "cer",
                "wer",
                "exact",
                "dataset",
                "level",
                "category",
                "image_path",
                "target_len",
                "pred_len",
            }:
                continue

            if k in LEAKY_OR_TARGET_FEATURES:
                continue

            if safe_float(v) is not None:
                candidates.add(k)

    # Put core graph features first when present, then any extras.
    ordered = [f for f in CORE_GRAPH_FEATURES if f in candidates]
    ordered.extend(sorted(candidates - set(ordered)))
    return ordered


def make_report_md(
    *,
    out_path: Path,
    summary: dict[str, Any],
    correlations: list[dict[str, Any]],
    high_error: list[dict[str, Any]],
    breakdown: list[dict[str, Any]],
) -> None:
    lines = []

    lines.append("# H3 graph quality vs CER report — v1")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report tests whether graph-derived structural features are diagnostically "
        "related to recognition errors. The primary analysis uses image-only predictions "
        "so the graph features are not part of the model input."
    )
    lines.append("")
    lines.append("## 2. Dataset")
    lines.append("")
    lines.append("```text")
    lines.append(f"joined samples: {summary['joined_n']}")
    lines.append(f"manifest samples: {summary['manifest_n']}")
    lines.append(f"prediction samples: {summary['prediction_n']}")
    lines.append(f"high-error quantile: {summary['high_error_quantile']}")
    lines.append(f"high-error CER threshold: {summary['high_error_threshold_cer']:.5f}")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Dataset CER breakdown")
    lines.append("")
    lines.append("| dataset | n | mean CER | median CER | p90 CER | exact-rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in breakdown:
        lines.append(
            f"| `{r['dataset']}` | {r['n']} | {r['mean_cer']:.5f} | "
            f"{r['median_cer']:.5f} | {r['p90_cer']:.5f} | {r['exact_rate']:.5f} |"
        )
    lines.append("")
    lines.append("## 4. Top Spearman correlations with CER")
    lines.append("")
    lines.append("| feature | n | Spearman r | p-value | Pearson r |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in correlations[:20]:
        lines.append(
            f"| `{r['feature']}` | {r['n']} | {r['spearman_r']:.4f} | "
            f"{r['spearman_p']:.3e} | {r['pearson_r']:.4f} |"
        )
    lines.append("")
    lines.append("## 5. Top single-feature high-error detectors")
    lines.append("")
    lines.append("| feature | ROC-AUC | PR-AUC | direction | top20 precision | top20 recall |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for r in high_error[:20]:
        lines.append(
            f"| `{r['feature']}` | {r['roc_auc_direction_invariant']:.4f} | "
            f"{r['pr_auc']:.4f} | {r['direction']} | "
            f"{r['top20_precision']:.4f} | {r['top20_recall']:.4f} |"
        )
    lines.append("")
    lines.append("## 6. Strict interpretation")
    lines.append("")

    best_corr = correlations[0] if correlations else None
    best_auc = high_error[0] if high_error else None

    if best_corr and abs(best_corr["spearman_r"]) >= 0.20:
        corr_text = "There is a non-trivial monotonic relation between at least one graph feature and CER."
    else:
        corr_text = "Correlations are weak; graph features do not strongly explain CER by themselves."

    if best_auc and best_auc["roc_auc_direction_invariant"] >= 0.65:
        auc_text = "At least one graph feature has useful single-feature high-error detection signal."
    else:
        auc_text = "Single-feature high-error detection is weak; multi-feature diagnostics or gold graph quality may be needed."

    lines.append(corr_text)
    lines.append("")
    lines.append(auc_text)
    lines.append("")
    lines.append(
        "This analysis is diagnostic only. It does not prove that graph features improve recognition. "
        "It tests whether graph-derived quality and structural measures are informative about failure cases."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--high_error_quantile", type=float, default=0.80)
    parser.add_argument("--worst_n", type=int, default=100)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    predictions_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_features(manifest_path)
    predictions = load_predictions(predictions_path)
    joined = join_predictions_and_features(
        manifest_features=manifest,
        predictions=predictions,
    )

    if not joined:
        raise RuntimeError(
            "No joined samples. Check that sample_id values match between manifest and predictions."
        )

    features = select_features(joined)

    corr = correlation_rows(joined, features)
    high_error, threshold = high_error_rows(
        joined,
        features,
        high_error_quantile=args.high_error_quantile,
    )
    breakdown = dataset_breakdown(joined)
    worst = worst_samples(joined, args.worst_n)

    summary = {
        "manifest": str(manifest_path),
        "predictions": str(predictions_path),
        "manifest_n": len(manifest),
        "prediction_n": len(predictions),
        "joined_n": len(joined),
        "feature_n": len(features),
        "features": features,
        "high_error_quantile": args.high_error_quantile,
        "high_error_threshold_cer": threshold,
        "best_abs_spearman": corr[0] if corr else None,
        "best_high_error_auc": high_error[0] if high_error else None,
        "dataset_breakdown": breakdown,
    }

    write_csv(corr, out_dir / "feature_cer_correlations.csv")
    write_csv(high_error, out_dir / "feature_high_error_detection.csv")
    write_csv(breakdown, out_dir / "dataset_breakdown.csv")
    write_csv(worst, out_dir / "worst_samples.csv")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_report_md(
        out_path=out_dir / "h3_graph_quality_vs_cer_report_v1.md",
        summary=summary,
        correlations=corr,
        high_error=high_error,
        breakdown=breakdown,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()