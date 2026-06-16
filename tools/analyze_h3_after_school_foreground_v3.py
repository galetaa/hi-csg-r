from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GEOMETRY_KEYS = {
    "width",
    "height",
    "aspect_ratio",
    "bbox_x0_frac",
    "bbox_y0_frac",
    "bbox_w_frac",
    "bbox_h_frac",
    "bbox_area_frac",
}

QUALITY_KEYS = {
    "warning_count",
    "graph_warning_count",
}

LEAKAGE_KEYS = {
    "text_len",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_category(row: dict[str, Any]) -> str:
    if row.get("category"):
        return str(row["category"])
    md = row.get("metadata") or {}
    if isinstance(md, dict) and md.get("category"):
        return str(md["category"])
    return "unknown"


def load_manifest_features(path: Path, drop_text_len: bool) -> dict[str, dict[str, Any]]:
    out = {}

    for r in read_jsonl(path):
        sid = str(r["sample_id"])
        names = list(r.get("graph_feature_names") or [])
        values = list(r.get("graph_features") or [])

        if not names or not values:
            continue

        if len(names) != len(values):
            raise ValueError(f"Feature name/value mismatch for {sid}: {len(names)} vs {len(values)}")

        fmap = {str(k): safe_float(v) for k, v in zip(names, values)}

        if drop_text_len:
            for k in LEAKAGE_KEYS:
                fmap.pop(k, None)

        out[sid] = {
            "sample_id": sid,
            "dataset": str(r.get("dataset") or r.get("source_dataset") or "unknown"),
            "level": str(r.get("level") or "unknown"),
            "category": get_category(r),
            "target": str(r.get("text") or r.get("target") or ""),
            "feature_names": sorted(fmap.keys()),
            "features": fmap,
        }

    return out


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for r in read_jsonl(path):
        sid = str(r["sample_id"])
        out[sid] = {
            "sample_id": sid,
            "pred_target": str(r.get("target") or ""),
            "pred": str(r.get("pred") or ""),
            "cer": safe_float(r.get("cer")),
            "wer": safe_float(r.get("wer")),
            "exact": safe_float(r.get("exact")),
            "pred_level": str(r.get("level") or "unknown"),
            "pred_category": str(r.get("category") or "unknown"),
        }
    return out


def join_rows(
    features: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for sid, fr in features.items():
        pr = predictions.get(sid)
        if pr is None:
            continue

        row = {
            **fr,
            **pr,
        }

        if row["level"] == "unknown" and row.get("pred_level") != "unknown":
            row["level"] = row["pred_level"]

        if row["category"] == "unknown" and row.get("pred_category") != "unknown":
            row["category"] = row["pred_category"]

        rows.append(row)

    return rows


def feature_sets(feature_names: list[str]) -> dict[str, list[str]]:
    names = set(feature_names)

    geometry = sorted(k for k in names if k in GEOMETRY_KEYS)
    quality = sorted(k for k in names if k in QUALITY_KEYS)
    structural = sorted(
        k for k in names
        if k not in GEOMETRY_KEYS and k not in QUALITY_KEYS and k not in LEAKAGE_KEYS
    )
    all_non_geometry = sorted(set(structural) | set(quality))
    all_features = sorted(k for k in names if k not in LEAKAGE_KEYS)

    return {
        "quality_only": quality,
        "geometry_control": geometry,
        "structural_core": structural,
        "all_non_geometry": all_non_geometry,
        "all_features_no_text_len": all_features,
    }


def summarize_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    by_ds = defaultdict(list)

    for r in rows:
        by_ds[r["dataset"]].append(r)

    for ds, group in sorted(by_ds.items()):
        out[ds] = {
            "n": len(group),
            "mean_cer": float(np.mean([r["cer"] for r in group])),
            "median_cer": float(np.median([r["cer"] for r in group])),
            "nonzero_cer_rate": float(np.mean([r["cer"] > 0 for r in group])),
        }

    return out


def compute_correlations(rows: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    cer = np.asarray([r["cer"] for r in rows], dtype=np.float64)

    results = []

    for name in feature_names:
        vals = np.asarray([safe_float(r["features"].get(name)) for r in rows], dtype=np.float64)

        if np.nanstd(vals) <= 1e-12:
            continue

        try:
            rho, p = spearmanr(vals, cer)
        except Exception:
            continue

        if not np.isfinite(rho):
            continue

        results.append({
            "feature": name,
            "spearman_r": float(rho),
            "abs_spearman_r": float(abs(rho)),
            "p_value": float(p) if np.isfinite(p) else None,
            "n": len(rows),
        })

    results.sort(key=lambda x: x["abs_spearman_r"], reverse=True)
    return results


def high_error_labels(cer: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(cer, quantile))

    if threshold <= 0.0:
        y = cer > 0.0
    else:
        y = cer >= threshold

    return y.astype(int), threshold


def evaluate_feature_set(
    rows: list[dict[str, Any]],
    features: list[str],
    *,
    high_error_quantile: float,
    min_n: int,
    min_pos: int,
) -> dict[str, Any] | None:
    if len(features) == 0:
        return None

    if len(rows) < min_n:
        return None

    cer = np.asarray([r["cer"] for r in rows], dtype=np.float64)
    y, threshold = high_error_labels(cer, high_error_quantile)

    pos = int(y.sum())
    neg = int(len(y) - pos)

    if pos < min_pos or neg < min_pos:
        return None

    X = np.asarray(
        [[safe_float(r["features"].get(f)) for f in features] for r in rows],
        dtype=np.float64,
    )

    n_splits = min(5, pos, neg)
    if n_splits < 2:
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=48,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=48)

    try:
        scores = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    except Exception as e:
        return {
            "error": repr(e),
            "n": len(rows),
            "feature_n": len(features),
            "positive_n": pos,
            "negative_n": neg,
            "high_error_threshold": threshold,
        }

    try:
        roc = float(roc_auc_score(y, scores))
    except Exception:
        roc = None

    try:
        pr = float(average_precision_score(y, scores))
    except Exception:
        pr = None

    base_rate = pos / len(y)
    top_k = max(1, int(math.ceil(0.20 * len(y))))
    order = np.argsort(-scores)
    top = order[:top_k]

    top20_precision = float(y[top].mean())
    top20_recall = float(y[top].sum() / max(pos, 1))

    return {
        "n": len(rows),
        "feature_n": len(features),
        "positive_n": pos,
        "negative_n": neg,
        "base_rate": float(base_rate),
        "high_error_quantile": high_error_quantile,
        "high_error_threshold": threshold,
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_auc_lift_over_base_rate": None if pr is None else float(pr / max(base_rate, 1e-12)),
        "top20_precision": top20_precision,
        "top20_recall": top20_recall,
    }


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {"global": rows}

    for key in ["dataset"]:
        by = defaultdict(list)
        for r in rows:
            by[str(r[key])].append(r)
        for k, v in by.items():
            groups[f"{key}:{k}"] = v

    by_combo = defaultdict(list)
    for r in rows:
        by_combo[f"{r['dataset']}|{r['level']}|{r['category']}"].append(r)

    for k, v in by_combo.items():
        groups[k] = v

    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--high_error_quantile", type=float, default=0.80)
    parser.add_argument("--min_n", type=int, default=250)
    parser.add_argument("--min_pos", type=int, default=30)
    parser.add_argument("--drop_text_len", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = load_manifest_features(Path(args.manifest), drop_text_len=args.drop_text_len)
    preds = load_predictions(Path(args.predictions))
    rows = join_rows(features, preds)

    if not rows:
        raise RuntimeError("No joined rows")

    feature_names = sorted(set.intersection(*(set(r["feature_names"]) for r in rows)))
    if args.drop_text_len and "text_len" in feature_names:
        feature_names.remove("text_len")

    fsets = feature_sets(feature_names)
    correlations = compute_correlations(rows, feature_names)

    grouped = group_rows(rows)
    high_error_results = []

    for group_name, group in sorted(grouped.items()):
        for set_name, names in fsets.items():
            res = evaluate_feature_set(
                group,
                names,
                high_error_quantile=args.high_error_quantile,
                min_n=args.min_n,
                min_pos=args.min_pos,
            )

            if res is None:
                continue

            res = {
                "group": group_name,
                "feature_set": set_name,
                **res,
            }

            high_error_results.append(res)

    valid_auc = [
        r for r in high_error_results
        if r.get("roc_auc") is not None and "error" not in r
    ]

    valid_auc.sort(
        key=lambda r: (
            r.get("roc_auc") or -1,
            r.get("pr_auc_lift_over_base_rate") or -1,
        ),
        reverse=True,
    )

    by_feature_set_best = {}
    for set_name in fsets:
        candidates = [r for r in valid_auc if r["feature_set"] == set_name]
        if candidates:
            by_feature_set_best[set_name] = candidates[0]

    summary = {
        "manifest": args.manifest,
        "predictions": args.predictions,
        "joined_n": len(rows),
        "feature_n": len(feature_names),
        "drop_text_len": args.drop_text_len,
        "feature_sets": {k: len(v) for k, v in fsets.items()},
        "dataset_summary": summarize_dataset(rows),
        "best_abs_spearman": correlations[0] if correlations else None,
        "top_correlations": correlations[:20],
        "best_high_error_results": valid_auc[:20],
        "best_by_feature_set": by_feature_set_best,
    }

    write_json(out_dir / "h3_after_school_foreground_v3_summary.json", summary)
    write_json(out_dir / "h3_after_school_foreground_v3_high_error_all.json", high_error_results)
    write_json(out_dir / "h3_after_school_foreground_v3_correlations.json", correlations)

    lines = []
    lines.append("# H3 after school foreground v3")
    lines.append("")
    lines.append("## 1. Input")
    lines.append("")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- predictions: `{args.predictions}`")
    lines.append(f"- joined n: {len(rows)}")
    lines.append(f"- feature n: {len(feature_names)}")
    lines.append("")
    lines.append("## 2. Dataset summary")
    lines.append("")
    lines.append("| dataset | n | mean CER | nonzero CER rate |")
    lines.append("|---|---:|---:|---:|")
    for ds, s in summary["dataset_summary"].items():
        lines.append(
            f"| `{ds}` | {s['n']} | {s['mean_cer']:.4f} | {s['nonzero_cer_rate']:.4f} |"
        )
    lines.append("")
    lines.append("## 3. Best correlation")
    lines.append("")
    if summary["best_abs_spearman"]:
        b = summary["best_abs_spearman"]
        lines.append(
            f"`{b['feature']}`: Spearman r={b['spearman_r']:.4f}, abs={b['abs_spearman_r']:.4f}, n={b['n']}"
        )
    lines.append("")
    lines.append("## 4. Best high-error detection by feature set")
    lines.append("")
    lines.append("| feature set | group | n | ROC-AUC | PR-AUC | lift | top20 precision |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for set_name, r in sorted(by_feature_set_best.items()):
        lines.append(
            f"| `{set_name}` | `{r['group']}` | {r['n']} | "
            f"{(r.get('roc_auc') or 0):.4f} | "
            f"{(r.get('pr_auc') or 0):.4f} | "
            f"{(r.get('pr_auc_lift_over_base_rate') or 0):.4f} | "
            f"{r.get('top20_precision', 0):.4f} |"
        )
    lines.append("")
    lines.append("## 5. Top high-error results")
    lines.append("")
    lines.append("| rank | feature set | group | n | ROC-AUC | PR-AUC | lift | top20 precision |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(valid_auc[:10], 1):
        lines.append(
            f"| {i} | `{r['feature_set']}` | `{r['group']}` | {r['n']} | "
            f"{(r.get('roc_auc') or 0):.4f} | "
            f"{(r.get('pr_auc') or 0):.4f} | "
            f"{(r.get('pr_auc_lift_over_base_rate') or 0):.4f} | "
            f"{r.get('top20_precision', 0):.4f} |"
        )

    (out_dir / "h3_after_school_foreground_v3_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()