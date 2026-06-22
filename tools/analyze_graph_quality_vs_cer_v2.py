from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.analyze_graph_quality_vs_cer_v1 import (
    correlation_rows,
    join_predictions_and_features,
    load_manifest_features,
    load_predictions,
    safe_float,
    select_features,
    write_csv,
)


GEOMETRY_FEATURES = {
    "width",
    "height",
    "aspect_ratio",
    "bbox_x0_frac",
    "bbox_y0_frac",
    "bbox_w_frac",
    "bbox_h_frac",
    "bbox_area_frac",
}

STRUCTURAL_KEYWORDS = [
    "fg_",
    "cc_",
    "skel",
    "graph_",
    "endpoint",
    "branchpoint",
    "degree_hist",
    "dir_",
    "stroke_width",
]

QUALITY_KEYWORDS = [
    "warning",
    "quality",
    "confidence",
    "risk",
]


def is_quality_feature(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in QUALITY_KEYWORDS)


def is_structural_feature(name: str) -> bool:
    n = name.lower()
    if name in GEOMETRY_FEATURES:
        return False
    return any(k in n for k in STRUCTURAL_KEYWORDS)


def make_feature_sets(features: list[str]) -> dict[str, list[str]]:
    quality = [f for f in features if is_quality_feature(f)]
    structural = [f for f in features if is_structural_feature(f)]
    geometry = [f for f in features if f in GEOMETRY_FEATURES]
    non_geometry = [f for f in features if f not in GEOMETRY_FEATURES]

    return {
        "quality_only": quality,
        "structural_core": structural,
        "geometry_control": geometry,
        "all_non_geometry": non_geometry,
        "all_features": features,
    }


def group_rows(rows: list[dict[str, Any]], mode: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    if mode == "global":
        groups["global"] = list(rows)
        return groups

    if mode == "dataset":
        for r in rows:
            groups[str(r.get("dataset") or "unknown")].append(r)
        return groups

    if mode == "dataset_level":
        for r in rows:
            key = f"{r.get('dataset') or 'unknown'}|{r.get('level') or 'unknown'}"
            groups[key].append(r)
        return groups

    if mode == "dataset_category":
        for r in rows:
            key = (
                f"{r.get('dataset') or 'unknown'}|"
                f"{r.get('level') or 'unknown'}|"
                f"{r.get('category') or 'unknown'}"
            )
            groups[key].append(r)
        return groups

    raise ValueError(f"Unknown group mode: {mode}")


def rows_to_xy(
    rows: list[dict[str, Any]],
    features: list[str],
    high_error_quantile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    cers = np.asarray([float(r["cer"]) for r in rows], dtype=np.float64)
    threshold = float(np.quantile(cers, high_error_quantile))
    y = np.asarray([1 if float(r["cer"]) >= threshold else 0 for r in rows], dtype=np.int32)

    x_rows = []
    for r in rows:
        x_rows.append([safe_float(r.get(f)) for f in features])

    X = np.asarray(x_rows, dtype=np.float64)
    return X, y, threshold


def top20_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    threshold = float(np.quantile(score, 0.80))
    pred = (score >= threshold).astype(np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0,
    )

    return {
        "top20_precision": float(precision),
        "top20_recall": float(recall),
        "top20_f1": float(f1),
    }


def cv_high_error_detection(
    *,
    rows: list[dict[str, Any]],
    group_name: str,
    group_mode: str,
    feature_set_name: str,
    features: list[str],
    high_error_quantile: float,
    min_class_n: int,
) -> dict[str, Any] | None:
    if not features:
        return None

    X, y, cer_threshold = rows_to_xy(rows, features, high_error_quantile)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    if n_pos < min_class_n or n_neg < min_class_n:
        return None

    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        return None

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=52)

    score = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    roc_auc = float(roc_auc_score(y, score))
    pr_auc = float(average_precision_score(y, score))
    top = top20_metrics(y, score)

    return {
        "group_mode": group_mode,
        "group": group_name,
        "feature_set": feature_set_name,
        "n": int(len(y)),
        "positive_n": n_pos,
        "negative_n": n_neg,
        "high_error_rate": float(np.mean(y)),
        "high_error_quantile": high_error_quantile,
        "high_error_threshold_cer": cer_threshold,
        "feature_n": len(features),
        "features": ";".join(features),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pr_auc_lift_over_base_rate": pr_auc / max(float(np.mean(y)), 1e-12),
        **top,
    }


def stratified_correlations(
    *,
    rows: list[dict[str, Any]],
    features: list[str],
    group_modes: list[str],
    min_group_n: int,
) -> list[dict[str, Any]]:
    out = []

    for mode in group_modes:
        groups = group_rows(rows, mode)
        for group_name, group in sorted(groups.items()):
            if len(group) < min_group_n:
                continue

            corr = correlation_rows(group, features)
            for r in corr:
                rr = dict(r)
                rr["group_mode"] = mode
                rr["group"] = group_name
                rr["group_n"] = len(group)
                out.append(rr)

    out.sort(
        key=lambda r: (
            r["group_mode"],
            r["group"],
            -float(r["abs_spearman_r"]),
            r["feature"],
        )
    )
    return out


def multifeature_cv_rows(
    *,
    rows: list[dict[str, Any]],
    feature_sets: dict[str, list[str]],
    group_modes: list[str],
    high_error_quantile: float,
    min_group_n: int,
    min_class_n: int,
) -> list[dict[str, Any]]:
    out = []

    for mode in group_modes:
        groups = group_rows(rows, mode)

        for group_name, group in sorted(groups.items()):
            if len(group) < min_group_n:
                continue

            for fs_name, fs_features in feature_sets.items():
                result = cv_high_error_detection(
                    rows=group,
                    group_name=group_name,
                    group_mode=mode,
                    feature_set_name=fs_name,
                    features=fs_features,
                    high_error_quantile=high_error_quantile,
                    min_class_n=min_class_n,
                )

                if result is not None:
                    out.append(result)

    out.sort(key=lambda r: (-float(r["roc_auc"]), -float(r["pr_auc"]), r["group_mode"], r["group"]))
    return out


def make_summary(
    *,
    rows: list[dict[str, Any]],
    features: list[str],
    feature_sets: dict[str, list[str]],
    cv_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    best_cv_by_set: dict[str, Any] = {}

    for row in cv_rows:
        fs = row["feature_set"]
        if fs not in best_cv_by_set:
            best_cv_by_set[fs] = row

    best_corr_global = [
        r for r in corr_rows
        if r["group_mode"] == "global" and r["group"] == "global"
    ]

    return {
        "manifest": args.manifest,
        "predictions": args.predictions,
        "joined_n": len(rows),
        "feature_n": len(features),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "high_error_quantile": args.high_error_quantile,
        "min_group_n": args.min_group_n,
        "min_class_n": args.min_class_n,
        "best_cv_by_feature_set": best_cv_by_set,
        "best_global_correlation": best_corr_global[0] if best_corr_global else None,
        "strict_h3_interpretation": interpret_h3(cv_rows, corr_rows),
    }


def interpret_h3(cv_rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]]) -> str:
    structural = [
        r for r in cv_rows
        if r["feature_set"] in {"structural_core", "quality_only", "all_non_geometry"}
        and r["group_mode"] in {"global", "dataset"}
    ]

    best_auc = max([float(r["roc_auc"]) for r in structural], default=0.0)

    global_corr = [
        abs(float(r["spearman_r"]))
        for r in corr_rows
        if r["group_mode"] == "global" and r["group"] == "global"
    ]
    best_corr = max(global_corr, default=0.0)

    if best_auc >= 0.75:
        return "strong_multifeature_h3_signal"
    if best_auc >= 0.65:
        return "useful_multifeature_h3_signal"
    if best_corr >= 0.20:
        return "weak_to_moderate_correlation_h3_signal"
    return "weak_or_no_h3_signal_for_current_features"


def make_report_md(path: Path, summary: dict[str, Any], cv_rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("# H3 graph diagnostics report — v2")
    lines.append("")
    lines.append("## 1. Strict interpretation")
    lines.append("")
    lines.append(f"```text\n{summary['strict_h3_interpretation']}\n```")
    lines.append("")
    lines.append("This v2 analysis adds stratification and multifeature cross-validated high-error detection.")
    lines.append("")
    lines.append("## 2. Best multifeature high-error detectors")
    lines.append("")
    lines.append("| group mode | group | feature set | n | ROC-AUC | PR-AUC | PR-AUC lift | top20 precision |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")

    for r in cv_rows[:30]:
        lines.append(
            f"| `{r['group_mode']}` | `{r['group']}` | `{r['feature_set']}` | "
            f"{r['n']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['pr_auc_lift_over_base_rate']:.3f} | {r['top20_precision']:.4f} |"
        )

    lines.append("")
    lines.append("## 3. Best stratified correlations")
    lines.append("")
    lines.append("| group mode | group | feature | n | Spearman r | abs r |")
    lines.append("|---|---|---|---:|---:|---:|")

    for r in sorted(corr_rows, key=lambda x: -float(x["abs_spearman_r"]))[:30]:
        lines.append(
            f"| `{r['group_mode']}` | `{r['group']}` | `{r['feature']}` | "
            f"{r['n']} | {r['spearman_r']:.4f} | {r['abs_spearman_r']:.4f} |"
        )

    lines.append("")
    lines.append("## 4. Methodological note")
    lines.append("")
    lines.append(
        "Geometry-control features are reported as controls. If geometry features dominate, "
        "that is not strong evidence for graph quality. The primary evidence should come "
        "from `quality_only`, `structural_core`, or `all_non_geometry` feature sets."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--high_error_quantile", type=float, default=0.80)
    parser.add_argument("--min_group_n", type=int, default=250)
    parser.add_argument("--min_class_n", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_features(Path(args.manifest))
    predictions = load_predictions(Path(args.predictions))
    rows = join_predictions_and_features(
        manifest_features=manifest,
        predictions=predictions,
    )

    if not rows:
        raise RuntimeError("No joined rows. sample_id mismatch likely.")

    features = select_features(rows)
    feature_sets = make_feature_sets(features)

    group_modes = ["global", "dataset", "dataset_level", "dataset_category"]

    corr_rows = stratified_correlations(
        rows=rows,
        features=features,
        group_modes=group_modes,
        min_group_n=args.min_group_n,
    )

    cv_rows = multifeature_cv_rows(
        rows=rows,
        feature_sets=feature_sets,
        group_modes=group_modes,
        high_error_quantile=args.high_error_quantile,
        min_group_n=args.min_group_n,
        min_class_n=args.min_class_n,
    )

    write_csv(corr_rows, out_dir / "stratified_feature_cer_correlations.csv")
    write_csv(cv_rows, out_dir / "multifeature_high_error_cv.csv")

    summary = make_summary(
        rows=rows,
        features=features,
        feature_sets=feature_sets,
        cv_rows=cv_rows,
        corr_rows=corr_rows,
        args=args,
    )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_report_md(
        out_dir / "h3_graph_quality_vs_cer_report_v2.md",
        summary=summary,
        cv_rows=cv_rows,
        corr_rows=corr_rows,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()