from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.analyze_graph_quality_vs_cer_v1 import (
    join_predictions_and_features,
    load_manifest_features,
    load_predictions,
    safe_float,
    select_features,
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

ANNOTATION_COLUMNS = [
    "sample_id",
    "dataset",
    "level",
    "category",
    "audit_cell",
    "cer",
    "structural_risk_score",
    "image_path",
    "target",
    "pred",
    "annotator",
    "ink_visible_ok",
    "skeleton_follows_ink",
    "missed_visible_stroke",
    "spurious_stroke",
    "endpoint_error",
    "junction_error",
    "loop_error",
    "critical_topology_error",
    "graph_quality_0_3",
    "notes",
]


def is_structural_feature(name: str) -> bool:
    if name in GEOMETRY_FEATURES:
        return False
    n = name.lower()
    return any(k in n for k in STRUCTURAL_KEYWORDS)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def group_key(row: dict[str, Any], mode: str) -> str:
    dataset = str(row.get("dataset") or "unknown")
    level = str(row.get("level") or "unknown")
    category = str(row.get("category") or "unknown")

    if mode == "global":
        return "global"
    if mode == "dataset":
        return dataset
    if mode == "dataset_level":
        return f"{dataset}|{level}"
    if mode == "dataset_category":
        return f"{dataset}|{level}|{category}"

    raise ValueError(f"Unknown grouping mode: {mode}")


def rows_to_matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    X = []
    for r in rows:
        X.append([safe_float(r.get(f)) for f in features])
    return np.asarray(X, dtype=np.float64)


def add_group_thresholds(
    rows: list[dict[str, Any]],
    *,
    grouping_mode: str,
    high_error_quantile: float,
    low_error_quantile: float,
) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        groups[group_key(r, grouping_mode)].append(r)

    for key, group in groups.items():
        cers = np.asarray([float(r["cer"]) for r in group], dtype=np.float64)

        high_thr = float(np.quantile(cers, high_error_quantile))
        low_thr = float(np.quantile(cers, low_error_quantile))

        for r in group:
            r["audit_group"] = key
            r["high_error_threshold_cer"] = high_thr
            r["low_error_threshold_cer"] = low_thr
            r["is_high_error"] = bool(float(r["cer"]) >= high_thr)
            r["is_low_error"] = bool(float(r["cer"]) <= low_thr)


def fit_cv_structural_risk(
    rows: list[dict[str, Any]],
    *,
    features: list[str],
    grouping_mode: str,
    high_error_quantile: float,
    min_group_n: int,
    min_class_n: int,
) -> None:
    """
    Adds structural_risk_score to rows.

    The score is cross-validated probability of high-CER within each audit group.
    It is used only for candidate selection, not as an independent evaluation result.
    """
    for r in rows:
        r["structural_risk_score"] = None
        r["risk_model_group"] = None
        r["risk_model_mode"] = None

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r, grouping_mode)].append(r)

    fallback_rows = []

    for key, group in sorted(groups.items()):
        if len(group) < min_group_n:
            fallback_rows.extend(group)
            continue

        cers = np.asarray([float(r["cer"]) for r in group], dtype=np.float64)
        thr = float(np.quantile(cers, high_error_quantile))
        y = np.asarray([1 if float(r["cer"]) >= thr else 0 for r in group], dtype=np.int32)

        pos = int(y.sum())
        neg = int(len(y) - pos)

        if pos < min_class_n or neg < min_class_n:
            fallback_rows.extend(group)
            continue

        X = rows_to_matrix(group, features)
        n_splits = min(5, pos, neg)

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

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=53)

        score = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        for r, s in zip(group, score):
            r["structural_risk_score"] = float(s)
            r["risk_model_group"] = key
            r["risk_model_mode"] = grouping_mode

    # Fallback: global CV for rows that could not be scored in their subgroup.
    if fallback_rows:
        cers = np.asarray([float(r["cer"]) for r in rows], dtype=np.float64)
        thr = float(np.quantile(cers, high_error_quantile))
        y_all = np.asarray([1 if float(r["cer"]) >= thr else 0 for r in rows], dtype=np.int32)

        pos = int(y_all.sum())
        neg = int(len(y_all) - pos)

        if pos < min_class_n or neg < min_class_n:
            raise RuntimeError("Not enough classes for global fallback risk model.")

        X_all = rows_to_matrix(rows, features)

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

        cv = StratifiedKFold(
            n_splits=min(5, pos, neg),
            shuffle=True,
            random_state=54,
        )

        score_all = cross_val_predict(
            model,
            X_all,
            y_all,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        by_id = {r["sample_id"]: float(s) for r, s in zip(rows, score_all)}

        for r in fallback_rows:
            if r["structural_risk_score"] is None:
                r["structural_risk_score"] = by_id[r["sample_id"]]
                r["risk_model_group"] = "global_fallback"
                r["risk_model_mode"] = "global"


def add_risk_bins(
    rows: list[dict[str, Any]],
    *,
    grouping_mode: str,
    high_risk_quantile: float,
    low_risk_quantile: float,
) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        groups[group_key(r, grouping_mode)].append(r)

    for key, group in groups.items():
        scores = np.asarray([float(r["structural_risk_score"]) for r in group], dtype=np.float64)

        high_thr = float(np.quantile(scores, high_risk_quantile))
        low_thr = float(np.quantile(scores, low_risk_quantile))

        for r in group:
            s = float(r["structural_risk_score"])
            r["high_risk_threshold"] = high_thr
            r["low_risk_threshold"] = low_thr
            r["is_high_structural_risk"] = bool(s >= high_thr)
            r["is_low_structural_risk"] = bool(s <= low_thr)


def assign_audit_cells(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        he = bool(r["is_high_error"])
        le = bool(r["is_low_error"])
        hr = bool(r["is_high_structural_risk"])
        lr = bool(r["is_low_structural_risk"])

        cell = None

        if he and hr:
            cell = "A_highCER_highRisk"
        elif he and lr:
            cell = "B_highCER_lowRisk"
        elif le and hr:
            cell = "C_lowCER_highRisk"
        elif le and lr:
            cell = "D_lowCER_lowRisk"

        r["audit_cell"] = cell


def rank_for_cell(row: dict[str, Any]) -> tuple[float, float]:
    cer = float(row["cer"])
    risk = float(row["structural_risk_score"])
    cell = row["audit_cell"]

    if cell == "A_highCER_highRisk":
        return (-risk, -cer)
    if cell == "B_highCER_lowRisk":
        return (risk, -cer)
    if cell == "C_lowCER_highRisk":
        return (-risk, cer)
    if cell == "D_lowCER_lowRisk":
        return (risk, cer)

    return (0.0, 0.0)


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    per_cell: int,
    max_per_dataset_cell: int,
) -> list[dict[str, Any]]:
    cells = [
        "A_highCER_highRisk",
        "B_highCER_lowRisk",
        "C_lowCER_highRisk",
        "D_lowCER_lowRisk",
    ]

    selected = []
    used_ids = set()

    for cell in cells:
        pool = [r for r in rows if r.get("audit_cell") == cell]
        pool.sort(key=rank_for_cell)

        per_dataset_count: dict[str, int] = defaultdict(int)
        chosen = []

        for r in pool:
            sid = r["sample_id"]
            dataset = str(r.get("dataset") or "unknown")

            if sid in used_ids:
                continue

            if per_dataset_count[dataset] >= max_per_dataset_cell:
                continue

            chosen.append(r)
            used_ids.add(sid)
            per_dataset_count[dataset] += 1

            if len(chosen) >= per_cell:
                break

        # If dataset cap made the cell too small, fill remaining without cap.
        if len(chosen) < per_cell:
            for r in pool:
                sid = r["sample_id"]
                if sid in used_ids:
                    continue

                chosen.append(r)
                used_ids.add(sid)

                if len(chosen) >= per_cell:
                    break

        selected.extend(chosen)

    return selected


def slim_candidate(row: dict[str, Any], features: list[str]) -> dict[str, Any]:
    keep = {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "level": row.get("level"),
        "category": row.get("category"),
        "audit_group": row.get("audit_group"),
        "audit_cell": row.get("audit_cell"),
        "cer": row.get("cer"),
        "target": row.get("target"),
        "pred": row.get("pred"),
        "image_path": row.get("image_path"),
        "structural_risk_score": row.get("structural_risk_score"),
        "risk_model_group": row.get("risk_model_group"),
        "risk_model_mode": row.get("risk_model_mode"),
        "high_error_threshold_cer": row.get("high_error_threshold_cer"),
        "low_error_threshold_cer": row.get("low_error_threshold_cer"),
        "high_risk_threshold": row.get("high_risk_threshold"),
        "low_risk_threshold": row.get("low_risk_threshold"),
        "is_high_error": row.get("is_high_error"),
        "is_low_error": row.get("is_low_error"),
        "is_high_structural_risk": row.get("is_high_structural_risk"),
        "is_low_structural_risk": row.get("is_low_structural_risk"),
    }

    for f in features:
        if f in row:
            keep[f] = row[f]

    return keep


def annotation_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: "" for k in ANNOTATION_COLUMNS}

    for k in [
        "sample_id",
        "dataset",
        "level",
        "category",
        "audit_cell",
        "cer",
        "structural_risk_score",
        "image_path",
        "target",
        "pred",
    ]:
        out[k] = row.get(k, "")

    return out


def make_report(
    *,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    features: list[str],
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, int] = defaultdict(int)

    for r in selected:
        by_cell[str(r["audit_cell"])].append(r)
        by_dataset[str(r.get("dataset") or "unknown")] += 1

    lines = []
    lines.append("# H2 gold audit candidate pool — v1")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This candidate pool is designed for manual structural audit of visible-stroke graphs. "
        "It samples across CER and graph-structural-risk quadrants."
    )
    lines.append("")
    lines.append("## 2. Inputs")
    lines.append("")
    lines.append("```text")
    lines.append(f"manifest: {args.manifest}")
    lines.append(f"predictions: {args.predictions}")
    lines.append(f"joined samples: {len(rows)}")
    lines.append(f"structural features: {len(features)}")
    lines.append(f"grouping mode: {args.grouping_mode}")
    lines.append(f"per cell target: {args.per_cell}")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Selected candidates by cell")
    lines.append("")
    lines.append("| cell | n | mean CER | mean risk |")
    lines.append("|---|---:|---:|---:|")

    for cell in [
        "A_highCER_highRisk",
        "B_highCER_lowRisk",
        "C_lowCER_highRisk",
        "D_lowCER_lowRisk",
    ]:
        group = by_cell.get(cell, [])
        if group:
            mean_cer = float(np.mean([float(r["cer"]) for r in group]))
            mean_risk = float(np.mean([float(r["structural_risk_score"]) for r in group]))
        else:
            mean_cer = float("nan")
            mean_risk = float("nan")

        lines.append(f"| `{cell}` | {len(group)} | {mean_cer:.4f} | {mean_risk:.4f} |")

    lines.append("")
    lines.append("## 4. Selected candidates by dataset")
    lines.append("")
    lines.append("| dataset | n |")
    lines.append("|---|---:|")

    for dataset, n in sorted(by_dataset.items()):
        lines.append(f"| `{dataset}` | {n} |")

    lines.append("")
    lines.append("## 5. Manual annotation fields")
    lines.append("")
    lines.append("Use `annotation_template.csv`. Fill these fields:")
    lines.append("")
    lines.append("- `ink_visible_ok`: 0/1")
    lines.append("- `skeleton_follows_ink`: 0/1")
    lines.append("- `missed_visible_stroke`: 0/1")
    lines.append("- `spurious_stroke`: 0/1")
    lines.append("- `endpoint_error`: 0/1")
    lines.append("- `junction_error`: 0/1")
    lines.append("- `loop_error`: 0/1")
    lines.append("- `critical_topology_error`: 0/1")
    lines.append("- `graph_quality_0_3`: 0=bad, 1=weak, 2=usable, 3=good")
    lines.append("- `notes`: short free-text note")
    lines.append("")
    lines.append("## 6. Strict use")
    lines.append("")
    lines.append(
        "Do not use this pool to estimate population-level graph quality. "
        "It is deliberately biased toward informative cases. Use it to build the H2 rubric, "
        "failure taxonomy, and later a balanced gold subset."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--grouping_mode", default="dataset_level")
    parser.add_argument("--high_error_quantile", type=float, default=0.80)
    parser.add_argument("--low_error_quantile", type=float, default=0.20)
    parser.add_argument("--high_risk_quantile", type=float, default=0.80)
    parser.add_argument("--low_risk_quantile", type=float, default=0.20)

    parser.add_argument("--per_cell", type=int, default=25)
    parser.add_argument("--max_per_dataset_cell", type=int, default=10)
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

    all_features = select_features(rows)
    structural_features = [f for f in all_features if is_structural_feature(f)]

    if not structural_features:
        raise RuntimeError("No structural features found.")

    add_group_thresholds(
        rows,
        grouping_mode=args.grouping_mode,
        high_error_quantile=args.high_error_quantile,
        low_error_quantile=args.low_error_quantile,
    )

    fit_cv_structural_risk(
        rows,
        features=structural_features,
        grouping_mode=args.grouping_mode,
        high_error_quantile=args.high_error_quantile,
        min_group_n=args.min_group_n,
        min_class_n=args.min_class_n,
    )

    add_risk_bins(
        rows,
        grouping_mode=args.grouping_mode,
        high_risk_quantile=args.high_risk_quantile,
        low_risk_quantile=args.low_risk_quantile,
    )

    assign_audit_cells(rows)

    selected = select_candidates(
        rows,
        per_cell=args.per_cell,
        max_per_dataset_cell=args.max_per_dataset_cell,
    )

    selected_slim = [slim_candidate(r, structural_features) for r in selected]
    annotation = [annotation_row(r) for r in selected_slim]

    write_jsonl(selected_slim, out_dir / "h2_audit_candidates.jsonl")
    write_csv(selected_slim, out_dir / "h2_audit_candidates.csv")
    write_csv(annotation, out_dir / "annotation_template.csv", fieldnames=ANNOTATION_COLUMNS)

    cell_counts = defaultdict(int)
    for r in selected_slim:
        cell_counts[str(r["audit_cell"])] += 1

    summary = {
        "manifest": args.manifest,
        "predictions": args.predictions,
        "joined_n": len(rows),
        "selected_n": len(selected_slim),
        "structural_feature_n": len(structural_features),
        "structural_features": structural_features,
        "grouping_mode": args.grouping_mode,
        "high_error_quantile": args.high_error_quantile,
        "low_error_quantile": args.low_error_quantile,
        "high_risk_quantile": args.high_risk_quantile,
        "low_risk_quantile": args.low_risk_quantile,
        "per_cell": args.per_cell,
        "max_per_dataset_cell": args.max_per_dataset_cell,
        "cell_counts": dict(sorted(cell_counts.items())),
        "outputs": {
            "candidates_jsonl": str(out_dir / "h2_audit_candidates.jsonl"),
            "candidates_csv": str(out_dir / "h2_audit_candidates.csv"),
            "annotation_template_csv": str(out_dir / "annotation_template.csv"),
            "report_md": str(out_dir / "h2_gold_audit_candidate_report_v1.md"),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_report(
        selected=selected_slim,
        rows=rows,
        features=structural_features,
        out_path=out_dir / "h2_gold_audit_candidate_report_v1.md",
        args=args,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()