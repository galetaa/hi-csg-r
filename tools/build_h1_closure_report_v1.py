from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ["image_only", "graph_vector_v2", "gated_v2_dist"]
GRAPH_MODELS = ["graph_vector_v2", "gated_v2_dist"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def cer(target: str, pred: str) -> float:
    return edit_distance(list(target), list(pred)) / max(len(target), 1)


def aggregate_cer(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    if not rows:
        return None
    edits = 0
    total = 0
    for row in rows:
        target = str(row["target"])
        pred = str(row[pred_key])
        edits += edit_distance(list(target), list(pred))
        total += max(len(target), 1)
    return edits / max(total, 1)


def distortion(condition: str) -> str:
    if condition == "clean":
        return "clean"
    for name in ["low_contrast", "thin_strokes", "thick_strokes", "blur", "noise"]:
        if condition.startswith(name):
            return name
    return condition.rsplit("_", 1)[0]


def distortion_group(condition: str) -> str:
    d = distortion(condition)
    if d in {"blur", "noise", "low_contrast"}:
        return "visual"
    if d in {"thin_strokes", "thick_strokes"}:
        return "structural"
    return d


def graph_score(row: dict[str, Any]) -> float:
    for key in ("graph_quality_score", "graph_confidence", "graph_quality", "quality_score"):
        if key in row:
            return float(row[key])
    if "graph_warning_count" in row:
        return -float(row["graph_warning_count"])
    if "warning_count" in row:
        return -float(row["warning_count"])
    feats = row.get("graph_features")
    names = row.get("graph_feature_names")
    if isinstance(feats, list) and isinstance(names, list):
        values = {str(name): float(value) for name, value in zip(names, feats)}
        return -float(values.get("warning_count", 0.0) + values.get("short_branch_ratio", 0.0))
    raise KeyError("No graph quality proxy found")


def assign_strata(manifest_rows: list[dict[str, Any]]) -> dict[str, str]:
    scored = [(graph_score(row), str(row["sample_id"])) for row in manifest_rows]
    scored.sort(key=lambda item: item[0])
    n = len(scored)
    lo = n // 3
    hi = 2 * n // 3
    out = {}
    for _, sample_id in scored[:lo]:
        out[sample_id] = "low"
    for _, sample_id in scored[lo:hi]:
        out[sample_id] = "medium"
    for _, sample_id in scored[hi:]:
        out[sample_id] = "high"
    return out


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        row = dict(row)
        sample_id = str(row["sample_id"])
        row["target"] = str(row.get("target", row.get("text", "")))
        row["pred"] = str(row.get("pred", ""))
        out[sample_id] = row
    return out


def bootstrap_delta(
    deltas: list[float],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    if not deltas:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robustness_root", default="outputs/robustness_v1")
    parser.add_argument("--robustness_data_root", default="data/experiments/robustness_v1/tri10k_mixed_test")
    parser.add_argument("--out_dir", default="outputs/robustness_v1/h1_closure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    args = parser.parse_args()

    root = Path(args.robustness_root)
    data_root = Path(args.robustness_data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["clean"] + [
        line.strip()
        for line in (data_root / "conditions.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    summary_by_model = {}
    degradation_rows = []
    for model in MODELS:
        clean = read_json(root / model / "clean" / "summary.json")
        clean_cer = float(clean["metrics"]["cer"] if "metrics" in clean else clean["cer"])
        distorted = []
        for condition in conditions:
            obj = read_json(root / model / condition / "summary.json")
            metrics = obj["metrics"] if "metrics" in obj else obj
            row = {
                "model": model,
                "condition": condition,
                "distortion": distortion(condition),
                "distortion_group": distortion_group(condition),
                "cer": float(metrics["cer"]),
                "wer": float(metrics["wer"]),
                "exact": float(metrics["exact"]),
                "clean_cer": clean_cer,
                "absolute_cer_delta": float(metrics["cer"]) - clean_cer,
                "relative_cer_degradation": (float(metrics["cer"]) - clean_cer) / clean_cer if clean_cer else 0.0,
            }
            if condition != "clean":
                distorted.append(row)
            degradation_rows.append(row)
        visual = [row for row in distorted if row["distortion_group"] == "visual"]
        structural = [row for row in distorted if row["distortion_group"] == "structural"]
        summary_by_model[model] = {
            "clean_cer": clean_cer,
            "mean_distorted_cer": float(np.mean([row["cer"] for row in distorted])),
            "mean_absolute_delta_cer": float(np.mean([row["absolute_cer_delta"] for row in distorted])),
            "mean_relative_degradation": float(np.mean([row["relative_cer_degradation"] for row in distorted])),
            "visual_relative_degradation": float(np.mean([row["relative_cer_degradation"] for row in visual])),
            "structural_relative_degradation": float(np.mean([row["relative_cer_degradation"] for row in structural])),
        }

    condition_rows = []
    for condition in conditions:
        row = {
            "condition": condition,
            "distortion": distortion(condition),
            "distortion_group": distortion_group(condition),
        }
        for model in MODELS:
            obj = read_json(root / model / condition / "summary.json")
            metrics = obj["metrics"] if "metrics" in obj else obj
            row[f"{model}_cer"] = float(metrics["cer"])
        best = min(MODELS, key=lambda name: row[f"{name}_cer"])
        row["best_model"] = best
        row["graph_vector_minus_image_only"] = row["graph_vector_v2_cer"] - row["image_only_cer"]
        row["gated_minus_image_only"] = row["gated_v2_dist_cer"] - row["image_only_cer"]
        condition_rows.append(row)

    strata_rows = []
    bootstrap_rows = []
    for condition in conditions:
        if condition == "clean":
            manifest_path = data_root / "graph_ready" / "clean.jsonl"
            if not manifest_path.exists():
                manifest_path = Path("data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/test.jsonl")
        else:
            manifest_path = data_root / "graph_ready" / f"{condition}.jsonl"
        manifest_rows = read_jsonl(manifest_path)
        stratum_by_id = assign_strata(manifest_rows)

        predictions = {
            model: read_predictions(root / model / condition / "predictions.jsonl")
            for model in MODELS
        }
        common = sorted(set.intersection(*(set(p) for p in predictions.values())))

        for graph_model in GRAPH_MODELS:
            deltas = []
            wins = losses = ties = 0
            for sample_id in common:
                base = predictions["image_only"][sample_id]
                cand = predictions[graph_model][sample_id]
                target = str(base["target"])
                base_cer = cer(target, str(base["pred"]))
                cand_cer = cer(target, str(cand["pred"]))
                delta = cand_cer - base_cer
                deltas.append(delta)
                if delta < 0:
                    wins += 1
                elif delta > 0:
                    losses += 1
                else:
                    ties += 1
            boot = bootstrap_delta(
                deltas,
                seed=args.seed + len(bootstrap_rows),
                n_bootstrap=args.n_bootstrap,
            )
            bootstrap_rows.append(
                {
                    "condition": condition,
                    "model": graph_model,
                    "n": len(common),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "mean_delta_cer": boot["mean"],
                    "ci95_low": boot["ci95_low"],
                    "ci95_high": boot["ci95_high"],
                }
            )

        for stratum in ["low", "medium", "high"]:
            sample_ids = [sample_id for sample_id in common if stratum_by_id.get(sample_id) == stratum]
            joined = []
            for sample_id in sample_ids:
                target = str(predictions["image_only"][sample_id]["target"])
                joined.append(
                    {
                        "target": target,
                        "image_only_pred": predictions["image_only"][sample_id]["pred"],
                        "graph_vector_pred": predictions["graph_vector_v2"][sample_id]["pred"],
                        "gated_pred": predictions["gated_v2_dist"][sample_id]["pred"],
                    }
                )
            image_cer = aggregate_cer(
                [{"target": row["target"], "pred": row["image_only_pred"]} for row in joined],
                "pred",
            )
            graph_cer = aggregate_cer(
                [{"target": row["target"], "pred": row["graph_vector_pred"]} for row in joined],
                "pred",
            )
            gated_cer = aggregate_cer(
                [{"target": row["target"], "pred": row["gated_pred"]} for row in joined],
                "pred",
            )
            strata_rows.append(
                {
                    "condition": condition,
                    "stratum": stratum,
                    "n": len(sample_ids),
                    "image_only_cer": image_cer,
                    "graph_vector_v2_cer": graph_cer,
                    "gated_v2_dist_cer": gated_cer,
                    "graph_vector_delta_vs_image_only": graph_cer - image_cer if graph_cer is not None and image_cer is not None else None,
                    "gated_delta_vs_image_only": gated_cer - image_cer if gated_cer is not None and image_cer is not None else None,
                }
            )

    gate_rows = []
    for condition in conditions:
        obj = read_json(root / "gated_v2_dist" / condition / "summary.json")
        metrics = obj["metrics"] if "metrics" in obj else obj
        gate_rows.append(
            {
                "condition": condition,
                "distortion": distortion(condition),
                "distortion_group": distortion_group(condition),
                "gate_mean": float(metrics.get("gate_mean", 0.0)),
                "gate_median": None,
                "gate_p10": None,
                "gate_p90": None,
                "gate_max": None,
                "note": "Existing gated predictions store only condition-level gate_mean, not per-sample/pixel gate distribution.",
            }
        )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_dir / "model_degradation_summary.csv", [
        {"model": model, **values}
        for model, values in summary_by_model.items()
    ])
    write_csv(out_dir / "condition_model_comparison.csv", condition_rows)
    write_csv(out_dir / "strata_model_comparison.csv", strata_rows)
    write_csv(out_dir / "paired_bootstrap_by_condition.csv", bootstrap_rows)
    write_csv(out_dir / "gate_distribution_summary.csv", gate_rows)

    result = {
        "model_degradation_summary": summary_by_model,
        "condition_model_comparison": condition_rows,
        "paired_bootstrap_by_condition": bootstrap_rows,
        "strata_model_comparison": strata_rows,
        "gate_distribution_summary": gate_rows,
        "interpretation": {
            "h1_status": "not_confirmed_current_implementation",
            "reason": (
                "Graph-aware models have worse clean CER and remain worse than image-only "
                "on most distorted conditions; strata comparison does not establish a "
                "high-quality-graph practical win; gated v2 stores only low condition-level "
                "gate means and does not show systematic graph-branch activation."
            ),
        },
    }
    write_json(out_dir / "h1_closure_summary.json", result)

    lines = [
        "# H1 Closure Report",
        "",
        "## Model Degradation Summary",
        "",
        "| model | clean CER | mean distorted CER | mean absolute ΔCER | mean relative degradation | visual rel deg | structural rel deg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in summary_by_model.items():
        lines.append(
            f"| `{model}` | {fmt(row['clean_cer'])} | {fmt(row['mean_distorted_cer'])} | "
            f"{fmt(row['mean_absolute_delta_cer'])} | {fmt(row['mean_relative_degradation'])} | "
            f"{fmt(row['visual_relative_degradation'])} | {fmt(row['structural_relative_degradation'])} |"
        )

    lines.extend([
        "",
        "## Condition Comparison",
        "",
        "| condition | image-only CER | graph-vector CER | gated CER | best model | graph-vector - image-only | gated - image-only |",
        "|---|---:|---:|---:|---|---:|---:|",
    ])
    for row in condition_rows:
        lines.append(
            f"| `{row['condition']}` | {fmt(row['image_only_cer'])} | {fmt(row['graph_vector_v2_cer'])} | "
            f"{fmt(row['gated_v2_dist_cer'])} | `{row['best_model']}` | "
            f"{fmt(row['graph_vector_minus_image_only'])} | {fmt(row['gated_minus_image_only'])} |"
        )

    lines.extend([
        "",
        "## Paired Sign / Bootstrap",
        "",
        "| condition | model | wins | losses | ties | mean ΔCER | CI95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in bootstrap_rows:
        lines.append(
            f"| `{row['condition']}` | `{row['model']}` | {row['wins']} | {row['losses']} | {row['ties']} | "
            f"{fmt(row['mean_delta_cer'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] |"
        )

    lines.extend([
        "",
        "## Gate Distribution",
        "",
        "Existing gated outputs store only condition-level `gate_mean`; median/p10/p90/max are unavailable without re-running eval with per-sample or per-pixel gate logging.",
        "",
        "| condition | gate mean | gate median | gate p10 | gate p90 | gate max |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in gate_rows:
        lines.append(
            f"| `{row['condition']}` | {fmt(row['gate_mean'])} | n/a | n/a | n/a | n/a |"
        )

    lines.extend([
        "",
        "## Conclusion",
        "",
        "H1 is not confirmed for the current implementation. The image-only model remains the practical winner across clean and distorted conditions. Graph-aware variants sometimes show lower relative degradation because their clean CER is already much worse, but they do not provide practical distorted-CER wins. Gated v2 also does not show evidence of systematic graph-branch activation.",
        "",
    ])
    (out_dir / "h1_closure_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "files": [
            "h1_closure_summary.json",
            "h1_closure_summary.md",
            "model_degradation_summary.csv",
            "condition_model_comparison.csv",
            "strata_model_comparison.csv",
            "paired_bootstrap_by_condition.csv",
            "gate_distribution_summary.csv",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
