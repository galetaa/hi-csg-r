from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Supports both formats:
    1) {"metrics": {"cer": ...}}
    2) {"cer": ...}
    """
    if key in summary:
        return summary[key]

    m = summary.get("metrics")
    if isinstance(m, dict) and key in m:
        return m[key]

    if default is not None:
        return default

    raise KeyError(f"Metric {key!r} not found. Top keys: {sorted(summary.keys())}")


def as_float(x: Any, default: float | None = None) -> float | None:
    if x is None:
        return default
    return float(x)


def parse_condition(condition: str) -> tuple[str, str]:
    if condition == "clean":
        return "clean", "clean"

    known_levels = {"mild", "medium", "strong"}
    parts = condition.split("_")

    if parts[-1] in known_levels:
        return "_".join(parts[:-1]), parts[-1]

    return condition, "unknown"


def read_model_rows(model: str, model_dir: Path) -> list[dict[str, Any]]:
    clean_path = model_dir / "clean" / "summary.json"
    clean_summary = load_json(clean_path)

    clean_cer = float(metric(clean_summary, "cer"))
    clean_wer = as_float(metric(clean_summary, "wer", None))
    clean_exact = as_float(metric(clean_summary, "exact", None))
    clean_n = int(metric(clean_summary, "n"))

    rows: list[dict[str, Any]] = []

    condition_dirs = [p for p in sorted(model_dir.iterdir()) if p.is_dir()]

    for cond_dir in condition_dirs:
        summary_path = cond_dir / "summary.json"
        if not summary_path.exists():
            continue

        condition = cond_dir.name
        summary = load_json(summary_path)

        cer = float(metric(summary, "cer"))
        wer = as_float(metric(summary, "wer", None))
        exact = as_float(metric(summary, "exact", None))
        n = int(metric(summary, "n"))

        distortion, level = parse_condition(condition)

        rows.append(
            {
                "model": model,
                "condition": condition,
                "distortion": distortion,
                "level": level,
                "n": n,
                "cer": cer,
                "wer": wer,
                "exact": exact,
                "clean_n": clean_n,
                "clean_cer": clean_cer,
                "clean_wer": clean_wer,
                "clean_exact": clean_exact,
                "absolute_cer_delta": cer - clean_cer,
                "relative_cer_degradation": (cer - clean_cer) / max(clean_cer, 1e-12),
                "summary_path": str(summary_path),
            }
        )

    if not any(r["condition"] == "clean" for r in rows):
        raise RuntimeError(f"No clean summary found for model={model} at {clean_path}")

    return rows


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "condition",
        "distortion",
        "level",
        "n",
        "cer",
        "wer",
        "exact",
        "clean_cer",
        "absolute_cer_delta",
        "relative_cer_degradation",
        "robustness_advantage_vs_image_only",
        "summary_path",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    for model, mrows in sorted(by_model.items()):
        distorted = [r for r in mrows if r["condition"] != "clean"]

        out[model] = {
            "clean": next(r for r in mrows if r["condition"] == "clean"),
            "num_distorted_conditions": len(distorted),
            "mean_distorted_cer": mean([float(r["cer"]) for r in distorted]),
            "mean_absolute_cer_delta": mean([float(r["absolute_cer_delta"]) for r in distorted]),
            "mean_relative_cer_degradation": mean(
                [float(r["relative_cer_degradation"]) for r in distorted]
            ),
            "by_distortion": {},
            "by_level": {},
        }

        by_distortion: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for r in distorted:
            by_distortion[r["distortion"]].append(r)
            by_level[r["level"]].append(r)

        for distortion, drows in sorted(by_distortion.items()):
            out[model]["by_distortion"][distortion] = {
                "conditions": [r["condition"] for r in drows],
                "mean_cer": mean([float(r["cer"]) for r in drows]),
                "mean_relative_cer_degradation": mean(
                    [float(r["relative_cer_degradation"]) for r in drows]
                ),
                "mean_robustness_advantage_vs_image_only": mean(
                    [
                        float(r["robustness_advantage_vs_image_only"])
                        for r in drows
                        if r.get("robustness_advantage_vs_image_only") is not None
                    ]
                ),
            }

        for level, lrows in sorted(by_level.items()):
            out[model]["by_level"][level] = {
                "conditions": [r["condition"] for r in lrows],
                "mean_cer": mean([float(r["cer"]) for r in lrows]),
                "mean_relative_cer_degradation": mean(
                    [float(r["relative_cer_degradation"]) for r in lrows]
                ),
                "mean_robustness_advantage_vs_image_only": mean(
                    [
                        float(r["robustness_advantage_vs_image_only"])
                        for r in lrows
                        if r.get("robustness_advantage_vs_image_only") is not None
                    ]
                ),
            }

    return out


def add_image_only_advantage(rows: list[dict[str, Any]]) -> None:
    """
    robustness_advantage_vs_image_only:
      positive  => model degraded less than image-only
      zero      => same degradation
      negative  => model degraded more than image-only
    """
    image_only_by_condition = {
        r["condition"]: float(r["relative_cer_degradation"])
        for r in rows
        if r["model"] == "image_only"
    }

    for r in rows:
        if r["condition"] == "clean":
            r["robustness_advantage_vs_image_only"] = 0.0
            continue

        base = image_only_by_condition.get(r["condition"])
        if base is None:
            r["robustness_advantage_vs_image_only"] = None
        else:
            r["robustness_advantage_vs_image_only"] = (
                base - float(r["relative_cer_degradation"])
            )


def make_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "criterion": (
            "H1 is supported only if a graph-aware model has lower relative CER "
            "degradation than image-only across multiple distortion families, "
            "not just one isolated condition."
        ),
        "models": {},
    }

    image = summary.get("image_only", {})
    image_mean = image.get("mean_relative_cer_degradation")

    for model in ["graph_vector_v2", "gated_v2_dist"]:
        m = summary.get(model)
        if not m:
            continue

        mean_deg = m.get("mean_relative_cer_degradation")
        clean_cer = m.get("clean", {}).get("cer")
        clean_image_cer = image.get("clean", {}).get("cer")

        by_dist = m.get("by_distortion", {})
        better_distortions = []
        worse_distortions = []

        for dist, vals in by_dist.items():
            adv = vals.get("mean_robustness_advantage_vs_image_only")
            if adv is None:
                continue
            if adv > 0:
                better_distortions.append(dist)
            elif adv < 0:
                worse_distortions.append(dist)

        verdict["models"][model] = {
            "clean_cer": clean_cer,
            "image_only_clean_cer": clean_image_cer,
            "mean_relative_cer_degradation": mean_deg,
            "image_only_mean_relative_cer_degradation": image_mean,
            "mean_degradation_advantage_vs_image_only": (
                image_mean - mean_deg
                if image_mean is not None and mean_deg is not None
                else None
            ),
            "better_distortion_families_than_image_only": better_distortions,
            "worse_distortion_families_than_image_only": worse_distortions,
            "strict_interpretation": (
                "potential_partial_H1_support"
                if len(better_distortions) >= 2
                else "no_H1_support_or_too_narrow"
            ),
        }

    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_only_dir", required=True)
    parser.add_argument("--graph_vector_dir", required=True)
    parser.add_argument("--gated_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    specs = [
        ("image_only", Path(args.image_only_dir)),
        ("graph_vector_v2", Path(args.graph_vector_dir)),
        ("gated_v2_dist", Path(args.gated_dir)),
    ]

    rows: list[dict[str, Any]] = []
    for model, model_dir in specs:
        model_rows = read_model_rows(model, model_dir)
        rows.extend(model_rows)

    add_image_only_advantage(rows)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    write_csv(rows, out_csv)

    summary = {
        "num_rows": len(rows),
        "models": group_summary(rows),
        "verdict": {},
        "definitions": {
            "absolute_cer_delta": "CER(condition) - CER(clean)",
            "relative_cer_degradation": "(CER(condition) - CER(clean)) / CER(clean)",
            "robustness_advantage_vs_image_only": (
                "image_only_relative_degradation - model_relative_degradation; "
                "positive means the model degraded less than image-only"
            ),
        },
    }

    summary["verdict"] = make_verdict(summary["models"])

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["verdict"], ensure_ascii=False, indent=2))
    print("wrote:", out_csv)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()