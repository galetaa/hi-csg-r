from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LEVELS = {"mild", "medium", "strong"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_text(row: dict[str, Any]) -> str:
    for key in ["target", "text", "label", "transcription"]:
        if key in row:
            return str(row[key])
    return ""


def edit_distance(a: list[Any], b: list[Any]) -> int:
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )

        previous = current

    return previous[-1]


def sample_cer(target: str, pred: str) -> float:
    return edit_distance(list(target), list(pred)) / max(len(target), 1)


def parse_family(condition: str) -> str:
    parts = condition.split("_")

    if parts and parts[-1] in LEVELS:
        return "_".join(parts[:-1])

    return condition


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in read_jsonl(path)
    }


def build_records(
    *,
    image_dir: Path,
    graph_dir: Path,
    distorted_manifest_dir: Path,
    clean_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_clean = load_predictions(
        image_dir / "clean" / "predictions.jsonl"
    )
    graph_clean = load_predictions(
        graph_dir / "clean" / "predictions.jsonl"
    )

    clean_rows = read_jsonl(clean_manifest)
    clean_metadata = {
        str(row["sample_id"]): row
        for row in clean_rows
    }

    records = []
    missing = defaultdict(int)

    manifest_paths = sorted(distorted_manifest_dir.glob("*.jsonl"))

    if not manifest_paths:
        raise RuntimeError(
            f"No distorted manifests found in {distorted_manifest_dir}"
        )

    for manifest_path in manifest_paths:
        condition = manifest_path.stem
        family = parse_family(condition)

        image_pred_path = image_dir / condition / "predictions.jsonl"
        graph_pred_path = graph_dir / condition / "predictions.jsonl"

        if not image_pred_path.exists():
            raise FileNotFoundError(image_pred_path)

        if not graph_pred_path.exists():
            raise FileNotFoundError(graph_pred_path)

        image_distorted = load_predictions(image_pred_path)
        graph_distorted = load_predictions(graph_pred_path)

        for manifest_row in read_jsonl(manifest_path):
            distorted_id = str(manifest_row["sample_id"])
            clean_id = str(
                manifest_row.get("clean_sample_id", "")
            )

            if not clean_id:
                missing["missing_clean_sample_id"] += 1
                continue

            ip = image_distorted.get(distorted_id)
            gp = graph_distorted.get(distorted_id)
            ic = image_clean.get(clean_id)
            gc = graph_clean.get(clean_id)

            if ip is None:
                missing["image_distorted_prediction"] += 1
                continue
            if gp is None:
                missing["graph_distorted_prediction"] += 1
                continue
            if ic is None:
                missing["image_clean_prediction"] += 1
                continue
            if gc is None:
                missing["graph_clean_prediction"] += 1
                continue

            metadata = clean_metadata.get(clean_id, manifest_row)

            dataset = str(
                metadata.get("dataset")
                or metadata.get("source_dataset")
                or manifest_row.get("dataset")
                or "unknown"
            )

            target = (
                get_text(ip)
                or get_text(gp)
                or get_text(manifest_row)
                or get_text(metadata)
            )

            image_clean_target = get_text(ic) or target
            graph_clean_target = get_text(gc) or target

            records.append(
                {
                    "clean_sample_id": clean_id,
                    "distorted_sample_id": distorted_id,
                    "dataset": dataset,
                    "condition": condition,
                    "family": family,
                    "image_clean_cer": sample_cer(
                        image_clean_target,
                        str(ic.get("pred", "")),
                    ),
                    "graph_clean_cer": sample_cer(
                        graph_clean_target,
                        str(gc.get("pred", "")),
                    ),
                    "image_distorted_cer": sample_cer(
                        target,
                        str(ip.get("pred", "")),
                    ),
                    "graph_distorted_cer": sample_cer(
                        target,
                        str(gp.get("pred", "")),
                    ),
                }
            )

    diagnostics = {
        "record_n": len(records),
        "clean_sample_n": len({
            r["clean_sample_id"] for r in records
        }),
        "condition_n": len({
            r["condition"] for r in records
        }),
        "missing": dict(missing),
    }

    return records, diagnostics


def aggregate_by_sample(
    rows: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["clean_sample_id"]].append(row)

    output = []

    for sample_id, sample_rows in grouped.items():
        first = sample_rows[0]

        output.append(
            {
                "sample_id": sample_id,
                "image_clean": float(first["image_clean_cer"]),
                "graph_clean": float(first["graph_clean_cer"]),
                "image_distorted": float(np.mean([
                    r["image_distorted_cer"]
                    for r in sample_rows
                ])),
                "graph_distorted": float(np.mean([
                    r["graph_distorted_cer"]
                    for r in sample_rows
                ])),
            }
        )

    return output


def calculate_metrics(
    samples: list[dict[str, float | str]],
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    image_clean = np.asarray(
        [float(row["image_clean"]) for row in samples],
        dtype=np.float64,
    )
    graph_clean = np.asarray(
        [float(row["graph_clean"]) for row in samples],
        dtype=np.float64,
    )
    image_distorted = np.asarray(
        [float(row["image_distorted"]) for row in samples],
        dtype=np.float64,
    )
    graph_distorted = np.asarray(
        [float(row["graph_distorted"]) for row in samples],
        dtype=np.float64,
    )

    if indices is not None:
        image_clean = image_clean[indices]
        graph_clean = graph_clean[indices]
        image_distorted = image_distorted[indices]
        graph_distorted = graph_distorted[indices]

    image_clean_mean = float(image_clean.mean())
    graph_clean_mean = float(graph_clean.mean())
    image_distorted_mean = float(image_distorted.mean())
    graph_distorted_mean = float(graph_distorted.mean())

    image_absolute_degradation = (
        image_distorted_mean - image_clean_mean
    )
    graph_absolute_degradation = (
        graph_distorted_mean - graph_clean_mean
    )

    image_relative_degradation = (
        image_absolute_degradation
        / max(image_clean_mean, 1e-12)
    )
    graph_relative_degradation = (
        graph_absolute_degradation
        / max(graph_clean_mean, 1e-12)
    )

    paired_sample_advantages = (
        image_distorted
        - image_clean
        - graph_distorted
        + graph_clean
    )

    return {
        "n": len(image_clean),
        "image_clean_cer": image_clean_mean,
        "graph_clean_cer": graph_clean_mean,
        "image_distorted_cer": image_distorted_mean,
        "graph_distorted_cer": graph_distorted_mean,
        "image_absolute_degradation": image_absolute_degradation,
        "graph_absolute_degradation": graph_absolute_degradation,
        "image_relative_degradation": image_relative_degradation,
        "graph_relative_degradation": graph_relative_degradation,
        "relative_degradation_advantage": (
            image_relative_degradation
            - graph_relative_degradation
        ),
        "paired_absolute_degradation_advantage": float(
            paired_sample_advantages.mean()
        ),
        "graph_minus_image_distorted_cer": (
            graph_distorted_mean - image_distorted_mean
        ),
    }


def percentile_interval(
    values: np.ndarray,
) -> tuple[float, float]:
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def bootstrap_scope(
    samples: list[dict[str, float | str]],
    *,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    observed = calculate_metrics(samples)
    n = len(samples)

    relative_advantages = np.empty(
        iterations,
        dtype=np.float64,
    )
    absolute_advantages = np.empty(
        iterations,
        dtype=np.float64,
    )
    distorted_gaps = np.empty(
        iterations,
        dtype=np.float64,
    )

    for i in range(iterations):
        indices = rng.integers(0, n, size=n)
        metrics = calculate_metrics(samples, indices)

        relative_advantages[i] = metrics[
            "relative_degradation_advantage"
        ]
        absolute_advantages[i] = metrics[
            "paired_absolute_degradation_advantage"
        ]
        distorted_gaps[i] = metrics[
            "graph_minus_image_distorted_cer"
        ]

    rel_low, rel_high = percentile_interval(
        relative_advantages
    )
    abs_low, abs_high = percentile_interval(
        absolute_advantages
    )
    gap_low, gap_high = percentile_interval(
        distorted_gaps
    )

    observed["bootstrap"] = {
        "iterations": iterations,
        "relative_degradation_advantage_ci95": [
            rel_low,
            rel_high,
        ],
        "paired_absolute_degradation_advantage_ci95": [
            abs_low,
            abs_high,
        ],
        "graph_minus_image_distorted_cer_ci95": [
            gap_low,
            gap_high,
        ],
    }

    return observed


def permutation_test(
    samples: list[dict[str, float | str]],
    *,
    permutations: int,
    rng: np.random.Generator,
    batch_size: int = 500,
) -> dict[str, float]:
    advantages = np.asarray(
        [
            float(row["image_distorted"])
            - float(row["image_clean"])
            - float(row["graph_distorted"])
            + float(row["graph_clean"])
            for row in samples
        ],
        dtype=np.float64,
    )

    observed = float(advantages.mean())
    one_sided_extreme = 0
    two_sided_extreme = 0
    completed = 0

    while completed < permutations:
        current = min(
            batch_size,
            permutations - completed,
        )

        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(current, len(advantages)),
        )
        permuted = (
            signs * advantages[None, :]
        ).mean(axis=1)

        one_sided_extreme += int(
            np.sum(permuted >= observed)
        )
        two_sided_extreme += int(
            np.sum(np.abs(permuted) >= abs(observed))
        )

        completed += current

    return {
        "observed_paired_absolute_advantage": observed,
        "one_sided_p": (
            one_sided_extreme + 1
        ) / (permutations + 1),
        "two_sided_p": (
            two_sided_extreme + 1
        ) / (permutations + 1),
        "permutations": permutations,
    }


def analyse_scope(
    name: str,
    rows: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    samples = aggregate_by_sample(rows)

    if len(samples) < 30:
        return {
            "scope": name,
            "n": len(samples),
            "error": "too_few_samples",
        }

    rng_bootstrap = np.random.default_rng(
        seed + sum(ord(ch) for ch in name)
    )
    rng_permutation = np.random.default_rng(
        seed + 100000 + sum(ord(ch) for ch in name)
    )

    result = bootstrap_scope(
        samples,
        iterations=bootstrap_iterations,
        rng=rng_bootstrap,
    )
    result["scope"] = name
    result["permutation"] = permutation_test(
        samples,
        permutations=permutations,
        rng=rng_permutation,
    )

    return result


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return "n/a"


def pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def ci_pct(values: list[float]) -> str:
    return (
        f"{100.0 * values[0]:.2f}%–"
        f"{100.0 * values[1]:.2f}%"
    )


def ci_raw(values: list[float]) -> str:
    return f"{values[0]:.5f}–{values[1]:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument(
        "--distorted_manifest_dir",
        required=True,
    )
    parser.add_argument("--clean_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=20000,
    )
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    records, diagnostics = build_records(
        image_dir=Path(args.image_dir),
        graph_dir=Path(args.graph_dir),
        distorted_manifest_dir=Path(
            args.distorted_manifest_dir
        ),
        clean_manifest=Path(args.clean_manifest),
    )

    scopes: dict[str, list[dict[str, Any]]] = {
        "overall": records,
    }

    datasets = sorted({
        row["dataset"] for row in records
    })
    families = sorted({
        row["family"] for row in records
    })

    for dataset in datasets:
        scopes[f"dataset:{dataset}"] = [
            row for row in records
            if row["dataset"] == dataset
        ]

    for family in families:
        scopes[f"family:{family}"] = [
            row for row in records
            if row["family"] == family
        ]

    results = []

    for index, (name, rows) in enumerate(
        sorted(scopes.items()),
        start=1,
    ):
        print(
            f"{index}/{len(scopes)} {name}: "
            f"{len(rows)} records"
        )

        results.append(
            analyse_scope(
                name,
                rows,
                bootstrap_iterations=args.bootstrap_iterations,
                permutations=args.permutations,
                seed=args.seed,
            )
        )

    result = {
        "diagnostics": diagnostics,
        "configuration": {
            "image_dir": args.image_dir,
            "graph_dir": args.graph_dir,
            "distorted_manifest_dir": (
                args.distorted_manifest_dir
            ),
            "clean_manifest": args.clean_manifest,
            "bootstrap_iterations": (
                args.bootstrap_iterations
            ),
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "paired_robustness_v2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Paired robustness analysis v2")
    lines.append("")
    lines.append("## 1. Diagnostics")
    lines.append("")
    lines.append(
        f"- joined records: {diagnostics['record_n']}"
    )
    lines.append(
        f"- unique clean samples: "
        f"{diagnostics['clean_sample_n']}"
    )
    lines.append(
        f"- distortion conditions: "
        f"{diagnostics['condition_n']}"
    )
    lines.append(
        f"- missing joins: `{diagnostics['missing']}`"
    )
    lines.append("")

    lines.append("## 2. Paired results")
    lines.append("")
    lines.append(
        "| scope | n | image rel. degradation | "
        "graph rel. degradation | advantage | "
        "advantage 95% CI | paired ΔCER advantage | "
        "paired 95% CI | one-sided p | "
        "graph−image distorted CER |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for row in results:
        if "error" in row:
            continue

        boot = row["bootstrap"]
        perm = row["permutation"]

        lines.append(
            f"| `{row['scope']}` | {row['n']} | "
            f"{pct(row['image_relative_degradation'])} | "
            f"{pct(row['graph_relative_degradation'])} | "
            f"{pct(row['relative_degradation_advantage'])} | "
            f"{ci_pct(boot['relative_degradation_advantage_ci95'])} | "
            f"{fmt(row['paired_absolute_degradation_advantage'])} | "
            f"{ci_raw(boot['paired_absolute_degradation_advantage_ci95'])} | "
            f"{perm['one_sided_p']:.6f} | "
            f"{fmt(row['graph_minus_image_distorted_cer'])} |"
        )

    lines.append("")
    lines.append("## 3. Interpretation")
    lines.append("")

    overall = next(
        row for row in results
        if row.get("scope") == "overall"
    )

    rel_ci = overall["bootstrap"][
        "relative_degradation_advantage_ci95"
    ]
    abs_ci = overall["bootstrap"][
        "paired_absolute_degradation_advantage_ci95"
    ]

    if rel_ci[0] > 0 and abs_ci[0] > 0:
        lines.append(
            "The graph model has a statistically supported "
            "paired robustness advantage: both the aggregate "
            "relative-degradation advantage and the paired "
            "absolute degradation advantage have positive "
            "95% bootstrap intervals."
        )
    else:
        lines.append(
            "The paired analysis does not provide unambiguous "
            "statistical support for a graph robustness "
            "advantage."
        )

    if overall["graph_minus_image_distorted_cer"] > 0:
        lines.append(
            "The graph model nevertheless has worse absolute "
            "CER on distorted samples."
        )
    else:
        lines.append(
            "The graph model has equal or better absolute CER "
            "on distorted samples."
        )

    lines.append("")
    lines.append(
        "The paired test evaluates robustness change for the "
        "same source samples. It does not convert a relative "
        "robustness result into an absolute HTR improvement."
    )

    (out_dir / "paired_robustness_v2.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "wrote:",
        out_dir / "paired_robustness_v2.md",
    )


if __name__ == "__main__":
    main()