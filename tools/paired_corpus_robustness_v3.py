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


def parse_family(condition: str) -> str:
    parts = condition.split("_")

    if parts and parts[-1] in LEVELS:
        return "_".join(parts[:-1])

    return condition


def load_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in read_jsonl(path)
    }


def build_records(
    image_dir: Path,
    graph_dir: Path,
    distorted_manifest_dir: Path,
    clean_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_clean = load_prediction_map(
        image_dir / "clean" / "predictions.jsonl"
    )
    graph_clean = load_prediction_map(
        graph_dir / "clean" / "predictions.jsonl"
    )

    clean_metadata = {
        str(row["sample_id"]): row
        for row in read_jsonl(clean_manifest)
    }

    records = []
    missing: dict[str, int] = defaultdict(int)

    manifests = sorted(distorted_manifest_dir.glob("*.jsonl"))

    if not manifests:
        raise RuntimeError(
            f"No distorted manifests in {distorted_manifest_dir}"
        )

    for manifest_path in manifests:
        condition = manifest_path.stem
        family = parse_family(condition)

        image_distorted = load_prediction_map(
            image_dir / condition / "predictions.jsonl"
        )
        graph_distorted = load_prediction_map(
            graph_dir / condition / "predictions.jsonl"
        )

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
                missing["image_distorted"] += 1
                continue
            if gp is None:
                missing["graph_distorted"] += 1
                continue
            if ic is None:
                missing["image_clean"] += 1
                continue
            if gc is None:
                missing["graph_clean"] += 1
                continue

            metadata = clean_metadata.get(
                clean_id,
                manifest_row,
            )

            dataset = str(
                metadata.get("dataset")
                or metadata.get("source_dataset")
                or manifest_row.get("dataset")
                or "unknown"
            )

            distorted_target = (
                get_text(ip)
                or get_text(gp)
                or get_text(manifest_row)
                or get_text(metadata)
            )

            image_clean_target = (
                get_text(ic)
                or distorted_target
            )
            graph_clean_target = (
                get_text(gc)
                or distorted_target
            )

            records.append({
                "clean_sample_id": clean_id,
                "dataset": dataset,
                "condition": condition,
                "family": family,

                "image_clean_errors": edit_distance(
                    list(image_clean_target),
                    list(str(ic.get("pred", ""))),
                ),
                "graph_clean_errors": edit_distance(
                    list(graph_clean_target),
                    list(str(gc.get("pred", ""))),
                ),
                "clean_chars": max(
                    len(image_clean_target),
                    1,
                ),

                "image_distorted_errors": edit_distance(
                    list(distorted_target),
                    list(str(ip.get("pred", ""))),
                ),
                "graph_distorted_errors": edit_distance(
                    list(distorted_target),
                    list(str(gp.get("pred", ""))),
                ),
                "distorted_chars": max(
                    len(distorted_target),
                    1,
                ),
            })

    diagnostics = {
        "record_n": len(records),
        "clean_sample_n": len({
            row["clean_sample_id"]
            for row in records
        }),
        "condition_n": len({
            row["condition"]
            for row in records
        }),
        "missing": dict(missing),
    }

    return records, diagnostics


def aggregate_samples(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["clean_sample_id"]].append(row)

    samples = []

    for sample_id, sample_rows in grouped.items():
        first = sample_rows[0]

        samples.append({
            "sample_id": sample_id,
            "image_clean_errors": int(
                first["image_clean_errors"]
            ),
            "graph_clean_errors": int(
                first["graph_clean_errors"]
            ),
            "clean_chars": int(first["clean_chars"]),

            "image_distorted_errors": int(sum(
                row["image_distorted_errors"]
                for row in sample_rows
            )),
            "graph_distorted_errors": int(sum(
                row["graph_distorted_errors"]
                for row in sample_rows
            )),
            "distorted_chars": int(sum(
                row["distorted_chars"]
                for row in sample_rows
            )),
        })

    return samples


def arrays_from_samples(
    samples: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(
            [sample[key] for sample in samples],
            dtype=np.float64,
        )
        for key in [
            "image_clean_errors",
            "graph_clean_errors",
            "clean_chars",
            "image_distorted_errors",
            "graph_distorted_errors",
            "distorted_chars",
        ]
    }


def calculate(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    def selected(key: str) -> np.ndarray:
        arr = arrays[key]
        return arr if indices is None else arr[indices]

    ice = selected("image_clean_errors").sum()
    gce = selected("graph_clean_errors").sum()
    clean_chars = selected("clean_chars").sum()

    ide = selected("image_distorted_errors").sum()
    gde = selected("graph_distorted_errors").sum()
    distorted_chars = selected("distorted_chars").sum()

    image_clean = ice / max(clean_chars, 1.0)
    graph_clean = gce / max(clean_chars, 1.0)
    image_distorted = ide / max(distorted_chars, 1.0)
    graph_distorted = gde / max(distorted_chars, 1.0)

    image_abs = image_distorted - image_clean
    graph_abs = graph_distorted - graph_clean

    image_rel = image_abs / max(image_clean, 1e-12)
    graph_rel = graph_abs / max(graph_clean, 1e-12)

    return {
        "n": len(
            arrays["clean_chars"]
            if indices is None
            else indices
        ),
        "image_clean_cer": float(image_clean),
        "graph_clean_cer": float(graph_clean),
        "image_distorted_cer": float(image_distorted),
        "graph_distorted_cer": float(graph_distorted),
        "image_absolute_degradation": float(image_abs),
        "graph_absolute_degradation": float(graph_abs),
        "image_relative_degradation": float(image_rel),
        "graph_relative_degradation": float(graph_rel),
        "relative_advantage": float(
            image_rel - graph_rel
        ),
        "absolute_advantage": float(
            image_abs - graph_abs
        ),
        "distorted_cer_gap": float(
            graph_distorted - image_distorted
        ),
    }


def ci95(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def bootstrap(
    arrays: dict[str, np.ndarray],
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(arrays["clean_chars"])

    relative = np.empty(iterations)
    absolute = np.empty(iterations)
    distorted_gap = np.empty(iterations)

    for i in range(iterations):
        indices = rng.integers(0, n, size=n)
        metrics = calculate(arrays, indices)

        relative[i] = metrics["relative_advantage"]
        absolute[i] = metrics["absolute_advantage"]
        distorted_gap[i] = metrics["distorted_cer_gap"]

    return {
        "iterations": iterations,
        "relative_advantage_ci95": ci95(relative),
        "absolute_advantage_ci95": ci95(absolute),
        "distorted_cer_gap_ci95": ci95(distorted_gap),
    }


def permutation_test(
    arrays: dict[str, np.ndarray],
    permutations: int,
    rng: np.random.Generator,
    batch_size: int = 200,
) -> dict[str, Any]:
    observed = calculate(arrays)

    ice = arrays["image_clean_errors"]
    gce = arrays["graph_clean_errors"]
    ide = arrays["image_distorted_errors"]
    gde = arrays["graph_distorted_errors"]

    clean_chars = arrays["clean_chars"].sum()
    distorted_chars = arrays["distorted_chars"].sum()

    relative_extreme = 0
    relative_two_sided = 0
    absolute_extreme = 0
    completed = 0

    while completed < permutations:
        current = min(
            batch_size,
            permutations - completed,
        )

        swap = rng.integers(
            0,
            2,
            size=(current, len(ice)),
            dtype=np.int8,
        ).astype(bool)

        perm_ice = np.where(
            swap,
            gce[None, :],
            ice[None, :],
        ).sum(axis=1)

        perm_gce = np.where(
            swap,
            ice[None, :],
            gce[None, :],
        ).sum(axis=1)

        perm_ide = np.where(
            swap,
            gde[None, :],
            ide[None, :],
        ).sum(axis=1)

        perm_gde = np.where(
            swap,
            ide[None, :],
            gde[None, :],
        ).sum(axis=1)

        image_clean = perm_ice / clean_chars
        graph_clean = perm_gce / clean_chars
        image_distorted = perm_ide / distorted_chars
        graph_distorted = perm_gde / distorted_chars

        image_abs = image_distorted - image_clean
        graph_abs = graph_distorted - graph_clean

        image_rel = image_abs / np.maximum(
            image_clean,
            1e-12,
        )
        graph_rel = graph_abs / np.maximum(
            graph_clean,
            1e-12,
        )

        rel_adv = image_rel - graph_rel
        abs_adv = image_abs - graph_abs

        relative_extreme += int(np.sum(
            rel_adv >= observed["relative_advantage"]
        ))
        relative_two_sided += int(np.sum(
            np.abs(rel_adv)
            >= abs(observed["relative_advantage"])
        ))
        absolute_extreme += int(np.sum(
            abs_adv >= observed["absolute_advantage"]
        ))

        completed += current

    return {
        "permutations": permutations,
        "relative_advantage_one_sided_p": (
            relative_extreme + 1
        ) / (permutations + 1),
        "relative_advantage_two_sided_p": (
            relative_two_sided + 1
        ) / (permutations + 1),
        "absolute_advantage_one_sided_p": (
            absolute_extreme + 1
        ) / (permutations + 1),
    }


def analyse_scope(
    name: str,
    rows: list[dict[str, Any]],
    bootstrap_iterations: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    samples = aggregate_samples(rows)

    if len(samples) < 30:
        return {
            "scope": name,
            "n": len(samples),
            "error": "too_few_samples",
        }

    arrays = arrays_from_samples(samples)
    observed = calculate(arrays)

    offset = sum(ord(ch) for ch in name)

    observed["scope"] = name
    observed["bootstrap"] = bootstrap(
        arrays,
        iterations=bootstrap_iterations,
        rng=np.random.default_rng(seed + offset),
    )
    observed["permutation"] = permutation_test(
        arrays,
        permutations=permutations,
        rng=np.random.default_rng(
            seed + 100000 + offset
        ),
    )

    return observed


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


def pct_ci(values: list[float]) -> str:
    return (
        f"{100.0 * values[0]:.2f}%–"
        f"{100.0 * values[1]:.2f}%"
    )


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

    for dataset in sorted({
        row["dataset"] for row in records
    }):
        scopes[f"dataset:{dataset}"] = [
            row for row in records
            if row["dataset"] == dataset
        ]

    for family in sorted({
        row["family"] for row in records
    }):
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
            f"{index}/{len(scopes)} "
            f"{name}: {len(rows)} records"
        )

        results.append(
            analyse_scope(
                name=name,
                rows=rows,
                bootstrap_iterations=(
                    args.bootstrap_iterations
                ),
                permutations=args.permutations,
                seed=args.seed,
            )
        )

    result = {
        "diagnostics": diagnostics,
        "configuration": vars(args),
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "paired_corpus_v3.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Paired corpus robustness analysis v3")
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
        f"- conditions: {diagnostics['condition_n']}"
    )
    lines.append(
        f"- missing joins: `{diagnostics['missing']}`"
    )
    lines.append("")

    lines.append("## 2. Corpus-level paired results")
    lines.append("")
    lines.append(
        "| scope | n | image relative degradation | "
        "graph relative degradation | relative advantage | "
        "relative 95% CI | relative one-sided p | "
        "absolute advantage | absolute 95% CI | "
        "graph−image distorted CER |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|"
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
            f"{pct(row['relative_advantage'])} | "
            f"{pct_ci(boot['relative_advantage_ci95'])} | "
            f"{perm['relative_advantage_one_sided_p']:.6f} | "
            f"{fmt(row['absolute_advantage'])} | "
            f"{fmt(boot['absolute_advantage_ci95'][0])}–"
            f"{fmt(boot['absolute_advantage_ci95'][1])} | "
            f"{fmt(row['distorted_cer_gap'])} |"
        )

    lines.append("")
    lines.append("## 3. Strict interpretation")
    lines.append("")

    overall = next(
        row for row in results
        if row.get("scope") == "overall"
    )

    rel_ci = overall["bootstrap"][
        "relative_advantage_ci95"
    ]
    rel_p = overall["permutation"][
        "relative_advantage_one_sided_p"
    ]
    abs_ci = overall["bootstrap"][
        "absolute_advantage_ci95"
    ]

    if rel_ci[0] > 0 and rel_p < 0.05:
        lines.append(
            "The graph model has a statistically supported "
            "corpus-level relative robustness advantage."
        )
    else:
        lines.append(
            "The corpus-level analysis does not establish a "
            "statistically supported relative robustness advantage."
        )

    if abs_ci[0] > 0:
        lines.append(
            "The graph model also has a positive absolute "
            "degradation advantage."
        )
    else:
        lines.append(
            "A consistent positive absolute degradation "
            "advantage is not established."
        )

    if overall["distorted_cer_gap"] > 0:
        lines.append(
            "Absolute distorted CER remains worse for the "
            "graph model."
        )
    else:
        lines.append(
            "Absolute distorted CER is equal or better for "
            "the graph model."
        )

    lines.append("")
    lines.append(
        "This supports only a partial robustness claim. "
        "It is not evidence of superior absolute HTR accuracy."
    )

    (out_dir / "paired_corpus_v3.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("wrote:", out_dir / "paired_corpus_v3.md")


if __name__ == "__main__":
    main()