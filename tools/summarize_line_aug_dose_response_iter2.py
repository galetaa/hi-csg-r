from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_predictions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row["dataset"] = dataset_from_sample_id(str(row["sample_id"]))
            out[str(row["sample_id"])] = row
    return out


def load_aug_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return load_predictions([path])


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def aggregate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "cer": mean([float(row["cer"]) for row in rows]),
        "wer": mean([float(row["wer"]) for row in rows]),
        "exact": mean([float(row.get("exact", 0.0)) for row in rows]),
    }


def aggregate_by_dataset(preds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds.values():
        grouped[str(row["dataset"])].append(row)
    return {
        dataset: aggregate_predictions(grouped.get(dataset, []))
        for dataset in DATASETS
    }


def bootstrap_delta_ci(
    baseline: dict[str, dict[str, Any]],
    augmented: dict[str, dict[str, Any]],
    *,
    dataset: str | None,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    common = sorted(set(baseline) & set(augmented))
    if dataset is not None:
        common = [
            sample_id for sample_id in common
            if baseline[sample_id]["dataset"] == dataset
        ]

    deltas = np.asarray([
        float(augmented[sample_id]["cer"]) - float(baseline[sample_id]["cer"])
        for sample_id in common
    ], dtype=np.float64)

    if deltas.size == 0:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "mean_delta_cer": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(deltas, size=deltas.size, replace=True)
        means[i] = sample.mean()

    return {
        "n": int(deltas.size),
        "wins": int(np.sum(deltas < 0)),
        "losses": int(np.sum(deltas > 0)),
        "ties": int(np.sum(deltas == 0)),
        "mean_delta_cer": float(deltas.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    args = parser.parse_args()

    baseline_preds = load_predictions([
        Path("outputs/htr_graph_v1/eval_tri10k_image_only_v1_hkr_words_test_final/predictions.jsonl"),
        Path("outputs/htr_graph_v1/eval_tri10k_image_only_v1_cyrillic_handwriting_test_final/predictions.jsonl"),
        Path("outputs/htr_graph_v1/eval_tri10k_image_only_v1_school_notebooks_clean_test_final/predictions.jsonl"),
    ])

    runs = {
        "baseline": {
            "train_summary": None,
            "predictions": baseline_preds,
            "output": "outputs/htr_graph_v1/tri10k_image_only_v1",
        },
    }

    for tag in ["2k", "5k", "10k"]:
        runs[f"+{tag} lines"] = {
            "train_summary": read_json(Path(
                f"data/experiments/htr_baseline_v1_ctc_ready/"
                f"tri10k_mixed_plus_school_lines_{tag}_context_v1/summary.json"
            )),
            "config": read_json(Path(
                f"outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_{tag}_context_v1/config.json"
            )),
            "eval_summary": read_json(Path(
                f"outputs/htr_graph_v1/eval_tri10k_image_only_plus_school_lines_{tag}_context_v1_test_final/summary.json"
            )),
            "predictions": load_aug_predictions(Path(
                f"outputs/htr_graph_v1/eval_tri10k_image_only_plus_school_lines_{tag}_context_v1_test_final/predictions.jsonl"
            )),
            "output": f"outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_{tag}_context_v1",
        }

    baseline_overall = aggregate_predictions(list(baseline_preds.values()))
    baseline_by_dataset = aggregate_by_dataset(baseline_preds)

    result: dict[str, Any] = {
        "runs": {},
        "paired_bootstrap": {},
    }

    result["runs"]["baseline"] = {
        "train_n": 30000,
        "line_train_n": 0,
        "overall": baseline_overall,
        "by_dataset": baseline_by_dataset,
    }

    for run_name, run in runs.items():
        if run_name == "baseline":
            continue

        preds = run["predictions"]
        train_summary = run["train_summary"]
        overall = aggregate_predictions(list(preds.values()))
        by_dataset = aggregate_by_dataset(preds)

        result["runs"][run_name] = {
            "train_n": train_summary["merged_train_n"],
            "line_train_input_n": train_summary["line_train_input_n"],
            "line_train_n": train_summary["line_train_n"],
            "line_train_oov_filtered_n": train_summary["line_train_oov_filtered_n"],
            "overall": overall,
            "by_dataset": by_dataset,
            "checkpoint_epoch": run["eval_summary"].get("checkpoint_epoch"),
            "checkpoint_val_cer": run["eval_summary"].get("checkpoint_val_cer"),
            "blank_logit_penalty": run["eval_summary"].get("blank_logit_penalty"),
            "pred_len_mean": run["eval_summary"]["metrics"].get("pred_len_mean"),
            "argmax_blank_ratio": run["eval_summary"]["metrics"].get("argmax_blank_ratio"),
        }

        result["paired_bootstrap"][run_name] = {
            "overall": bootstrap_delta_ci(
                baseline_preds,
                preds,
                dataset=None,
                seed=args.seed,
                n_bootstrap=args.n_bootstrap,
            ),
            **{
                dataset: bootstrap_delta_ci(
                    baseline_preds,
                    preds,
                    dataset=dataset,
                    seed=args.seed + i + 1,
                    n_bootstrap=args.n_bootstrap,
                )
                for i, dataset in enumerate(DATASETS)
            },
        }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    write_json(out_json, result)

    lines = [
        "# School Natural-Line Augmentation Dose Response",
        "",
        "| model | train_n | line_n | overall CER | overall WER | exact | HKR CER | Cyrillic CER | School CER | School WER | School exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name in ["baseline", "+2k lines", "+5k lines", "+10k lines"]:
        row = result["runs"][run_name]
        overall = row["overall"]
        by_ds = row["by_dataset"]
        lines.append(
            f"| {run_name} | {row['train_n']} | {row.get('line_train_n', 0)} | "
            f"{fmt(overall['cer'])} | {fmt(overall['wer'])} | {fmt(overall['exact'])} | "
            f"{fmt(by_ds['hkr_words']['cer'])} | "
            f"{fmt(by_ds['cyrillic_handwriting']['cer'])} | "
            f"{fmt(by_ds['school_notebooks_clean']['cer'])} | "
            f"{fmt(by_ds['school_notebooks_clean']['wer'])} | "
            f"{fmt(by_ds['school_notebooks_clean']['exact'])} |"
        )

    lines.extend([
        "",
        "## Paired CER Delta vs Baseline",
        "",
        "| model | scope | n | wins | losses | ties | mean delta CER | CI95 low | CI95 high |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run_name in ["+2k lines", "+5k lines", "+10k lines"]:
        for scope in ["overall", *DATASETS]:
            row = result["paired_bootstrap"][run_name][scope]
            lines.append(
                f"| {run_name} | {scope} | {row['n']} | {row['wins']} | {row['losses']} | {row['ties']} | "
                f"{fmt(row['mean_delta_cer'])} | {fmt(row['ci95_low'])} | {fmt(row['ci95_high'])} |"
            )

    lines.extend([
        "",
        "## Training Notes",
        "",
    ])
    for run_name in ["+2k lines", "+5k lines", "+10k lines"]:
        row = result["runs"][run_name]
        lines.append(
            f"- {run_name}: line input {row['line_train_input_n']}, "
            f"used {row['line_train_n']}, OOV filtered {row['line_train_oov_filtered_n']}, "
            f"best epoch {row['checkpoint_epoch']}, val CER {fmt(row['checkpoint_val_cer'])}, "
            f"blank penalty {fmt(row['blank_logit_penalty'])}."
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "overall_cer": {
            run_name: result["runs"][run_name]["overall"]["cer"]
            for run_name in ["baseline", "+2k lines", "+5k lines", "+10k lines"]
        },
        "school_cer": {
            run_name: result["runs"][run_name]["by_dataset"]["school_notebooks_clean"]["cer"]
            for run_name in ["baseline", "+2k lines", "+5k lines", "+10k lines"]
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
