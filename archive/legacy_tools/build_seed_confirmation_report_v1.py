from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


BASELINE_EXPERIMENT_IDS = {
    "42": [
        "eval_tri10k_image_only_v1_test_final",
        "eval_tri10k_image_only_v1_seed42_test_final",
    ],
    "43": [
        "eval_tri10k_image_only_v1_seed43_test_final",
    ],
    "44": [
        "eval_tri10k_image_only_v1_seed44_test_final",
    ],
}

PLUS10K_EXPERIMENT_IDS = {
    "42": [
        "eval_tri10k_image_only_plus_school_lines_10k_context_v1_test_final",
        "eval_tri10k_image_only_plus_school_lines_10k_context_v1_seed42_test_final",
    ],
    "43": [
        "eval_tri10k_image_only_plus_school_lines_10k_context_v1_seed43_test_final",
    ],
    "44": [
        "eval_tri10k_image_only_plus_school_lines_10k_context_v1_seed44_test_final",
    ],
}


DOMAINS = [
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks_clean",
    "school",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_inventory(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def find_summary_path(
    inventory_rows: list[dict[str, str]],
    candidate_ids: list[str],
) -> Path:
    by_experiment = {
        row.get("experiment_id", ""): row
        for row in inventory_rows
    }

    for experiment_id in candidate_ids:
        row = by_experiment.get(experiment_id)
        if row and row.get("summary_path"):
            path = Path(row["summary_path"])
            if path.exists():
                return path

    for experiment_id in candidate_ids:
        for row in inventory_rows:
            if experiment_id in row.get("experiment_id", "") and row.get("summary_path"):
                path = Path(row["summary_path"])
                if path.exists():
                    return path

    raise FileNotFoundError(
        "No summary found for candidate ids: " + ", ".join(candidate_ids)
    )


def get_float(summary: dict[str, Any], key: str) -> float | None:
    if key in summary and summary[key] not in (None, ""):
        return float(summary[key])

    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and key in metrics and metrics[key] not in (None, ""):
        return float(metrics[key])

    return None


def get_grouped(summary: dict[str, Any]) -> dict[str, Any]:
    grouped = summary.get("grouped")
    if isinstance(grouped, dict):
        return grouped

    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("grouped"), dict):
        return metrics["grouped"]

    return {}


def domain_rollup(grouped: dict[str, Any], domain: str) -> dict[str, float] | None:
    direct = grouped.get(domain)
    if isinstance(direct, dict):
        return {
            "n": float(direct.get("n", 0.0)),
            "cer": float(direct.get("cer", 0.0)),
            "wer": float(direct.get("wer", 0.0)),
            "exact": float(direct.get("exact", 0.0)),
        }

    matches = [
        value
        for key, value in grouped.items()
        if isinstance(value, dict) and str(key).split("|", 1)[0] == domain
    ]

    if not matches:
        return None

    n_total = sum(float(row.get("n", 0.0)) for row in matches)
    if n_total <= 0:
        return None

    return {
        "n": n_total,
        "cer": sum(float(row.get("cer", 0.0)) * float(row.get("n", 0.0)) for row in matches)
        / n_total,
        "wer": sum(float(row.get("wer", 0.0)) * float(row.get("n", 0.0)) for row in matches)
        / n_total,
        "exact": sum(float(row.get("exact", 0.0)) * float(row.get("n", 0.0)) for row in matches)
        / n_total,
    }


def extract_summary_row(
    model: str,
    seed: str,
    summary_path: Path,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    grouped = get_grouped(summary)

    row: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "summary_path": str(summary_path),
        "n": get_float(summary, "n"),
        "cer": get_float(summary, "cer"),
        "wer": get_float(summary, "wer"),
        "exact": get_float(summary, "exact"),
        "pred_len_mean": get_float(summary, "pred_len_mean"),
        "blank_logit_penalty": summary.get("blank_logit_penalty", ""),
        "checkpoint_epoch": summary.get("checkpoint_epoch", ""),
        "checkpoint_val_cer": summary.get("checkpoint_val_cer", ""),
    }

    for domain in DOMAINS:
        rolled = domain_rollup(grouped, domain)
        if rolled is not None:
            row[f"{domain}_n"] = rolled["n"]
            row[f"{domain}_cer"] = rolled["cer"]
            row[f"{domain}_wer"] = rolled["wer"]
            row[f"{domain}_exact"] = rolled["exact"]
        else:
            row[f"{domain}_n"] = ""
            row[f"{domain}_cer"] = ""
            row[f"{domain}_wer"] = ""
            row[f"{domain}_exact"] = ""

    return row


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def safe_float(x: Any) -> float | None:
    if x in (None, ""):
        return None
    return float(x)


def build_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed_model = {
        (str(row["seed"]), str(row["model"])): row
        for row in rows
    }

    delta_rows: list[dict[str, Any]] = []

    for seed in ["42", "43", "44"]:
        base = by_seed_model[(seed, "baseline")]
        plus = by_seed_model[(seed, "plus_10k")]

        base_cer = safe_float(base["cer"])
        plus_cer = safe_float(plus["cer"])
        base_wer = safe_float(base["wer"])
        plus_wer = safe_float(plus["wer"])
        base_exact = safe_float(base["exact"])
        plus_exact = safe_float(plus["exact"])

        assert base_cer is not None and plus_cer is not None
        assert base_wer is not None and plus_wer is not None
        assert base_exact is not None and plus_exact is not None

        row: dict[str, Any] = {
            "seed": seed,
            "baseline_cer": base_cer,
            "plus_10k_cer": plus_cer,
            "delta_cer": plus_cer - base_cer,
            "relative_delta_cer": (plus_cer - base_cer) / max(base_cer, 1e-12),
            "baseline_wer": base_wer,
            "plus_10k_wer": plus_wer,
            "delta_wer": plus_wer - base_wer,
            "baseline_exact": base_exact,
            "plus_10k_exact": plus_exact,
            "delta_exact": plus_exact - base_exact,
        }

        for domain in DOMAINS:
            bdc = safe_float(base.get(f"{domain}_cer"))
            pdc = safe_float(plus.get(f"{domain}_cer"))
            if bdc is not None and pdc is not None:
                row[f"{domain}_baseline_cer"] = bdc
                row[f"{domain}_plus_10k_cer"] = pdc
                row[f"{domain}_delta_cer"] = pdc - bdc
                row[f"{domain}_relative_delta_cer"] = (pdc - bdc) / max(bdc, 1e-12)
            else:
                row[f"{domain}_baseline_cer"] = ""
                row[f"{domain}_plus_10k_cer"] = ""
                row[f"{domain}_delta_cer"] = ""
                row[f"{domain}_relative_delta_cer"] = ""

        delta_rows.append(row)

    return delta_rows


def aggregate_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for model in ["baseline", "plus_10k"]:
        group = [row for row in rows if row["model"] == model]

        out_row: dict[str, Any] = {
            "model": model,
            "seeds_n": len(group),
            "mean_cer": mean([float(row["cer"]) for row in group]),
            "std_cer": stdev([float(row["cer"]) for row in group]),
            "mean_wer": mean([float(row["wer"]) for row in group]),
            "std_wer": stdev([float(row["wer"]) for row in group]),
            "mean_exact": mean([float(row["exact"]) for row in group]),
            "std_exact": stdev([float(row["exact"]) for row in group]),
        }

        for domain in DOMAINS:
            vals = [
                value
                for row in group
                if (value := safe_float(row.get(f"{domain}_cer"))) is not None
            ]
            if vals:
                out_row[f"{domain}_mean_cer"] = mean(vals)
                out_row[f"{domain}_std_cer"] = stdev(vals)
            else:
                out_row[f"{domain}_mean_cer"] = ""
                out_row[f"{domain}_std_cer"] = ""

        out.append(out_row)

    return out


def aggregate_delta_rows(delta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta_cer = [float(row["delta_cer"]) for row in delta_rows]
    rel_delta_cer = [float(row["relative_delta_cer"]) for row in delta_rows]
    delta_wer = [float(row["delta_wer"]) for row in delta_rows]
    delta_exact = [float(row["delta_exact"]) for row in delta_rows]

    out: dict[str, Any] = {
        "seeds": [row["seed"] for row in delta_rows],
        "mean_delta_cer": mean(delta_cer),
        "std_delta_cer": stdev(delta_cer),
        "mean_relative_delta_cer": mean(rel_delta_cer),
        "std_relative_delta_cer": stdev(rel_delta_cer),
        "mean_delta_wer": mean(delta_wer),
        "std_delta_wer": stdev(delta_wer),
        "mean_delta_exact": mean(delta_exact),
        "std_delta_exact": stdev(delta_exact),
        "improved_cer_seeds_n": sum(1 for x in delta_cer if x < 0),
        "improved_wer_seeds_n": sum(1 for x in delta_wer if x < 0),
        "improved_exact_seeds_n": sum(1 for x in delta_exact if x > 0),
    }

    for domain in DOMAINS:
        vals = [
            value
            for row in delta_rows
            if (value := safe_float(row.get(f"{domain}_delta_cer"))) is not None
        ]
        rel_vals = [
            value
            for row in delta_rows
            if (value := safe_float(row.get(f"{domain}_relative_delta_cer"))) is not None
        ]

        if vals:
            out[f"{domain}_mean_delta_cer"] = mean(vals)
            out[f"{domain}_std_delta_cer"] = stdev(vals)
            out[f"{domain}_improved_seeds_n"] = sum(1 for x in vals if x < 0)
        else:
            out[f"{domain}_mean_delta_cer"] = None
            out[f"{domain}_std_delta_cer"] = None
            out[f"{domain}_improved_seeds_n"] = 0

        if rel_vals:
            out[f"{domain}_mean_relative_delta_cer"] = mean(rel_vals)
            out[f"{domain}_std_relative_delta_cer"] = stdev(rel_vals)
        else:
            out[f"{domain}_mean_relative_delta_cer"] = None
            out[f"{domain}_std_relative_delta_cer"] = None

    if out["improved_cer_seeds_n"] == 3:
        out["interpretation"] = (
            "+10k natural-line context improves CER in all three seeds; "
            "primary HTR gain is seed-stable."
        )
    elif out["improved_cer_seeds_n"] == 2:
        out["interpretation"] = (
            "+10k natural-line context improves CER in two of three seeds; "
            "effect is positive but training-seed stability should be discussed."
        )
    else:
        out["interpretation"] = (
            "+10k natural-line context is not seed-stable; "
            "claim must be weakened."
        )

    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    aggregate_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    delta_summary: dict[str, Any],
    path: Path,
) -> None:
    lines: list[str] = []

    lines.append("# Seed confirmation report v1\n")
    lines.append("## Purpose\n")
    lines.append(
        "This report checks whether the primary HTR improvement "
        "from natural-line context augmentation is stable across three training seeds.\n"
    )

    lines.append("## Models\n")
    lines.append("- `baseline`: image-only baseline")
    lines.append("- `plus_10k`: image-only + 10k natural-line context augmentation\n")

    lines.append("## Per-seed CER deltas\n")
    lines.append(
        "| seed | baseline CER | +10k CER | ΔCER | relative ΔCER | baseline WER | +10k WER | ΔWER | baseline exact | +10k exact | Δexact |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in delta_rows:
        lines.append(
            f"| {row['seed']} "
            f"| {float(row['baseline_cer']):.6f} "
            f"| {float(row['plus_10k_cer']):.6f} "
            f"| {float(row['delta_cer']):.6f} "
            f"| {float(row['relative_delta_cer']) * 100:.2f}% "
            f"| {float(row['baseline_wer']):.6f} "
            f"| {float(row['plus_10k_wer']):.6f} "
            f"| {float(row['delta_wer']):.6f} "
            f"| {float(row['baseline_exact']):.6f} "
            f"| {float(row['plus_10k_exact']):.6f} "
            f"| {float(row['delta_exact']):.6f} |"
        )

    lines.append("\n## Aggregate\n")
    lines.append(f"- mean ΔCER: `{delta_summary['mean_delta_cer']:.6f}`")
    lines.append(f"- std ΔCER: `{delta_summary['std_delta_cer']:.6f}`")
    lines.append(f"- mean relative ΔCER: `{delta_summary['mean_relative_delta_cer'] * 100:.2f}%`")
    lines.append(f"- improved CER seeds: `{delta_summary['improved_cer_seeds_n']}/3`")
    lines.append(f"- mean ΔWER: `{delta_summary['mean_delta_wer']:.6f}`")
    lines.append(f"- mean Δexact: `{delta_summary['mean_delta_exact']:.6f}`")
    lines.append(f"- interpretation: {delta_summary['interpretation']}\n")

    lines.append("## Model means\n")
    lines.append("| model | mean CER | std CER | mean WER | std WER | mean exact | std exact |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in aggregate_rows:
        lines.append(
            f"| {row['model']} "
            f"| {float(row['mean_cer']):.6f} "
            f"| {float(row['std_cer']):.6f} "
            f"| {float(row['mean_wer']):.6f} "
            f"| {float(row['std_wer']):.6f} "
            f"| {float(row['mean_exact']):.6f} "
            f"| {float(row['std_exact']):.6f} |"
        )

    lines.append("\n## Strict interpretation\n")
    lines.append(
        "This result supports the primary HTR claim only if +10k improves CER "
        "consistently across seeds. It does not prove graph-fusion superiority. "
        "It supports the data/context part of the final experimental protocol."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="outputs/final_result_package_v1/results_inventory.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/final_result_package_v1",
    )
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = read_inventory(inventory_path)

    model_rows: list[dict[str, Any]] = []

    for seed in ["42", "43", "44"]:
        baseline_summary = find_summary_path(
            inventory_rows,
            BASELINE_EXPERIMENT_IDS[seed],
        )
        plus_summary = find_summary_path(
            inventory_rows,
            PLUS10K_EXPERIMENT_IDS[seed],
        )

        model_rows.append(
            extract_summary_row("baseline", seed, baseline_summary)
        )
        model_rows.append(
            extract_summary_row("plus_10k", seed, plus_summary)
        )

    delta_rows = build_delta_rows(model_rows)
    aggregate_rows = aggregate_model_rows(model_rows)
    delta_summary = aggregate_delta_rows(delta_rows)

    write_csv(model_rows, out_dir / "seed_confirmation_table.csv")
    write_csv(delta_rows, out_dir / "seed_confirmation_deltas.csv")
    write_csv(aggregate_rows, out_dir / "seed_confirmation_model_means.csv")

    (out_dir / "seed_confirmation_summary.json").write_text(
        json.dumps(
            {
                "delta_summary": delta_summary,
                "model_means": aggregate_rows,
                "per_seed_deltas": delta_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_markdown(
        aggregate_rows=aggregate_rows,
        delta_rows=delta_rows,
        delta_summary=delta_summary,
        path=out_dir / "seed_confirmation_summary.md",
    )

    print(json.dumps(delta_summary, ensure_ascii=False, indent=2))
    print("wrote:", out_dir / "seed_confirmation_table.csv")
    print("wrote:", out_dir / "seed_confirmation_deltas.csv")
    print("wrote:", out_dir / "seed_confirmation_model_means.csv")
    print("wrote:", out_dir / "seed_confirmation_summary.json")
    print("wrote:", out_dir / "seed_confirmation_summary.md")


if __name__ == "__main__":
    main()
