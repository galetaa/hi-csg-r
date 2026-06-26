from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


CANONICAL_DOMAINS = [
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks_clean",
    "school",
]


DOMAIN_ALIASES = {
    "cyrillic_handwriting": [
        "cyrillic_handwriting",
        "cyrillic",
        "Cyrillic",
        "Cyrillic Handwriting",
    ],
    "hkr_words": [
        "hkr_words",
        "hkr",
        "HKR",
        "HKR Words",
    ],
    "school_notebooks_clean": [
        "school_notebooks_clean",
        "school",
        "School",
        "School Notebooks",
        "school_notebooks",
    ],
    "school": [
        "school",
        "school_notebooks_clean",
        "School",
        "School Notebooks",
        "school_notebooks",
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def get_grouped(summary: dict[str, Any]) -> dict[str, Any]:
    grouped = summary.get("grouped")
    if isinstance(grouped, dict):
        return grouped

    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("grouped"), dict):
        return metrics["grouped"]

    by_dataset = summary.get("by_dataset")
    if isinstance(by_dataset, dict):
        return by_dataset

    domains = summary.get("domains")
    if isinstance(domains, dict):
        return domains

    return {}


def weighted_rollup(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    n_total = sum(float(row.get("n", 0.0) or 0.0) for row in rows)
    if n_total <= 0:
        return None

    out: dict[str, Any] = {"n": n_total}
    for metric in ["cer", "wer", "exact"]:
        vals = [
            (safe_float(row.get(metric)), safe_float(row.get("n")))
            for row in rows
        ]
        vals = [
            (value, n)
            for value, n in vals
            if value is not None and n is not None
        ]
        if vals:
            out[metric] = sum(value * n for value, n in vals) / n_total
    return out


def find_domain_data(grouped: dict[str, Any], domain: str) -> dict[str, Any] | None:
    aliases = DOMAIN_ALIASES.get(domain, [domain])

    for alias in aliases:
        if alias in grouped and isinstance(grouped[alias], dict):
            return grouped[alias]

    lower_map = {str(key).lower(): key for key in grouped.keys()}
    for alias in aliases:
        key = lower_map.get(alias.lower())
        if key is not None and isinstance(grouped[key], dict):
            return grouped[key]

    for alias in aliases:
        matches = [
            value
            for key, value in grouped.items()
            if isinstance(value, dict) and str(key).split("|", 1)[0].lower() == alias.lower()
        ]
        rolled = weighted_rollup(matches)
        if rolled is not None:
            return rolled

    return None


def seed_table_domain_value(
    row: dict[str, str],
    domain: str,
    metric: str,
) -> float | None:
    for alias in DOMAIN_ALIASES.get(domain, [domain]):
        value = safe_float(row.get(f"{alias}_{metric}"))
        if value is not None:
            return value
    return None


def extract_domain_rows_from_seed_table(seed_table: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in seed_table:
        model = row["model"]
        seed = row["seed"]
        summary_path = Path(row["summary_path"])
        summary = read_json(summary_path)
        grouped = get_grouped(summary)

        for domain in CANONICAL_DOMAINS:
            data = find_domain_data(grouped, domain)

            if data is None:
                cer = seed_table_domain_value(row, domain, "cer")
                wer = seed_table_domain_value(row, domain, "wer")
                exact = seed_table_domain_value(row, domain, "exact")
                n = seed_table_domain_value(row, domain, "n")
            else:
                cer = safe_float(data.get("cer"))
                wer = safe_float(data.get("wer"))
                exact = safe_float(data.get("exact"))
                n = safe_float(data.get("n"))

            if cer is None:
                continue

            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "domain": domain,
                    "n": n if n is not None else "",
                    "cer": cer,
                    "wer": wer if wer is not None else "",
                    "exact": exact if exact is not None else "",
                    "summary_path": str(summary_path),
                }
            )

    return rows


def build_domain_delta_rows(domain_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["seed"]), str(row["domain"]), str(row["model"])): row
        for row in domain_rows
    }

    seeds = sorted({str(row["seed"]) for row in domain_rows})
    domains = sorted({str(row["domain"]) for row in domain_rows})

    delta_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for domain in domains:
            baseline = by_key.get((seed, domain, "baseline"))
            plus = by_key.get((seed, domain, "plus_10k"))

            if baseline is None or plus is None:
                continue

            baseline_cer = safe_float(baseline.get("cer"))
            plus_cer = safe_float(plus.get("cer"))
            baseline_wer = safe_float(baseline.get("wer"))
            plus_wer = safe_float(plus.get("wer"))
            baseline_exact = safe_float(baseline.get("exact"))
            plus_exact = safe_float(plus.get("exact"))

            if baseline_cer is None or plus_cer is None:
                continue

            row: dict[str, Any] = {
                "seed": seed,
                "domain": domain,
                "n_baseline": baseline.get("n", ""),
                "n_plus_10k": plus.get("n", ""),
                "baseline_cer": baseline_cer,
                "plus_10k_cer": plus_cer,
                "delta_cer": plus_cer - baseline_cer,
                "relative_delta_cer": (plus_cer - baseline_cer) / max(baseline_cer, 1e-12),
            }

            if baseline_wer is not None and plus_wer is not None:
                row["baseline_wer"] = baseline_wer
                row["plus_10k_wer"] = plus_wer
                row["delta_wer"] = plus_wer - baseline_wer

            if baseline_exact is not None and plus_exact is not None:
                row["baseline_exact"] = baseline_exact
                row["plus_10k_exact"] = plus_exact
                row["delta_exact"] = plus_exact - baseline_exact

            delta_rows.append(row)

    return delta_rows


def build_domain_summary(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    domains = sorted({str(row["domain"]) for row in delta_rows})
    summary_rows: list[dict[str, Any]] = []

    for domain in domains:
        rows = [row for row in delta_rows if row["domain"] == domain]

        delta_cer = [float(row["delta_cer"]) for row in rows]
        rel_delta_cer = [float(row["relative_delta_cer"]) for row in rows]

        baseline_cer = [float(row["baseline_cer"]) for row in rows]
        plus_cer = [float(row["plus_10k_cer"]) for row in rows]

        delta_wer = [
            value
            for row in rows
            if (value := safe_float(row.get("delta_wer"))) is not None
        ]
        delta_exact = [
            value
            for row in rows
            if (value := safe_float(row.get("delta_exact"))) is not None
        ]

        improved_cer_n = sum(1 for value in delta_cer if value < 0)
        worsened_cer_n = sum(1 for value in delta_cer if value > 0)

        if improved_cer_n == len(rows):
            interpretation = "improves in all available seeds"
        elif improved_cer_n >= 2:
            interpretation = "mostly improves, but not fully seed-stable"
        elif improved_cer_n == 1:
            interpretation = "weak or unstable domain effect"
        else:
            interpretation = "does not improve this domain"

        row: dict[str, Any] = {
            "domain": domain,
            "seeds_n": len(rows),
            "mean_baseline_cer": mean(baseline_cer),
            "std_baseline_cer": stdev(baseline_cer),
            "mean_plus_10k_cer": mean(plus_cer),
            "std_plus_10k_cer": stdev(plus_cer),
            "mean_delta_cer": mean(delta_cer),
            "std_delta_cer": stdev(delta_cer),
            "mean_relative_delta_cer": mean(rel_delta_cer),
            "std_relative_delta_cer": stdev(rel_delta_cer),
            "improved_cer_seeds_n": improved_cer_n,
            "worsened_cer_seeds_n": worsened_cer_n,
            "interpretation": interpretation,
        }

        if delta_wer:
            row["mean_delta_wer"] = mean(delta_wer)
            row["std_delta_wer"] = stdev(delta_wer)
        else:
            row["mean_delta_wer"] = ""
            row["std_delta_wer"] = ""

        if delta_exact:
            row["mean_delta_exact"] = mean(delta_exact)
            row["std_delta_exact"] = stdev(delta_exact)
        else:
            row["mean_delta_exact"] = ""
            row["std_delta_exact"] = ""

        summary_rows.append(row)

    return summary_rows


def build_overall_interpretation(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not summary_rows:
        return {
            "verdict": "FAIL",
            "claim": "No domain-wise results were found.",
        }

    all_domains_improve_all_seeds = all(
        int(row["improved_cer_seeds_n"]) == int(row["seeds_n"])
        for row in summary_rows
    )

    any_domain_worse_all_seeds = any(
        int(row["worsened_cer_seeds_n"]) == int(row["seeds_n"])
        for row in summary_rows
    )

    school_rows = [
        row for row in summary_rows
        if row["domain"] in {"school", "school_notebooks_clean"}
    ]

    school_strong = any(
        float(row["mean_delta_cer"]) < 0 and int(row["improved_cer_seeds_n"]) >= 2
        for row in school_rows
    )

    if all_domains_improve_all_seeds:
        verdict = "STRONG_ACROSS_DOMAINS"
        claim = (
            "Natural-line context improves CER across all reported domains "
            "and all available seeds."
        )
    elif school_strong and not any_domain_worse_all_seeds:
        verdict = "STRONG_HARD_DOMAIN"
        claim = (
            "Natural-line context is especially supported for the School/hard domain, "
            "with no domain consistently worsened across all seeds."
        )
    elif school_strong:
        verdict = "DOMAIN_SPECIFIC"
        claim = (
            "Natural-line context improves the School/hard domain, but the effect "
            "is domain-specific and should not be claimed as uniformly positive."
        )
    else:
        verdict = "WEAK_DOMAINWISE"
        claim = (
            "Overall improvement exists, but domain-wise evidence is weak or unstable."
        )

    return {
        "verdict": verdict,
        "claim": claim,
    }


def write_markdown(
    delta_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    overall: dict[str, Any],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Domain-wise seed confirmation v1\n")
    lines.append("## Purpose\n")
    lines.append(
        "This report checks whether the +10k natural-line context improvement "
        "is distributed across domains or concentrated in the hard notebook domain.\n"
    )

    lines.append("## Overall domain-wise verdict\n")
    lines.append(f"- verdict: `{overall['verdict']}`")
    lines.append(f"- claim: {overall['claim']}\n")

    lines.append("## Domain summary\n")
    lines.append(
        "| domain | seeds | mean baseline CER | mean +10k CER | mean ΔCER | relative ΔCER | improved seeds | interpretation |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")

    for row in summary_rows:
        lines.append(
            f"| {row['domain']} "
            f"| {int(row['seeds_n'])} "
            f"| {float(row['mean_baseline_cer']):.6f} "
            f"| {float(row['mean_plus_10k_cer']):.6f} "
            f"| {float(row['mean_delta_cer']):.6f} "
            f"| {float(row['mean_relative_delta_cer']) * 100:.2f}% "
            f"| {int(row['improved_cer_seeds_n'])}/{int(row['seeds_n'])} "
            f"| {row['interpretation']} |"
        )

    lines.append("\n## Per-seed domain deltas\n")
    lines.append(
        "| seed | domain | baseline CER | +10k CER | ΔCER | relative ΔCER |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|")

    for row in sorted(delta_rows, key=lambda x: (x["domain"], x["seed"])):
        lines.append(
            f"| {row['seed']} "
            f"| {row['domain']} "
            f"| {float(row['baseline_cer']):.6f} "
            f"| {float(row['plus_10k_cer']):.6f} "
            f"| {float(row['delta_cer']):.6f} "
            f"| {float(row['relative_delta_cer']) * 100:.2f}% |"
        )

    lines.append("\n## Strict interpretation\n")
    lines.append(
        "If the improvement is concentrated in School/hard-domain data, "
        "the final thesis claim should be phrased as hard-domain/context improvement, "
        "not as universal improvement across all Russian handwriting datasets."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_table",
        default="outputs/final_result_package_v1/seed_confirmation_table.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/final_result_package_v1",
    )
    args = parser.parse_args()

    seed_table_path = Path(args.seed_table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_table = read_csv(seed_table_path)

    domain_rows = extract_domain_rows_from_seed_table(seed_table)
    delta_rows = build_domain_delta_rows(domain_rows)
    summary_rows = build_domain_summary(delta_rows)
    overall = build_overall_interpretation(summary_rows)

    write_csv(domain_rows, out_dir / "domainwise_seed_metrics.csv")
    write_csv(delta_rows, out_dir / "domainwise_seed_deltas.csv")
    write_csv(summary_rows, out_dir / "domainwise_seed_summary.csv")

    report = {
        "overall": overall,
        "domain_summary": summary_rows,
        "per_seed_deltas": delta_rows,
    }

    (out_dir / "domainwise_seed_confirmation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown(
        delta_rows=delta_rows,
        summary_rows=summary_rows,
        overall=overall,
        path=out_dir / "domainwise_seed_confirmation.md",
    )

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("wrote:", out_dir / "domainwise_seed_metrics.csv")
    print("wrote:", out_dir / "domainwise_seed_deltas.csv")
    print("wrote:", out_dir / "domainwise_seed_summary.csv")
    print("wrote:", out_dir / "domainwise_seed_confirmation.json")
    print("wrote:", out_dir / "domainwise_seed_confirmation.md")


if __name__ == "__main__":
    main()
