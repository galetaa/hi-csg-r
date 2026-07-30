from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_comparison(value: str) -> tuple[str, dict[str, Any]]:
    if "=" not in value:
        raise ValueError("--comparison must use NAME=BOOTSTRAP_JSON")
    name, raw_path = value.split("=", 1)
    return name, json.loads(Path(raw_path).read_text(encoding="utf-8"))


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        raw = min(1.0, values[name] * (count - index))
        running = max(running, raw)
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="append", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    comparisons = dict(parse_comparison(value) for value in args.comparison)
    if args.primary not in comparisons:
        raise ValueError("--primary must name one supplied comparison")
    adjusted = holm_adjust(
        {name: float(value["p_two_sided"]) for name, value in comparisons.items()}
    )
    primary = comparisons[args.primary]
    relative_improvement = -float(primary["relative_delta"])
    seed_wins = sum(
        adapter < baseline
        for baseline, adapter in zip(
            primary["baseline_cer_by_seed"],
            primary["adapter_cer_by_seed"],
            strict=True,
        )
    )
    ci = [float(value) for value in primary["ci95"]]
    exact_gain = float(primary["delta_exact"])
    shuffle = comparisons.get("correct_vs_shuffle")
    correct_better_shuffle = bool(
        shuffle and float(shuffle["delta_cer"]) > 0.0
    )
    maximum_domain_degradation = max(
        (
            float(value["delta_cer"])
            for value in primary.get("domains", {}).values()
        ),
        default=0.0,
    )
    status = "not_supported"
    if (
        relative_improvement >= 0.05
        and seed_wins == 3
        and ci[1] < 0.0
        and exact_gain >= 0.01
        and correct_better_shuffle
        and maximum_domain_degradation <= 0.005
    ):
        status = "clear_superiority"
    elif (
        relative_improvement > 0.0
        and seed_wins >= 2
        and exact_gain >= 0.0
        and correct_better_shuffle
        and maximum_domain_degradation <= 0.005
    ):
        status = "minimal_or_partial_support"
    result = {
        "primary": args.primary,
        "status": status,
        "relative_cer_improvement": relative_improvement,
        "seed_wins": seed_wins,
        "correct_better_shuffle": correct_better_shuffle,
        "maximum_domain_cer_degradation": maximum_domain_degradation,
        "holm_adjusted_p": adjusted,
        "comparisons": comparisons,
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# HI-CSG-R Late Correction v2: final statistics",
        "",
        f"**Status:** `{status}`",
        "",
        "| Comparison | Baseline CER mean ± SD | Adapter CER mean ± SD | "
        "Delta CER | Relative delta | CI95 | p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in comparisons.items():
        lines.append(
            f"| {name} | {value['baseline_mean_cer']:.6f} ± "
            f"{value['baseline_sd_cer']:.6f} | "
            f"{value['adapter_mean_cer']:.6f} ± "
            f"{value['adapter_sd_cer']:.6f} | "
            f"{value['delta_cer']:+.6f} | {value['relative_delta']:+.3%} | "
            f"[{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] | "
            f"{value['p_two_sided']:.6g} | {adjusted[name]:.6g} |"
        )
    lines.extend(
        [
            "",
            f"- seed wins: `{seed_wins}/3`",
            f"- primary relative CER improvement: `{relative_improvement:.3%}`",
            f"- primary Exact delta: `{exact_gain:+.3%}`",
            f"- correct graph better shuffle: `{correct_better_shuffle}`",
            (
                "- maximum domain CER degradation: "
                f"`{maximum_domain_degradation:+.6f}`"
            ),
        ]
    )
    (output / "final_statistics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
