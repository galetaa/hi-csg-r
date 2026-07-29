from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_by_name(comparison: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((row for row in comparison["models"] if row["model"] == name), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--primary_bootstrap", required=True)
    parser.add_argument("--shuffle_bootstrap", required=True)
    parser.add_argument("--topology_bootstrap", required=True)
    parser.add_argument("--m0_name", default="M0-FT")
    parser.add_argument("--m3_name", default="M3")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    bootstrap = json.loads(Path(args.primary_bootstrap).read_text(encoding="utf-8"))
    shuffle = json.loads(Path(args.shuffle_bootstrap).read_text(encoding="utf-8"))
    topology = json.loads(Path(args.topology_bootstrap).read_text(encoding="utf-8"))
    baseline = model_by_name(comparison, args.m0_name)
    adapter = model_by_name(comparison, args.m3_name)
    if baseline is None or adapter is None:
        raise ValueError("Comparison must contain M0-FT and M3 model rows")

    improved_seeds = sum(
        adapter_value < baseline_value
        for baseline_value, adapter_value in zip(
            bootstrap["baseline_cer_by_seed"],
            bootstrap["adapter_cer_by_seed"],
            strict=True,
        )
    )
    max_domain_degradation = max(
        (float(value["delta_cer"]) for value in bootstrap["domains"].values()),
        default=0.0,
    )
    minimally_positive = (
        adapter["cer_mean"] < baseline["cer_mean"]
        and improved_seeds >= 2
        and bootstrap["ci95"][1] < 0
        and max_domain_degradation <= 0.005
        and float(shuffle["delta_cer"]) < 0
        and float(topology["delta_cer"]) < 0
    )
    classification = "supported" if minimally_positive else "exploratory"
    report = {
        "hypothesis_h4": classification,
        "criteria": {
            "mean_m3_better": adapter["cer_mean"] < baseline["cer_mean"],
            "improved_seeds": improved_seeds,
            "paired_ci_below_zero": bootstrap["ci95"][1] < 0,
            "max_domain_degradation": max_domain_degradation,
            "correct_better_than_shuffled": float(shuffle["delta_cer"]) < 0,
            "full_better_than_topology_off": float(topology["delta_cer"]) < 0,
        },
        "baseline": baseline,
        "adapter": adapter,
        "primary_bootstrap": bootstrap,
        "shuffle_bootstrap": shuffle,
        "topology_bootstrap": topology,
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = [
        "# CRNN-CTC + x-aligned HI-CSG-R: Final Report",
        "",
        f"H4 status: **{classification}**",
        "",
        f"- M0-FT CER: {baseline['cer_mean']:.6f} ± {baseline['cer_sd']:.6f}",
        f"- M3 CER: {adapter['cer_mean']:.6f} ± {adapter['cer_sd']:.6f}",
        f"- Paired ΔCER: {bootstrap['delta_cer']:.6f}",
        f"- 95% CI: [{bootstrap['ci95'][0]:.6f}, {bootstrap['ci95'][1]:.6f}]",
        f"- Better seeds: {improved_seeds}/{bootstrap['seeds']}",
        "",
        f"- Correct vs shuffled ΔCER: {shuffle['delta_cer']:.6f}",
        f"- Full vs topology-off ΔCER: {topology['delta_cer']:.6f}",
        "",
        "The conclusion is generated only from the preregistered minimum criteria.",
    ]
    (output / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
