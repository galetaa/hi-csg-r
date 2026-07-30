from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    return json.loads(candidate.read_text(encoding="utf-8"))


def compare(
    correct: dict[str, Any],
    shuffle: dict[str, Any],
    zero: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    baseline = correct["baseline"]
    baseline_cer = float(baseline["cer"])
    correct_cer = float(correct["cer"])
    relative_improvement = (baseline_cer - correct_cer) / max(baseline_cer, 1e-12)
    baseline_domains = correct["grouped"]["baseline_domain"]
    correct_domains = correct["grouped"]["domain"]
    shared = sorted(set(baseline_domains) & set(correct_domains))
    domain_deltas = {
        name: float(correct_domains[name]["cer"])
        - float(baseline_domains[name]["cer"])
        for name in shared
    }
    common = {
        "correct_better_shuffle": correct_cer < float(shuffle["cer"]),
        "empty_correction_invariant": (
            float(correct["correction"]["empty_max"]) < 1e-7
        ),
    }
    if stage == "dev":
        conditions = {
            "relative_cer_improvement_at_least_1_percent": (
                relative_improvement >= 0.01
            ),
            **common,
            "no_domain_degrades_over_0_003": (
                max(domain_deltas.values(), default=0.0) <= 0.003
            ),
            "exact_drop_at_most_0_005": (
                float(correct["exact"]) - float(baseline["exact"]) >= -0.005
            ),
        }
    elif stage == "holdout":
        conditions = {
            "relative_cer_improvement_at_least_2_percent": (
                relative_improvement >= 0.02
            ),
            **common,
            "at_least_two_domains_not_worse": (
                sum(delta <= 0.0 for delta in domain_deltas.values()) >= 2
            ),
            "no_domain_degrades_over_0_003": (
                max(domain_deltas.values(), default=0.0) <= 0.003
            ),
            "exact_not_below_baseline": (
                float(correct["exact"]) >= float(baseline["exact"])
            ),
            "wer_degradation_at_most_0_003": (
                float(correct["wer"]) - float(baseline["wer"]) <= 0.003
            ),
        }
    else:
        raise ValueError(f"Unsupported comparison stage: {stage}")
    return {
        "stage": stage,
        "status": "PASS" if all(conditions.values()) else "STOP",
        "conditions": conditions,
        "baseline": {
            "cer": baseline_cer,
            "wer": float(baseline["wer"]),
            "exact": float(baseline["exact"]),
        },
        "correct": {
            "cer": correct_cer,
            "wer": float(correct["wer"]),
            "exact": float(correct["exact"]),
        },
        "shuffle": {
            "cer": float(shuffle["cer"]),
            "wer": float(shuffle["wer"]),
            "exact": float(shuffle["exact"]),
        },
        "zero": {
            "cer": float(zero["cer"]),
            "wer": float(zero["wer"]),
            "exact": float(zero["exact"]),
        },
        "relative_cer_improvement": relative_improvement,
        "absolute_cer_delta": correct_cer - baseline_cer,
        "correct_vs_shuffle_cer": correct_cer - float(shuffle["cer"]),
        "domain_cer_deltas": domain_deltas,
        "alpha": float(correct["alpha"]),
        "empty_correction_max": float(correct["correction"]["empty_max"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "holdout"], required=True)
    parser.add_argument("--correct", required=True)
    parser.add_argument("--shuffle", required=True)
    parser.add_argument("--zero", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    result = compare(
        load_summary(args.correct),
        load_summary(args.shuffle),
        load_summary(args.zero),
        stage=args.stage,
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    name = f"{args.stage}_decision"
    (output / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# HI-CSG-R Late Correction v2 {args.stage} decision",
        "",
        f"**Status:** `{result['status']}`",
        "",
        "| Model | CER | WER | Exact |",
        "|---|---:|---:|---:|",
    ]
    for model in ("baseline", "correct", "shuffle", "zero"):
        values = result[model]
        lines.append(
            f"| {model} | {values['cer']:.6f} | {values['wer']:.6f} | "
            f"{values['exact']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- relative CER improvement: `{result['relative_cer_improvement']:.4%}`",
            f"- correct-shuffle CER delta: `{result['correct_vs_shuffle_cer']:.6f}`",
            f"- domain CER deltas: `{result['domain_cer_deltas']}`",
            f"- empty correction max: `{result['empty_correction_max']:.3e}`",
            "",
            "## Conditions",
            "",
        ]
    )
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} `{key}`"
        for key, value in result["conditions"].items()
    )
    (output / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

