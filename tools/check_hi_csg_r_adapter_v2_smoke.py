from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one_dir", required=True)
    parser.add_argument("--subset_dir", required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    one_dir = Path(args.one_dir)
    subset_dir = Path(args.subset_dir)
    evaluation = Path(args.eval_dir)
    one = load(one_dir / "val_summary.json")
    subset = load(subset_dir / "val_summary.json")
    train_summary = load(subset_dir / "train_summary.json")
    history = load(subset_dir / "history.json")
    correct = load(evaluation / "correct/summary.json")
    shuffle = load(evaluation / "shuffle/summary.json")
    conditions = {
        "one_sample_cer_zero": float(one["cer"]) == 0.0,
        "finite_losses": all(
            math.isfinite(float(row["train_total_loss"]))
            and math.isfinite(float(row["train_preservation_kl"]))
            for row in history
        ),
        "graph_gradient_nonzero": any(
            float(row["graph_adapter_grad_norm"]) > 0 for row in history
        ),
        "gate_nonconstant": float(subset["gate"]["std"]) > 0,
        "correct_not_worse_shuffle": float(correct["cer"]) <= float(shuffle["cer"]),
        "empty_correction_zero": (
            float(correct["correction"]["empty_max"]) == 0.0
            and all(float(row["empty_correction_max"]) == 0.0 for row in history)
        ),
        "backbone_unchanged": bool(train_summary["backbone_unchanged"]),
        "no_blank_collapse": float(correct["blank_ratio"]) < 0.99,
    }
    report = {
        "status": "PASS" if all(conditions.values()) else "STOP",
        "conditions": conditions,
        "one_sample": {
            "cer": one["cer"],
            "exact": one["exact"],
        },
        "subset": {
            "cer": subset["cer"],
            "exact": subset["exact"],
            "gate_std": subset["gate"]["std"],
            "correct_cer": correct["cer"],
            "shuffle_cer": shuffle["cer"],
            "empty_correction_max": correct["correction"]["empty_max"],
            "blank_ratio": correct["blank_ratio"],
        },
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# HI-CSG-R Late Correction v2 smoke gate",
        "",
        f"**Status:** `{report['status']}`",
        "",
        *[
            f"- {'PASS' if value else 'FAIL'} `{key}`"
            for key, value in conditions.items()
        ],
    ]
    (output / "smoke_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

