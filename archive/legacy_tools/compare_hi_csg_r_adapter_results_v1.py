from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    return name, candidate


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def validation_gate(
    grouped: dict[str, list[dict[str, Any]]],
    domain_paths: list[str],
    history_path: str,
) -> dict[str, Any]:
    required = ("M0-FT", "M2", "M3", "M3-shuffle")
    missing = [name for name in required if name not in grouped]
    if missing:
        raise ValueError(f"Validation gate is missing model summaries: {missing}")
    summaries = {name: grouped[name][0] for name in required}
    domains = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in (parse_named_path(value) for value in domain_paths)
    }
    if "M0-FT" not in domains or "M3" not in domains:
        raise ValueError("Validation gate requires M0-FT and M3 domain summaries")
    shared_domains = sorted(set(domains["M0-FT"]) & set(domains["M3"]))
    domain_deltas = {
        domain: float(domains["M3"][domain]["cer"])
        - float(domains["M0-FT"][domain]["cer"])
        for domain in shared_domains
    }
    history = [
        json.loads(line)
        for line in Path(history_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    joint = [row for row in history if row.get("stage") == "joint"]
    m0_cer = float(summaries["M0-FT"]["cer"])
    m3_cer = float(summaries["M3"]["cer"])
    conditions = {
        "relative_cer_improvement_at_least_2_percent": (
            (m0_cer - m3_cer) / max(m0_cer, 1e-12) >= 0.02
        ),
        "at_least_two_core_domains_not_worse": (
            sum(delta <= 0 for delta in domain_deltas.values()) >= 2
        ),
        "no_core_domain_degrades_over_0_005": (
            max(domain_deltas.values(), default=0.0) <= 0.005
        ),
        "correct_graph_better_than_shuffle": (
            m3_cer < float(summaries["M3-shuffle"]["cer"])
        ),
        "gate_has_variation": float((summaries["M3"].get("gate") or {}).get("std", 0.0))
        > 0,
        "graph_gradient_nonzero": any(
            float(row.get("graph_adapter_grad_norm", 0.0)) > 0 for row in joint
        ),
        "full_better_than_topology_off": m3_cer < float(summaries["M2"]["cer"]),
    }
    return {
        "status": "PASS" if all(conditions.values()) else "STOP",
        "conditions": conditions,
        "domain_deltas": domain_deltas,
        "m0_ft_cer": m0_cer,
        "m3_cer": m3_cer,
        "relative_improvement": (m0_cer - m3_cer) / max(m0_cer, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="MODEL=summary.json")
    parser.add_argument("--bootstrap", action="append", default=[], help="NAME=result.json")
    parser.add_argument("--validation_gate", action="store_true")
    parser.add_argument(
        "--domain_summary",
        action="append",
        default=[],
        help="MODEL=domain_summary.json",
    )
    parser.add_argument("--m3_history")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in args.run:
        name, path = parse_named_path(value)
        grouped[name].append(json.loads(path.read_text(encoding="utf-8")))
    table = []
    for name, summaries in sorted(grouped.items()):
        row: dict[str, Any] = {"model": name, "seeds": len(summaries)}
        for metric in ("cer", "wer", "exact", "macro_cer"):
            values = np.asarray([float(summary[metric]) for summary in summaries])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        table.append(row)

    bootstrap = {}
    p_values = {}
    for value in args.bootstrap:
        name, path = parse_named_path(value)
        result = json.loads(path.read_text(encoding="utf-8"))
        bootstrap[name] = result
        p_values[name] = float(result["p_two_sided"])
    adjusted = holm_adjust(p_values)
    for name, value in adjusted.items():
        bootstrap[name]["holm_adjusted_p"] = value

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    gate = None
    if args.validation_gate:
        if not args.m3_history:
            raise ValueError("--validation_gate requires --m3_history")
        gate = validation_gate(grouped, args.domain_summary, args.m3_history)
    payload = {"models": table, "comparisons": bootstrap, "validation_gate": gate}
    (output / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = list(table[0]) if table else ["model", "seeds"]
    with (output / "model_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "# HI-CSG-R Adapter Comparison",
        "",
        "| model | seeds | CER mean | CER SD | WER mean | Exact mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['seeds']} | {row['cer_mean']:.6f} | "
            f"{row['cer_sd']:.6f} | {row['wer_mean']:.6f} | {row['exact_mean']:.6f} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if gate is not None:
        (output / "validation_gate.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if gate is not None and gate["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
