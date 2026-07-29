from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.xaligned_hi_csg_r import read_jsonl


def load_aligned(paths: list[str]) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    runs = []
    reference_ids: list[str] | None = None
    target_lengths: np.ndarray | None = None
    domains: list[str] | None = None
    for path in paths:
        rows = read_jsonl(path)
        by_id = {str(row["sample_id"]): row for row in rows}
        ids = sorted(by_id)
        if reference_ids is None:
            reference_ids = ids
            target_lengths = np.asarray(
                [int(by_id[sample_id]["target_chars"]) for sample_id in ids],
                dtype=np.float64,
            )
            domains = [str(by_id[sample_id]["dataset"]) for sample_id in ids]
        elif ids != reference_ids:
            raise ValueError(f"Sample IDs do not align in {path}")
        assert target_lengths is not None
        for index, sample_id in enumerate(ids):
            if int(by_id[sample_id]["target_chars"]) != target_lengths[index]:
                raise ValueError(f"Target length mismatch for {sample_id}")
        runs.append(
            np.asarray(
                [float(by_id[sample_id]["char_edits"]) for sample_id in ids],
                dtype=np.float64,
            )
        )
    if reference_ids is None or target_lengths is None or domains is None:
        raise ValueError("No prediction files supplied")
    return reference_ids, np.stack(runs), target_lengths, domains


def aggregate_delta(
    baseline: np.ndarray,
    adapter: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray | None = None,
) -> float:
    if indices is None:
        indices = np.arange(len(lengths))
    denominator = float(lengths[indices].sum())
    base_cer = float(baseline[:, indices].mean(axis=0).sum() / max(denominator, 1.0))
    adapter_cer = float(adapter[:, indices].mean(axis=0).sum() / max(denominator, 1.0))
    return adapter_cer - base_cer


def comparison(
    baseline_paths: list[str],
    adapter_paths: list[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    ids, baseline, lengths, domains = load_aligned(baseline_paths)
    adapter_ids, adapter, adapter_lengths, adapter_domains = load_aligned(adapter_paths)
    if ids != adapter_ids or not np.array_equal(lengths, adapter_lengths):
        raise ValueError("Baseline and adapter samples do not align")
    if domains != adapter_domains:
        raise ValueError("Baseline and adapter domains do not align")
    if baseline.shape[0] != adapter.shape[0]:
        raise ValueError("Baseline and adapter seed counts differ")

    base_seed_cer = baseline.sum(axis=1) / max(lengths.sum(), 1.0)
    adapter_seed_cer = adapter.sum(axis=1) / max(lengths.sum(), 1.0)
    mean_base_edits = baseline.mean(axis=0)
    mean_adapter_edits = adapter.mean(axis=0)
    per_sample_delta = mean_adapter_edits - mean_base_edits
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    sample_count = len(ids)
    for iteration in range(iterations):
        indices = rng.integers(0, sample_count, size=sample_count)
        deltas[iteration] = aggregate_delta(baseline, adapter, lengths, indices)
    observed = aggregate_delta(baseline, adapter, lengths)
    p_two_sided = min(
        1.0,
        2.0
        * min(
            float((np.sum(deltas <= 0.0) + 1) / (iterations + 1)),
            float((np.sum(deltas >= 0.0) + 1) / (iterations + 1)),
        ),
    )
    domain_results = {}
    domain_array = np.asarray(domains)
    for domain in sorted(set(domains)):
        indices = np.flatnonzero(domain_array == domain)
        domain_results[domain] = {
            "samples": len(indices),
            "delta_cer": aggregate_delta(baseline, adapter, lengths, indices),
        }
    return {
        "baseline_files": [str(Path(path).resolve()) for path in baseline_paths],
        "adapter_files": [str(Path(path).resolve()) for path in adapter_paths],
        "samples": sample_count,
        "seeds": baseline.shape[0],
        "baseline_cer_by_seed": base_seed_cer.tolist(),
        "adapter_cer_by_seed": adapter_seed_cer.tolist(),
        "baseline_mean_cer": float(base_seed_cer.mean()),
        "baseline_sd_cer": float(base_seed_cer.std(ddof=1)) if len(base_seed_cer) > 1 else 0.0,
        "adapter_mean_cer": float(adapter_seed_cer.mean()),
        "adapter_sd_cer": (
            float(adapter_seed_cer.std(ddof=1)) if len(adapter_seed_cer) > 1 else 0.0
        ),
        "delta_cer": observed,
        "relative_delta": observed / max(float(base_seed_cer.mean()), 1e-12),
        "bootstrap_iterations": iterations,
        "ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "p_two_sided": p_two_sided,
        "wins": int(np.sum(per_sample_delta < 0)),
        "losses": int(np.sum(per_sample_delta > 0)),
        "ties": int(np.sum(per_sample_delta == 0)),
        "domains": domain_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_predictions", nargs="+", required=True)
    parser.add_argument("--adapter_predictions", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = comparison(
        args.baseline_predictions,
        args.adapter_predictions,
        iterations=args.iterations,
        seed=args.seed,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
