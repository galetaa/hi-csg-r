from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.model_hi_csg_r_late_correction_v2 import (
    RISK_FEATURE_INDICES,
    RISK_FEATURE_NAMES,
)
from src.htr.xaligned_hi_csg_r import (
    FEATURE_NAMES,
    XAlignedFeatureNormalizer,
    load_feature_record,
    read_jsonl,
    resolve_path,
)


def audit_manifest(path: str | Path) -> dict[str, int | str]:
    manifest = Path(path)
    records = 0
    bins = 0
    nonfinite = 0
    name_mismatches = 0
    for row in read_jsonl(manifest):
        record = load_feature_record(
            resolve_path(str(row["xaligned_graph_npz"]), manifest)
        )
        values = np.asarray(record["features"], dtype=np.float32)
        mask = np.asarray(record["valid_mask"], dtype=bool)
        names = tuple(str(name) for name in record["feature_names"].tolist())
        records += 1
        bins += int(mask.sum())
        nonfinite += int(not np.isfinite(values[mask]).all())
        name_mismatches += int(names != FEATURE_NAMES)
    return {
        "manifest": str(manifest.resolve()),
        "records": records,
        "valid_bins": bins,
        "nonfinite_records": nonfinite,
        "feature_name_mismatches": name_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--dev_manifest", required=True)
    parser.add_argument("--holdout_manifest", required=True)
    parser.add_argument("--normalizer_out", required=True)
    parser.add_argument("--risk_stats_out", required=True)
    parser.add_argument("--audit_out", required=True)
    args = parser.parse_args()

    train_manifest = Path(args.train_manifest)
    normalizer = XAlignedFeatureNormalizer.fit(train_manifest)
    normalizer.to_path(args.normalizer_out)

    chunks: list[np.ndarray] = []
    record_count = 0
    bin_count = 0
    failures: list[str] = []
    for row in read_jsonl(train_manifest):
        try:
            record = load_feature_record(
                resolve_path(str(row["xaligned_graph_npz"]), train_manifest)
            )
            values = np.asarray(record["features"], dtype=np.float32)
            mask = np.asarray(record["valid_mask"], dtype=bool)
            values = values[mask]
            if not np.isfinite(values).all():
                raise ValueError("non-finite features")
            chunks.append(values)
            record_count += 1
            bin_count += len(values)
        except Exception as error:
            failures.append(f"{row.get('sample_id')}: {error}")
    if failures:
        raise ValueError(f"Feature scan failed for {len(failures)} records: {failures[:5]}")
    all_values = np.concatenate(chunks, axis=0)
    risk_values = all_values[:, RISK_FEATURE_INDICES]
    quantiles = {
        name: {
            "q05": float(np.quantile(risk_values[:, index], 0.05)),
            "q50": float(np.quantile(risk_values[:, index], 0.50)),
            "q95": float(np.quantile(risk_values[:, index], 0.95)),
        }
        for index, name in enumerate(RISK_FEATURE_NAMES)
    }
    # A degenerate q95 is expanded only for numerical scaling; the raw audit
    # still records the feature as inactive.
    q05 = [quantiles[name]["q05"] for name in RISK_FEATURE_NAMES]
    q95 = [
        max(quantiles[name]["q95"], quantiles[name]["q05"] + 1e-6)
        for name in RISK_FEATURE_NAMES
    ]
    risk_stats = {
        "feature_names": list(RISK_FEATURE_NAMES),
        "q05": q05,
        "q50": [quantiles[name]["q50"] for name in RISK_FEATURE_NAMES],
        "q95": q95,
        "raw_quantiles": quantiles,
        "train_manifest_sha256": normalizer.train_manifest_sha256,
    }
    risk_path = Path(args.risk_stats_out)
    risk_path.parent.mkdir(parents=True, exist_ok=True)
    risk_path.write_text(
        json.dumps(risk_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    standard_deviations = all_values.std(axis=0)
    inactive = [
        name
        for name, std in zip(FEATURE_NAMES, standard_deviations, strict=True)
        if float(std) < 1e-6
    ]
    split_counts = {
        "train": len(read_jsonl(args.train_manifest)),
        "dev": len(read_jsonl(args.dev_manifest)),
        "holdout": len(read_jsonl(args.holdout_manifest)),
    }
    split_audits = {
        "train": audit_manifest(args.train_manifest),
        "dev": audit_manifest(args.dev_manifest),
        "holdout": audit_manifest(args.holdout_manifest),
    }
    if any(
        value["nonfinite_records"] or value["feature_name_mismatches"]
        for value in split_audits.values()
    ):
        raise ValueError(f"Feature audit failed: {split_audits}")
    audit = {
        "status": "PASS",
        "normalizer_fit_split": "train",
        "train_records": record_count,
        "train_bins": bin_count,
        "split_counts": split_counts,
        "split_audits": split_audits,
        "inactive_features": inactive,
        "ambiguous_edge_fraction_active": "ambiguous_edge_fraction" not in inactive,
        "risk_stats": risk_stats,
        "failures": failures,
    }
    audit_path = Path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# HI-CSG-R adapter v2 feature audit",
        "",
        "**Status:** `PASS`",
        "",
        "Normalizer fitted only on `adapter_v2_train`.",
        "",
        "| Split | Records | Valid bins | Non-finite | Name mismatches |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "dev", "holdout"):
        value = split_audits[name]
        lines.append(
            f"| {name} | {value['records']} | {value['valid_bins']} | "
            f"{value['nonfinite_records']} | {value['feature_name_mismatches']} |"
        )
    lines.extend(
        [
            "",
            f"- inactive features: `{inactive}`",
            (
                "- ambiguous_edge_fraction active: "
                f"`{audit['ambiguous_edge_fraction_active']}`"
            ),
            f"- risk quantiles: `{quantiles}`",
        ]
    )
    audit_path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
