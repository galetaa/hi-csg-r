from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from src.htr.xaligned_hi_csg_r import (
    FEATURE_BUILDER_VERSION,
    FEATURE_NAMES,
    QUALITY_FEATURE_INDICES,
    XAlignedFeatureNormalizer,
    compute_output_steps,
    load_feature_record,
    read_jsonl,
    resolve_path,
    verify_normalizer_for_manifest,
)

ALLOWED_NPZ_FIELDS = {
    "features",
    "raw_features",
    "quality",
    "valid_mask",
    "time_steps",
    "original_width",
    "feature_names",
    "quality_feature_names",
    "sample_id",
    "graph_version",
    "feature_version",
    "feature_builder_version",
    "source_image_sha1",
    "binarization",
    "diagnostics_json",
}
TARGET_DERIVED_MARKERS = {"text", "target", "transcription", "label", "dataset", "split"}


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def audit_manifest(path: Path, feature_field: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    count = 0
    zero_bins = 0
    total_bins = 0
    total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    total_sq = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    minimum = np.full(len(FEATURE_NAMES), np.inf, dtype=np.float64)
    maximum = np.full(len(FEATURE_NAMES), -np.inf, dtype=np.float64)
    feature_zero_count = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    failures: list[dict[str, str]] = []
    time_steps: list[int] = []
    sample_means: list[np.ndarray] = []
    widths: list[float] = []
    datasets: list[str] = []
    target_field_violations: Counter[str] = Counter()
    count_deltas: defaultdict[str, list[float]] = defaultdict(list)

    for row in rows:
        sample_id = str(row.get("sample_id"))
        try:
            feature_path = resolve_path(str(row[feature_field]), path)
            with np.load(feature_path, allow_pickle=False) as archive:
                archive_fields = set(archive.files)
            for field in archive_fields - ALLOWED_NPZ_FIELDS:
                if any(marker in field.lower() for marker in TARGET_DERIVED_MARKERS):
                    target_field_violations[field] += 1

            record = load_feature_record(feature_path)
            features = np.asarray(record["features"], dtype=np.float64)
            raw_features = np.asarray(record["raw_features"], dtype=np.float64)
            quality = np.asarray(record["quality"], dtype=np.float64)
            mask = np.asarray(record["valid_mask"], dtype=bool)
            names = tuple(str(value) for value in record["feature_names"].tolist())
            expected_steps = compute_output_steps(int(record["original_width"]))
            if names != FEATURE_NAMES:
                raise ValueError("feature_names mismatch")
            if features.shape != (expected_steps, len(FEATURE_NAMES)):
                raise ValueError(
                    f"shape={features.shape}, expected={(expected_steps, len(FEATURE_NAMES))}"
                )
            if raw_features.shape != features.shape:
                raise ValueError("raw_features shape mismatch")
            if str(record["feature_builder_version"]) != FEATURE_BUILDER_VERSION:
                raise ValueError(
                    "stale feature_builder_version="
                    f"{record['feature_builder_version']}; expected={FEATURE_BUILDER_VERSION}"
                )
            if mask.shape != (expected_steps,) or not mask.all():
                raise ValueError("main split feature records must contain all-valid real bins")
            if quality.shape != (expected_steps, len(QUALITY_FEATURE_INDICES)):
                raise ValueError("quality shape mismatch")
            if not np.allclose(quality, features[:, QUALITY_FEATURE_INDICES]):
                raise ValueError("quality values differ from features 18-20")
            if not np.isfinite(features).all():
                raise ValueError("NaN or Inf detected")

            values = features[mask]
            count += 1
            total_bins += len(values)
            zero_bins += int(np.all(values == 0.0, axis=1).sum())
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
            feature_zero_count += np.isclose(values, 0.0).sum(axis=0)
            time_steps.append(expected_steps)
            sample_means.append(values.mean(axis=0))
            widths.append(float(record["original_width"]))
            datasets.append(str(row.get("dataset") or row.get("source_dataset") or "unknown"))

            diagnostics = record.get("diagnostics", {})
            image_width = int(record["original_width"])
            image_height = int(diagnostics.get("image_height", 0))
            bin_areas = np.asarray(
                [
                    image_height
                    * max(
                        min(
                            int(np.floor((index + 1) * image_width / expected_steps)),
                            image_width,
                        )
                        - int(np.floor(index * image_width / expected_steps)),
                        1,
                    )
                    for index in range(expected_steps)
                ],
                dtype=np.float64,
            )
            reconstructed_edge_length = float(
                np.sum(raw_features[:, 2] * bin_areas)
            )
            count_deltas["node"].append(
                abs(
                    float(diagnostics.get("graph_nodes", 0.0))
                    - float(diagnostics.get("local_node_sum", 0.0))
                )
            )
            count_deltas["endpoint"].append(
                abs(
                    float(diagnostics.get("endpoint_nodes", 0.0))
                    - float(diagnostics.get("local_endpoint_sum", 0.0))
                )
            )
            count_deltas["junction"].append(
                abs(
                    float(diagnostics.get("junction_nodes", 0.0))
                    - float(diagnostics.get("local_junction_sum", 0.0))
                )
            )
            count_deltas["edge_length"].append(
                abs(
                    float(diagnostics.get("graph_edge_length", 0.0))
                    - float(diagnostics.get("local_edge_length_sum", 0.0))
                )
            )
            count_deltas["edge_length_record"].append(
                abs(
                    reconstructed_edge_length
                    - float(diagnostics.get("local_edge_length_sum", 0.0))
                )
            )
        except Exception as error:
            failures.append({"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"})

    means = total / max(total_bins, 1)
    variance = np.maximum(total_sq / max(total_bins, 1) - np.square(means), 0.0)
    sample_matrix = (
        np.stack(sample_means, axis=0)
        if sample_means
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    )
    width_values = np.asarray(widths, dtype=np.float64)
    dataset_names = sorted(set(datasets))
    correlations: dict[str, dict[str, float | None]] = {}
    for feature_index, name in enumerate(FEATURE_NAMES):
        values = sample_matrix[:, feature_index] if len(sample_matrix) else np.asarray([])
        row = {"width": _safe_corr(values, width_values)}
        for dataset in dataset_names:
            indicator = np.asarray([value == dataset for value in datasets], dtype=np.float64)
            row[f"dataset::{dataset}"] = _safe_corr(values, indicator)
        correlations[name] = row

    distributions = {
        name: {
            "min": float(minimum[index]) if count else None,
            "max": float(maximum[index]) if count else None,
            "mean": float(means[index]) if count else None,
            "std": float(math_value),
            "zero_fraction": float(feature_zero_count[index] / max(total_bins, 1)),
        }
        for index, (name, math_value) in enumerate(zip(FEATURE_NAMES, np.sqrt(variance), strict=True))
    }
    consistency = {
        key: {
            "max_abs_delta": max(values, default=0.0),
            "mean_abs_delta": sum(values) / max(len(values), 1),
        }
        for key, values in count_deltas.items()
    }
    consistency_pass = all(
        values["max_abs_delta"] <= (
            1e-2 if key.startswith("edge_length") else 1e-6
        )
        for key, values in consistency.items()
    )
    return {
        "manifest": str(path),
        "expected_n": len(rows),
        "audited_n": count,
        "failures": failures,
        "total_bins": total_bins,
        "zero_bin_fraction": zero_bins / max(total_bins, 1),
        "time_steps": {
            "min": min(time_steps, default=0),
            "max": max(time_steps, default=0),
            "mean": sum(time_steps) / max(len(time_steps), 1),
        },
        "distributions": distributions,
        "sample_mean_correlations": correlations,
        "target_field_violations": dict(target_field_violations),
        "count_consistency": consistency,
        "count_consistency_pass": consistency_pass,
        "status": (
            "PASS"
            if count == len(rows)
            and not failures
            and not target_field_violations
            and consistency_pass
            else "FAIL"
        ),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# X-Aligned HI-CSG-R Feature Audit v1",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "| split | expected | audited | failures | zero bins | T range |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in summary["manifests"]:
        lines.append(
            f"| `{Path(item['manifest']).stem}` | {item['expected_n']} | {item['audited_n']} | "
            f"{len(item['failures'])} | {item['zero_bin_fraction']:.4f} | "
            f"{item['time_steps']['min']}..{item['time_steps']['max']} |"
        )
    lines.extend(
        [
            "",
            "## Train Feature Distributions",
            "",
            "| feature | min | max | mean | std | zero fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    train = summary["manifests"][0]
    for name in FEATURE_NAMES:
        values = train["distributions"][name]
        lines.append(
            f"| `{name}` | {values['min']:.6g} | {values['max']:.6g} | "
            f"{values['mean']:.6g} | {values['std']:.6g} | "
            f"{values['zero_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Count Consistency",
            "",
            "| split | quantity | max abs delta | mean abs delta |",
            "|---|---|---:|---:|",
        ]
    )
    for item in summary["manifests"]:
        for name, values in item["count_consistency"].items():
            lines.append(
                f"| `{Path(item['manifest']).stem}` | `{name}` | "
                f"{values['max_abs_delta']:.6g} | "
                f"{values['mean_abs_delta']:.6g} |"
            )
    lines.extend(
        [
            "",
            "Normalizer provenance is accepted only when its stored train-manifest SHA256 "
            "matches the first audited manifest.",
        ]
    )
    return "\n".join(lines) + "\n"


def fit_normalizer(args: argparse.Namespace) -> None:
    normalizer = XAlignedFeatureNormalizer.fit(
        args.manifest,
        feature_field=args.feature_field,
        clip_value=args.clip_value,
    )
    normalizer.to_path(args.out)
    print(json.dumps({"status": "PASS", "out": args.out}, indent=2))


def run_audit(args: argparse.Namespace) -> None:
    manifests = [audit_manifest(Path(value), args.feature_field) for value in args.manifests]
    normalizer = XAlignedFeatureNormalizer.from_path(args.normalizer)
    verify_normalizer_for_manifest(normalizer, args.manifests[0])
    summary = {
        "status": "PASS" if all(item["status"] == "PASS" for item in manifests) else "FAIL",
        "normalizer": args.normalizer,
        "normalizer_train_manifest_sha256": normalizer.train_manifest_sha256,
        "manifests": manifests,
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "feature_audit.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_dir": str(output)}, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit-normalizer")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--feature_field", default="xaligned_graph_npz")
    fit.add_argument("--clip_value", type=float, default=5.0)
    fit.add_argument("--out", required=True)
    fit.set_defaults(func=fit_normalizer)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifests", nargs="+", required=True)
    audit.add_argument("--normalizer", required=True)
    audit.add_argument("--feature_field", default="xaligned_graph_npz")
    audit.add_argument("--out_dir", default="outputs/htr_adapter_v1/feature_audit")
    audit.set_defaults(func=run_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
