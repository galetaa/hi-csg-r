from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.xaligned_hi_csg_r import load_feature_record, read_jsonl, resolve_path


def sample_descriptor(row: dict[str, Any], manifest: Path, field: str) -> dict[str, Any]:
    record = load_feature_record(resolve_path(str(row[field]), manifest))
    features = np.asarray(record["features"], dtype=np.float32)
    return {
        "sample_id": str(row["sample_id"]),
        "dataset": str(row.get("dataset") or row.get("source_dataset") or "unknown"),
        "time_steps": int(record["time_steps"]),
        "ink_fraction": float(features[:, 0].mean()),
    }


def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    width_delta = abs(left["time_steps"] - right["time_steps"]) / max(
        left["time_steps"], right["time_steps"], 1
    )
    return width_delta + 4.0 * abs(left["ink_fraction"] - right["ink_fraction"])


def build_mapping(items: list[dict[str, Any]], seed: int) -> dict[str, str]:
    groups: defaultdict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = (
            item["dataset"],
            int(item["time_steps"] // 8),
            int(item["ink_fraction"] // 0.05),
        )
        groups[key].append(item)

    mapping: dict[str, str] = {}
    rng = random.Random(seed)
    singletons: list[dict[str, Any]] = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda item: item["sample_id"])
        if len(values) < 2:
            singletons.extend(values)
            continue
        rng.shuffle(values)
        for index, item in enumerate(values):
            mapping[item["sample_id"]] = values[(index + 1) % len(values)]["sample_id"]

    by_dataset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_dataset[item["dataset"]].append(item)
    for item in singletons:
        candidates = [
            candidate
            for candidate in by_dataset[item["dataset"]]
            if candidate["sample_id"] != item["sample_id"]
        ]
        if not candidates:
            raise ValueError(f"No within-domain shuffle donor for {item['sample_id']}")
        donor = min(
            candidates,
            key=lambda candidate: (distance(item, candidate), candidate["sample_id"]),
        )
        mapping[item["sample_id"]] = donor["sample_id"]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature_field", default="xaligned_graph_npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    items = [
        sample_descriptor(row, manifest, args.feature_field)
        for row in read_jsonl(manifest)
    ]
    mapping = build_mapping(items, args.seed)
    by_id = {item["sample_id"]: item for item in items}
    width_deltas = [
        abs(by_id[target]["time_steps"] - by_id[donor]["time_steps"])
        for target, donor in mapping.items()
    ]
    ink_deltas = [
        abs(by_id[target]["ink_fraction"] - by_id[donor]["ink_fraction"])
        for target, donor in mapping.items()
    ]
    payload = {
        "version": "hi_csg_r_matched_shuffle_v1",
        "seed": args.seed,
        "manifest": str(manifest.resolve()),
        "created_at": datetime.now(UTC).isoformat(),
        "matching": {
            "domain": "exact",
            "width_bucket_time_steps": 8,
            "ink_fraction_bucket": 0.05,
            "transcription_used": False,
            "model_error_used": False,
        },
        "summary": {
            "samples": len(items),
            "self_pairs": sum(target == donor for target, donor in mapping.items()),
            "mean_abs_time_step_delta": float(np.mean(width_deltas)) if width_deltas else 0.0,
            "mean_abs_ink_fraction_delta": (
                float(np.mean(ink_deltas)) if ink_deltas else 0.0
            ),
        },
        "mapping": mapping,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["summary"]["self_pairs"]:
        raise RuntimeError("Shuffle map contains self-pairs")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
