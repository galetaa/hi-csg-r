from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.extract_htr_graph_features import (
    FEATURE_KEYS,
    process_one,
    read_jsonl,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--min_object_size", type=int, default=0)
    parser.add_argument("--sauvola_window", type=int, default=25)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)

    out_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            task = (
                idx,
                row,
                str(manifest_path),
                args.dataset,
                args.split,
                args.min_object_size,
                args.sauvola_window,
            )
            feat = process_one(task)

            enriched = dict(row)
            enriched["sample_id"] = feat["sample_id"]
            enriched["dataset"] = args.dataset
            enriched["source_dataset"] = row.get("source_dataset", row.get("dataset", args.dataset))
            enriched["split"] = args.split
            enriched["text"] = feat["text"]
            enriched["graph_features"] = [float(feat[k]) for k in FEATURE_KEYS]
            enriched["graph_feature_names"] = list(FEATURE_KEYS)
            enriched["graph_warning_count"] = int(feat.get("warning_count", 0))
            enriched["graph_binarization"] = feat.get("binarization", "")
            enriched["graph_feature_source"] = "distorted_image"

            out_rows.append(enriched)

        except Exception as exc:
            failures.append(
                {
                    "idx": idx,
                    "sample_id": row.get("sample_id"),
                    "error": repr(exc),
                }
            )

    out_path = Path(args.out_manifest)
    write_jsonl(out_rows, out_path)

    failure_path = out_path.with_suffix(".failures.jsonl")
    write_jsonl(failures, failure_path)

    summary = {
        "source_manifest": str(manifest_path),
        "out_manifest": str(out_path),
        "expected_n": len(rows),
        "written_n": len(out_rows),
        "failures_n": len(failures),
        "failure_path": str(failure_path),
        "feature_keys": list(FEATURE_KEYS),
        "note": "Graph features are recomputed from distorted images, not copied from clean images.",
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
