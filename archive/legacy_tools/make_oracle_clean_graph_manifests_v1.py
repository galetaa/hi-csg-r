from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_graph_manifest", required=True)
    parser.add_argument("--distorted_manifest", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--graph_key", default="graph_features")
    args = parser.parse_args()

    clean_rows = read_jsonl(Path(args.clean_graph_manifest))
    distorted_rows = read_jsonl(Path(args.distorted_manifest))

    clean_by_id: dict[str, dict[str, Any]] = {}

    for row in clean_rows:
        sid = str(row.get("sample_id"))
        clean_by_id[sid] = row

    out_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for row in distorted_rows:
        clean_id = row.get("clean_sample_id")

        if clean_id is None:
            sid = str(row.get("sample_id"))
            clean_id = sid.split("__", 1)[0]

        clean_id = str(clean_id)
        clean_row = clean_by_id.get(clean_id)

        if clean_row is None:
            missing.append(clean_id)
            continue

        if args.graph_key not in clean_row:
            raise KeyError(
                f"Clean row {clean_id!r} has no {args.graph_key!r}. "
                f"Available keys: {sorted(clean_row.keys())}"
            )

        new_row = dict(row)

        # Keep distorted image, but copy clean graph features.
        new_row[args.graph_key] = clean_row[args.graph_key]

        if "graph_feature_names" in clean_row:
            new_row["graph_feature_names"] = clean_row["graph_feature_names"]

        new_row["graph_feature_source"] = "oracle_clean_graph"
        new_row["oracle_clean_graph"] = True
        new_row["oracle_warning"] = (
            "This manifest uses clean graph features with distorted images. "
            "Use only as diagnostic upper bound, not as final fair comparison."
        )

        out_rows.append(new_row)

    write_jsonl(out_rows, Path(args.out_manifest))

    summary = {
        "clean_graph_manifest": args.clean_graph_manifest,
        "distorted_manifest": args.distorted_manifest,
        "out_manifest": args.out_manifest,
        "distorted_n": len(distorted_rows),
        "written_n": len(out_rows),
        "missing_n": len(missing),
        "missing_examples": missing[:20],
        "graph_feature_source": "oracle_clean_graph",
    }

    summary_path = Path(args.out_manifest).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
