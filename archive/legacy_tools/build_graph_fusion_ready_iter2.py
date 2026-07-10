from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DROP_FEATURES = {"text_len"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def feature_map(row: dict[str, Any]) -> dict[str, float]:
    names = row.get("graph_feature_names") or []
    values = row.get("graph_features") or []
    return {
        str(name): float(value)
        for name, value in zip(names, values)
    }


def infer_feature_names(graph_rows: list[dict[str, Any]]) -> list[str]:
    for row in graph_rows:
        names = [str(name) for name in row.get("graph_feature_names") or []]
        if names:
            return [name for name in names if name not in DROP_FEATURES]
    raise RuntimeError("No graph_feature_names found")


def fit_scaler(
    rows: list[dict[str, Any]],
    graph_by_id: dict[str, dict[str, Any]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        source = graph_by_id.get(sample_id)
        if source is None:
            continue
        fmap = feature_map(source)
        vectors.append([float(fmap[name]) for name in feature_names])

    if not vectors:
        raise RuntimeError("No valid graph rows for scaler fit")

    arr = np.asarray(vectors, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalized_vector(
    source: dict[str, Any],
    feature_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> list[float]:
    fmap = feature_map(source)
    raw = np.asarray([float(fmap[name]) for name in feature_names], dtype=np.float32)
    z = (raw - mean) / std
    z = np.clip(z, -5.0, 5.0)
    return [float(value) for value in z]


def enrich_split(
    rows: list[dict[str, Any]],
    graph_by_id: dict[str, dict[str, Any]],
    feature_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    missing_word = []
    line_n = 0
    word_n = 0
    graph_valid_n = 0
    line_graph_valid_n = 0
    dataset_counts = Counter()

    zero = [0.0 for _ in feature_names]

    for row in rows:
        row = dict(row)
        sample_id = str(row.get("sample_id", ""))
        dataset = str(row.get("dataset", ""))
        is_line = dataset == "school_notebooks_line" or row.get("level") == "line"
        dataset_counts[dataset] += 1

        if is_line:
            line_n += 1
            row["graph_features"] = list(zero)
            row["graph_feature_names"] = list(feature_names)
            row["graph_valid"] = False
            row["graph_source"] = "none_context_line"
        else:
            word_n += 1
            source = graph_by_id.get(sample_id)
            if source is None:
                missing_word.append(sample_id)
                row["graph_features"] = list(zero)
                row["graph_feature_names"] = list(feature_names)
                row["graph_valid"] = False
                row["graph_source"] = "missing_word_graph"
            else:
                row["graph_features"] = normalized_vector(
                    source,
                    feature_names,
                    mean,
                    std,
                )
                row["graph_feature_names"] = list(feature_names)
                row["graph_valid"] = True
                row["graph_source"] = "tri10k_mixed_school_lineaware_v3"
                graph_valid_n += 1

        if is_line and row.get("graph_valid") is True:
            line_graph_valid_n += 1

        out.append(row)

    summary = {
        "n": len(out),
        "word_n": word_n,
        "line_n": line_n,
        "graph_valid_n": graph_valid_n,
        "graph_missing_word_n": len(missing_word),
        "graph_missing_word_examples": missing_word[:20],
        "line_graph_valid_n": line_graph_valid_n,
        "dataset_counts": dict(dataset_counts),
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", required=True)
    parser.add_argument("--graph_root", required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    graph_root = Path(args.graph_root)
    out_root = Path(args.out_root)

    graph_rows_all = []
    graph_by_id = {}
    for split in ["train", "val", "test"]:
        rows = read_jsonl(graph_root / f"{split}.jsonl")
        graph_rows_all.extend(rows)
        for row in rows:
            graph_by_id[str(row["sample_id"])] = row

    feature_names = infer_feature_names(graph_rows_all)
    train_rows = read_jsonl(base_root / "train.jsonl")
    mean, std = fit_scaler(train_rows, graph_by_id, feature_names)

    summary: dict[str, Any] = {
        "base_root": str(base_root),
        "graph_root": str(graph_root),
        "out_root": str(out_root),
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "dropped_features": sorted(DROP_FEATURES),
        "scaler_fit": "train graph_valid rows only",
        "clip": [-5.0, 5.0],
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        base_rows = read_jsonl(base_root / f"{split}.jsonl")
        enriched, split_summary = enrich_split(
            base_rows,
            graph_by_id,
            feature_names,
            mean,
            std,
        )
        write_jsonl(enriched, out_root / f"{split}.jsonl")
        summary["splits"][split] = split_summary

    for extra_name in ["vocab.json", "charset.json"]:
        src = base_root / extra_name
        if src.exists():
            (out_root / extra_name).write_text(
                src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    scaler = {
        "feature_names": feature_names,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "clip": [-5.0, 5.0],
        "fit_split": "train",
        "fit_graph_valid_only": True,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "graph_scaler.json").write_text(
        json.dumps(scaler, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
