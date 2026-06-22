from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def vocab_characters(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    if "idx_to_token" in obj:
        chars = set(obj["idx_to_token"])
    elif "itos" in obj:
        chars = set(obj["itos"])
    elif "idx_to_char" in obj:
        chars = set(obj["idx_to_char"])
    elif "characters" in obj:
        chars = set(obj["characters"])
    elif "chars" in obj:
        chars = set(obj["chars"])
    else:
        raise ValueError(f"Unknown vocab format: {sorted(obj.keys())}")

    chars.discard(str(obj.get("blank_token", "<blank>")))
    return chars


def normalize_line_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)

    text = str(
        row.get("text")
        or row.get("normalized_transcription")
        or row.get("raw_transcription")
        or ""
    )

    row["text"] = text
    row["normalized_transcription"] = text
    row["raw_transcription"] = text
    row["split"] = "train"

    row["dataset"] = "school_notebooks_line"
    row["source_dataset"] = "school_notebooks_clean"
    row["augmentation_source"] = "school_full_line_raw_context_v1_sampled_5k"
    row["source_type"] = "natural_line_group"
    row["level"] = "line"

    # Keep line samples image-only unless graph features are explicitly built later.
    row.pop("graph_features", None)
    row.pop("graph_feature_names", None)

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", required=True)
    parser.add_argument("--line_train_jsonl", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--copy_val_test", action="store_true")
    args = parser.parse_args()

    base_root = Path(args.base_root)
    line_train_path = Path(args.line_train_jsonl)
    out_root = Path(args.out_root)

    base_train = read_jsonl(base_root / "train.jsonl")
    line_rows_raw = read_jsonl(line_train_path)

    vocab_path = base_root / "vocab.json"
    allowed_chars = vocab_characters(vocab_path) if vocab_path.exists() else None
    filtered_oov = []
    line_train = []

    for row in line_rows_raw:
        normalized = normalize_line_row(row)

        if allowed_chars is not None:
            missing = sorted(set(normalized["text"]) - allowed_chars)

            if missing:
                filtered_oov.append({
                    "sample_id": normalized.get("sample_id"),
                    "text": normalized["text"],
                    "missing_characters": missing,
                })
                continue

        line_train.append(normalized)

    merged_train = base_train + line_train

    write_jsonl(
        merged_train,
        out_root / "train.jsonl",
    )

    if args.copy_val_test:
        for split in ["val", "test"]:
            src = base_root / f"{split}.jsonl"
            if src.exists():
                write_jsonl(
                    read_jsonl(src),
                    out_root / f"{split}.jsonl",
                )

    for extra_name in ["vocab.json", "charset.json"]:
        src = base_root / extra_name
        if src.exists():
            (out_root / extra_name).write_text(
                src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    summary = {
        "base_root": str(base_root),
        "line_train_jsonl": str(line_train_path),
        "out_root": str(out_root),
        "base_train_n": len(base_train),
        "line_train_input_n": len(line_rows_raw),
        "line_train_n": len(line_train),
        "line_train_oov_filtered_n": len(filtered_oov),
        "line_train_oov_filtered_examples": filtered_oov[:20],
        "merged_train_n": len(merged_train),
        "dataset_counts": dict(Counter(
            str(row.get("dataset", ""))
            for row in merged_train
        )),
        "augmentation_source": "school_full_line_raw_context_v1_sampled_5k",
        "line_crop_quality_note": (
            "Rendered line images are raw natural-line crops with contextual overlap; "
            "sanity gate: readable=100%, good_for_htr=100%, strict correct_crop=63.8%."
        ),
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
