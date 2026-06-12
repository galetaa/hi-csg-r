from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_DIRS = {
    "cyrillic_handwriting": "cyrillic_handwriting",
    "hkr_words": "hkr_words",
    "school_notebooks_clean": "school_notebooks_clean",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_rows(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def normalize_row(row: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    r = dict(row)
    r["source_dataset"] = dataset_key
    r["dataset"] = dataset_key
    return r


def collect_chars(rows: list[dict[str, Any]]) -> list[str]:
    chars = set()
    for r in rows:
        for ch in str(r["text"]):
            chars.add(ch)
    return sorted(chars)


def write_vocab_like_template(chars: list[str], template_path: Path, out_path: Path) -> None:
    template = json.loads(template_path.read_text(encoding="utf-8"))

    # Most likely project formats. We preserve the shape used by existing vocab.json.
    blank_token = (
        template.get("blank_token")
        or template.get("blank")
        or template.get("blank_symbol")
        or "<blank>"
    )

    tokens = [blank_token] + chars

    if "idx_to_token" in template and "token_to_idx" in template:
        vocab = {
            **template,
            "idx_to_token": tokens,
            "token_to_idx": {t: i for i, t in enumerate(tokens)},
            "blank_index": 0,
            "blank_token": blank_token,
            "num_classes": len(tokens),
        }
    elif "itos" in template and "stoi" in template:
        vocab = {
            **template,
            "itos": tokens,
            "stoi": {t: i for i, t in enumerate(tokens)},
            "blank_index": 0,
            "blank_token": blank_token,
            "num_classes": len(tokens),
        }
    elif "chars" in template:
        # Common compact format: blank is implicit index 0, chars exclude blank.
        vocab = {
            **template,
            "chars": chars,
            "blank_index": 0,
            "blank_token": blank_token,
            "num_classes": len(chars) + 1,
        }
        if "char_to_idx" in template:
            vocab["char_to_idx"] = {ch: i + 1 for i, ch in enumerate(chars)}
    elif "idx_to_char" in template and "char_to_idx" in template:
        vocab = {
            **template,
            "idx_to_char": tokens,
            "char_to_idx": {t: i for i, t in enumerate(tokens)},
            "blank_index": 0,
            "blank_token": blank_token,
            "num_classes": len(tokens),
        }
    else:
        raise ValueError(
            f"Unknown vocab format in {template_path}. "
            f"Keys: {sorted(template.keys())}. Paste this vocab.json shape if this fails."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_lens = [len(r["text"]) for r in rows]
    return {
        "count": len(rows),
        "by_dataset": dict(Counter(r["dataset"] for r in rows)),
        "by_level": dict(Counter(r.get("level") for r in rows)),
        "text_len": {
            "min": min(text_lens) if text_lens else None,
            "max": max(text_lens) if text_lens else None,
            "mean": sum(text_lens) / len(text_lens) if text_lens else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/experiments/htr_baseline_v1_ctc_ready")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_n_per_dataset", type=int, default=50000)
    parser.add_argument("--val_n_per_dataset", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    train_rows = []
    val_rows = []
    eval_manifest_paths: dict[str, dict[str, str]] = {}

    all_for_vocab = []

    for dataset_key, rel in DATASET_DIRS.items():
        ds_dir = root / rel

        split_rows = {
            "train": [normalize_row(r, dataset_key) for r in read_jsonl(ds_dir / "train.jsonl")],
            "val": [normalize_row(r, dataset_key) for r in read_jsonl(ds_dir / "val.jsonl")],
            "test": [normalize_row(r, dataset_key) for r in read_jsonl(ds_dir / "test.jsonl")],
        }

        sampled_train = sample_rows(split_rows["train"], args.train_n_per_dataset, rng)
        sampled_val = sample_rows(split_rows["val"], args.val_n_per_dataset, rng)

        train_rows.extend(sampled_train)
        val_rows.extend(sampled_val)

        # Full per-dataset eval manifests, encoded later with the same mixed vocab.
        eval_manifest_paths[dataset_key] = {}
        for split in ["train", "val", "test"]:
            p = out_dir / "eval_manifests" / f"{dataset_key}_{split}.jsonl"
            write_jsonl(split_rows[split], p)
            eval_manifest_paths[dataset_key][split] = str(p)

        all_for_vocab.extend(split_rows["train"])
        all_for_vocab.extend(split_rows["val"])
        all_for_vocab.extend(split_rows["test"])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    write_jsonl(train_rows, out_dir / "train.jsonl")
    write_jsonl(val_rows, out_dir / "val.jsonl")

    chars = collect_chars(all_for_vocab)
    template_vocab = root / "school_notebooks_clean" / "vocab.json"
    write_vocab_like_template(chars, template_vocab, out_dir / "vocab.json")

    summary = {
        "name": out_dir.name,
        "seed": args.seed,
        "train_n_per_dataset": args.train_n_per_dataset,
        "val_n_per_dataset": args.val_n_per_dataset,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "num_chars_without_blank": len(chars),
        "num_classes_with_blank": len(chars) + 1,
        "eval_manifests": eval_manifest_paths,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()