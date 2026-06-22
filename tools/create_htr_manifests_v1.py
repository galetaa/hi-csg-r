from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.datasets.metadata import read_jsonl


OUT_ROOT = Path("data/experiments/htr_baseline_v1")

HARD_BAD_FLAGS = {
    "missing_image",
    "broken_image",
    "empty_raw_transcription",
    "empty_normalized_transcription",
    "ocr_preprocess_failed",
    "feature_preprocess_failed",
}

SCHOOL_CLEAN_EXCLUDE_FLAGS = {
    "single_character_or_mark",
    "occluded",
}

DATASETS = {
    "iam": {
        "metadata": Path("data/processed/iam/metadata.preprocessed.jsonl"),
        "allowed_levels": {"line"},
        "image_field": "ocr_image_path",
        "text_mode": "ctc_default",
        "exclude_flags": HARD_BAD_FLAGS,
    },
    "cyrillic_handwriting": {
        "metadata": Path("data/processed/cyrillic_handwriting/metadata.preprocessed.jsonl"),
        "allowed_levels": {"word", "phrase"},
        "image_field": "ocr_image_path",
        "text_mode": "ctc_default",
        "exclude_flags": HARD_BAD_FLAGS,
    },
    "hkr_words": {
        "metadata": Path("data/processed/hkr_words/metadata.preprocessed.jsonl"),
        "allowed_levels": {"word", "phrase"},
        "image_field": "ocr_image_path",
        "text_mode": "ctc_default",
        "exclude_flags": HARD_BAD_FLAGS,
    },
    "school_notebooks_clean": {
        "metadata": Path("data/processed/school_notebooks/metadata.preprocessed.jsonl"),
        "source_dataset": "school_notebooks",
        "allowed_levels": {"word", "phrase"},
        "image_field": "ocr_image_path",
        "text_mode": "ctc_default",
        "exclude_flags": HARD_BAD_FLAGS | SCHOOL_CLEAN_EXCLUDE_FLAGS,
        "allowed_categories": {"pupil_text", "pupil_comment", "teacher_comment"},
    },
}


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_text(record: dict[str, Any], mode: str) -> str:
    modes = record.get("transcription_modes") or {}
    text = modes.get(mode) or record.get("normalized_transcription") or ""
    return str(text).strip()


def has_existing_path(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def is_eligible(record: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []

    split = record.get("split")
    if split not in {"train", "val", "test"}:
        reasons.append("bad_split")

    if record.get("metadata", {}).get("usable_for_htr") is not True:
        reasons.append("not_usable_for_htr")

    if record.get("level") not in cfg["allowed_levels"]:
        reasons.append("bad_level")

    flags = set(record.get("metadata", {}).get("quality_flags", []))
    excluded_flags = set(cfg.get("exclude_flags", set()))

    if flags & excluded_flags:
        reasons.extend(sorted(flags & excluded_flags))

    allowed_categories = cfg.get("allowed_categories")
    if allowed_categories is not None:
        category = record.get("metadata", {}).get("category")
        if category not in allowed_categories:
            reasons.append("bad_category")

    image_path = record.get(cfg["image_field"])
    if not has_existing_path(image_path):
        reasons.append("missing_ocr_image")

    text = get_text(record, cfg["text_mode"])
    if not text:
        reasons.append("empty_text")

    return len(reasons) == 0, reasons


def make_manifest_record(record: dict[str, Any], cfg: dict[str, Any], manifest_dataset: str) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    image_path = record[cfg["image_field"]]
    text = get_text(record, cfg["text_mode"])

    return {
        "sample_id": record["sample_id"],
        "dataset": manifest_dataset,
        "source_dataset": record["dataset"],
        "split": record["split"],
        "level": record.get("level"),
        "language": record.get("language"),
        "script": record.get("script"),
        "image_path": image_path,
        "text": text,
        "raw_transcription": record.get("raw_transcription"),
        "normalized_transcription": record.get("normalized_transcription"),
        "text_len": len(text),
        "writer_id": record.get("writer_id"),
        "category": metadata.get("category"),
        "source_flags": metadata.get("quality_flags", []),
        "image_info": metadata.get("image_info", {}),
        "source_metadata": {
            "page_id": metadata.get("page_id"),
            "line_id": metadata.get("line_id"),
            "word_id": metadata.get("word_id"),
            "source_image_file": metadata.get("source_image_file"),
            "transcription_scope": metadata.get("transcription_scope"),
        },
    }


def build_vocab(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chars = sorted({ch for r in rows for ch in r["text"]})

    char_to_idx = {"<blank>": 0}
    for i, ch in enumerate(chars, start=1):
        char_to_idx[ch] = i

    idx_to_char = {str(v): k for k, v in char_to_idx.items()}

    return {
        "blank_token": "<blank>",
        "blank_index": 0,
        "num_classes": len(char_to_idx),
        "characters": chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }


def summarize(rows: list[dict[str, Any]], rejected: Counter) -> dict[str, Any]:
    split_counts = Counter(r["split"] for r in rows)
    level_counts = Counter(r["level"] for r in rows)
    category_counts = Counter(r.get("category") for r in rows)
    text_lengths = [r["text_len"] for r in rows]
    chars = Counter(ch for r in rows for ch in r["text"])

    return {
        "num_records": len(rows),
        "splits": dict(split_counts),
        "levels": dict(level_counts),
        "categories": dict(category_counts),
        "rejected_reasons": dict(rejected),
        "text_len": {
            "min": min(text_lengths) if text_lengths else None,
            "max": max(text_lengths) if text_lengths else None,
            "mean": sum(text_lengths) / len(text_lengths) if text_lengths else None,
        },
        "num_unique_texts": len(set(r["text"] for r in rows)),
        "num_chars": len(chars),
        "top_chars": chars.most_common(50),
    }


def sample_smoke(rows: list[dict[str, Any]], max_per_split: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_split = defaultdict(list)

    for r in rows:
        by_split[r["split"]].append(r)

    out = []
    for split in ["train", "val", "test"]:
        split_rows = by_split[split]
        if len(split_rows) <= max_per_split:
            out.extend(split_rows)
        else:
            out.extend(rng.sample(split_rows, max_per_split))

    return out


def process_dataset(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = cfg["metadata"]
    records = read_jsonl(metadata_path)

    rows = []
    rejected = Counter()

    for r in records:
        ok, reasons = is_eligible(r, cfg)
        if not ok:
            for reason in reasons:
                rejected[reason] += 1
            continue

        rows.append(make_manifest_record(r, cfg, manifest_dataset=name))

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        write_jsonl(split_rows, out_dir / f"{split}.jsonl")

    write_jsonl(rows, out_dir / "all.jsonl")

    vocab = build_vocab(rows if args.vocab_scope == "all" else [r for r in rows if r["split"] == "train"])
    write_json(vocab, out_dir / "vocab.json")

    summary = summarize(rows, rejected)
    summary.update({
        "dataset": name,
        "source_metadata": str(metadata_path),
        "image_field": cfg["image_field"],
        "text_mode": cfg["text_mode"],
        "vocab_scope": args.vocab_scope,
        "vocab_path": str(out_dir / "vocab.json"),
    })

    write_json(summary, out_dir / "summary.json")

    if args.smoke_max_per_split > 0:
        smoke = sample_smoke(rows, args.smoke_max_per_split, args.seed)
        smoke_dir = out_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            split_rows = [r for r in smoke if r["split"] == split]
            write_jsonl(split_rows, smoke_dir / f"{split}.jsonl")

        write_jsonl(smoke, smoke_dir / "all.jsonl")
        write_json(vocab, smoke_dir / "vocab.json")
        write_json(summarize(smoke, Counter()), smoke_dir / "summary.json")

    print(f"\n{name}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def main() -> None:
    global OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    parser.add_argument("--vocab_scope", choices=["all", "train"], default="all")
    parser.add_argument("--smoke_max_per_split", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_ROOT = Path(args.out_root)

    missing = [str(cfg["metadata"]) for cfg in DATASETS.values() if not cfg["metadata"].exists()]
    if missing:
        raise FileNotFoundError(f"Missing metadata files: {missing}")

    all_summaries = {}

    for name, cfg in DATASETS.items():
        all_summaries[name] = process_dataset(name, cfg, args)

    write_json(
        {
            "stage": "htr_baseline_v1",
            "datasets": all_summaries,
            "notes": [
                "HTR baseline uses OCR-preprocessed images.",
                "Text target is transcription_modes.ctc_default.",
                "HWR200 and HKR Forms are excluded from HTR baseline.",
                "School Notebooks clean excludes single_character_or_mark and occluded samples.",
            ],
        },
        OUT_ROOT / "summary.json",
    )

    print(f"\nWrote global summary: {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()