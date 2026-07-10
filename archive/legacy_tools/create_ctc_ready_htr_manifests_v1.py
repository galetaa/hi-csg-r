from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


SOURCE_ROOT = Path("data/experiments/htr_baseline_v1")
OUT_ROOT = Path("data/experiments/htr_baseline_v1_ctc_ready")

DATASETS = [
    "iam",
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks_clean",
]


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ctc_min_timesteps(text: str) -> int:
    repeats = sum(1 for a, b in zip(text, text[1:]) if a == b)
    return len(text) + repeats


def image_width(path: str | Path) -> int | None:
    try:
        with Image.open(path) as img:
            return int(img.size[0])
    except Exception:
        return None


def is_ctc_ready(row: dict[str, Any], time_downsample: int) -> tuple[bool, str | None]:
    text = str(row.get("text") or "")
    image_path = row.get("image_path")

    if not text:
        return False, "empty_text"

    if not image_path or not Path(image_path).exists():
        return False, "missing_image"

    w = image_width(image_path)
    if w is None:
        return False, "unreadable_image"

    estimated_t = max(1, w // time_downsample)
    ctc_min = ctc_min_timesteps(text)

    if ctc_min > estimated_t:
        return False, "ctc_min_gt_timesteps"

    return True, None


def copy_vocab(src_dataset_dir: Path, dst_dataset_dir: Path) -> None:
    src_vocab = src_dataset_dir / "vocab.json"
    dst_vocab = dst_dataset_dir / "vocab.json"
    dst_vocab.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_vocab, dst_vocab)


def process_dataset(dataset: str, source_root: Path, out_root: Path, time_downsample: int) -> dict[str, Any]:
    src_dir = source_root / dataset
    dst_dir = out_root / dataset

    all_rows = read_jsonl(src_dir / "all.jsonl")

    kept = []
    rejected = []
    rejection_reasons = Counter()

    for row in all_rows:
        ok, reason = is_ctc_ready(row, time_downsample=time_downsample)
        if ok:
            kept.append(row)
        else:
            rejection_reasons[reason] += 1
            rejected.append({
                "sample_id": row.get("sample_id"),
                "split": row.get("split"),
                "image_path": row.get("image_path"),
                "text": row.get("text"),
                "reason": reason,
            })

    for split in ["train", "val", "test"]:
        split_rows = [r for r in kept if r["split"] == split]
        write_jsonl(split_rows, dst_dir / f"{split}.jsonl")

    write_jsonl(kept, dst_dir / "all.jsonl")
    write_jsonl(rejected, dst_dir / "rejected.jsonl")
    copy_vocab(src_dir, dst_dir)

    source_summary = read_json(src_dir / "summary.json")

    summary = {
        "dataset": dataset,
        "source_dir": str(src_dir),
        "num_source_records": len(all_rows),
        "num_kept": len(kept),
        "num_rejected": len(rejected),
        "rejection_reasons": dict(rejection_reasons),
        "splits": dict(Counter(r["split"] for r in kept)),
        "levels": dict(Counter(r.get("level") for r in kept)),
        "categories": dict(Counter(r.get("category") for r in kept)),
        "time_downsample": time_downsample,
        "vocab_path": str(dst_dir / "vocab.json"),
        "source_summary": {
            "num_records": source_summary.get("num_records"),
            "splits": source_summary.get("splits"),
            "num_chars": source_summary.get("num_chars"),
        },
    }

    write_json(summary, dst_dir / "summary.json")

    print("\n", dataset)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default=str(SOURCE_ROOT))
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    parser.add_argument("--time_downsample", type=int, default=4)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)

    summaries = {}
    for dataset in DATASETS:
        summaries[dataset] = process_dataset(
            dataset=dataset,
            source_root=source_root,
            out_root=out_root,
            time_downsample=args.time_downsample,
        )

    global_summary = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "time_downsample": args.time_downsample,
        "datasets": summaries,
    }

    write_json(global_summary, out_root / "summary.json")
    print("\nwrote:", out_root / "summary.json")


if __name__ == "__main__":
    main()