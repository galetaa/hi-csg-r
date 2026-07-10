from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


CORE_DATASETS = [
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks_clean",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def normalize_extra_row(row: dict[str, Any], *, control_name: str) -> dict[str, Any]:
    out = dict(row)
    text = str(
        out.get("text")
        or out.get("normalized_transcription")
        or out.get("raw_transcription")
        or ""
    )

    out["text"] = text
    out["normalized_transcription"] = text
    out["raw_transcription"] = text
    out["split"] = "train"
    out["augmentation_source"] = control_name
    out["source_type"] = "same_size_crop_control"

    # Keep these controls image-only; graph features are irrelevant for CRNN-CTC.
    out.pop("graph_features", None)
    out.pop("graph_feature_names", None)

    return out


def load_pool_rows(source_roots: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    for dataset, root in source_roots.items():
        pools[dataset] = read_jsonl(root / "train.jsonl")
    return pools


def filter_candidates(
    rows: list[dict[str, Any]],
    *,
    used_ids: set[str],
    allowed_chars: set[str],
    level: str | None,
    control_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for row in rows:
        if str(row.get("sample_id")) in used_ids:
            continue

        if level is not None and str(row.get("level")) != level:
            continue

        normalized = normalize_extra_row(row, control_name=control_name)
        text = normalized["text"]
        missing = sorted(set(text) - allowed_chars)

        if missing:
            rejected.append({
                "sample_id": normalized.get("sample_id"),
                "dataset": normalized.get("dataset"),
                "text": text,
                "missing_characters": missing,
            })
            continue

        kept.append(normalized)

    return kept, rejected


def sample_rows(rows: list[dict[str, Any]], *, n: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < n:
        raise ValueError(f"Need {n} rows, only {len(rows)} candidates available")

    rng = random.Random(seed)
    return rng.sample(rows, n)


def copy_base_files(base_root: Path, out_root: Path) -> None:
    for split in ["val", "test"]:
        write_jsonl(read_jsonl(base_root / f"{split}.jsonl"), out_root / f"{split}.jsonl")

    for name in ["vocab.json", "charset.json"]:
        src = base_root / name
        if src.exists():
            (out_root / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def build_random_mixed(
    *,
    base_train: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    used_ids: set[str],
    allowed_chars: set[str],
    target_total: int,
    seed: int,
    control_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = target_total // len(CORE_DATASETS)
    targets = {dataset: base for dataset in CORE_DATASETS}
    targets["school_notebooks_clean"] += target_total - sum(targets.values())

    selected: list[dict[str, Any]] = []
    rejected_all: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}

    for dataset in CORE_DATASETS:
        candidates, rejected = filter_candidates(
            pools[dataset],
            used_ids=used_ids,
            allowed_chars=allowed_chars,
            level=None,
            control_name=control_name,
        )
        candidate_counts[dataset] = len(candidates)
        rejected_all.extend(rejected)
        selected.extend(
            sample_rows(
                candidates,
                n=targets[dataset],
                seed=seed + sum(ord(ch) for ch in dataset),
            )
        )

    rng = random.Random(seed)
    rng.shuffle(selected)

    return base_train + selected, {
        "target_by_dataset": targets,
        "candidate_counts": candidate_counts,
        "selected_n": len(selected),
        "selected_by_dataset": dict(Counter(str(row.get("dataset")) for row in selected)),
        "oov_rejected_n": len(rejected_all),
        "oov_rejected_examples": rejected_all[:20],
    }


def build_school_words(
    *,
    base_train: list[dict[str, Any]],
    school_rows: list[dict[str, Any]],
    used_ids: set[str],
    allowed_chars: set[str],
    target_total: int,
    seed: int,
    control_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, rejected = filter_candidates(
        school_rows,
        used_ids=used_ids,
        allowed_chars=allowed_chars,
        level="word",
        control_name=control_name,
    )

    selected = sample_rows(candidates, n=target_total, seed=seed)

    return base_train + selected, {
        "target_by_dataset": {"school_notebooks_clean": target_total},
        "candidate_counts": {"school_notebooks_clean_word": len(candidates)},
        "selected_n": len(selected),
        "selected_by_dataset": dict(Counter(str(row.get("dataset")) for row in selected)),
        "selected_by_category": dict(Counter(str(row.get("category")) for row in selected)),
        "oov_rejected_n": len(rejected),
        "oov_rejected_examples": rejected[:20],
    }


def parse_source_roots(raw: str) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        dataset, path = item.split("=", 1)
        roots[dataset.strip()] = Path(path.strip())
    missing = sorted(set(CORE_DATASETS) - set(roots))
    if missing:
        raise ValueError(f"Missing source roots for: {missing}")
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", required=True)
    parser.add_argument("--source_roots", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument(
        "--control",
        choices=["random_mixed_10k", "school_words_10k"],
        required=True,
    )
    parser.add_argument("--target_total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    out_root = Path(args.out_root)
    control_name = args.control

    base_train = read_jsonl(base_root / "train.jsonl")
    used_ids = {str(row.get("sample_id")) for row in base_train}
    allowed_chars = vocab_characters(base_root / "vocab.json")
    pools = load_pool_rows(parse_source_roots(args.source_roots))

    if control_name == "random_mixed_10k":
        merged_train, control_summary = build_random_mixed(
            base_train=base_train,
            pools=pools,
            used_ids=used_ids,
            allowed_chars=allowed_chars,
            target_total=args.target_total,
            seed=args.seed,
            control_name=control_name,
        )
    else:
        merged_train, control_summary = build_school_words(
            base_train=base_train,
            school_rows=pools["school_notebooks_clean"],
            used_ids=used_ids,
            allowed_chars=allowed_chars,
            target_total=args.target_total,
            seed=args.seed,
            control_name=control_name,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(merged_train, out_root / "train.jsonl")
    copy_base_files(base_root, out_root)

    summary = {
        "control": control_name,
        "base_root": str(base_root),
        "source_roots": {
            key: str(value)
            for key, value in parse_source_roots(args.source_roots).items()
        },
        "out_root": str(out_root),
        "seed": args.seed,
        "target_total": args.target_total,
        "base_train_n": len(base_train),
        "merged_train_n": len(merged_train),
        "merged_by_dataset": dict(Counter(str(row.get("dataset")) for row in merged_train)),
        **control_summary,
        "interpretation": (
            "Same-size image-only augmentation control. It tests whether the natural-line "
            "context gain can be explained by adding the same number of ordinary crop "
            "training samples."
        ),
    }

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
