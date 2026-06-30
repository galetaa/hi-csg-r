from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HKR_ALL = Path("data/experiments/htr_baseline_v1_ctc_ready/hkr_words/all.jsonl")
SCHOOL_ALL = Path("data/experiments/htr_baseline_v1_ctc_ready/school_notebooks_clean/all.jsonl")
BASE_VOCAB = Path("data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/vocab.json")
LINE_FULL_TRAIN = Path("data/experiments/iter2_line_corpus/school_notebooks_full_line_v1/train.jsonl")
LINE_RENDERED_10K = Path("data/experiments/iter2_line_corpus/school_notebooks_full_line_v1_sampled_10k_rendered/train.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_dataset(row: dict[str, Any]) -> str:
    return str(row.get("source_dataset") or row.get("dataset") or "<unknown>")


def page_key(row: dict[str, Any]) -> str | None:
    meta = row.get("source_metadata") or {}
    page = meta.get("source_image_file") or meta.get("page_id") or row.get("source_image_file") or row.get("page_id")
    if page in (None, ""):
        return None
    return f"{source_dataset(row)}|{page}"


def allowed_chars(vocab_path: Path) -> set[str]:
    obj = json.loads(vocab_path.read_text(encoding="utf-8"))
    if "char_to_idx" in obj:
        return set(obj["char_to_idx"]) - {obj.get("blank_token", "<blank>")}
    if "idx_to_char" in obj:
        values = obj["idx_to_char"]
        return set(values if isinstance(values, list) else values.values()) - {obj.get("blank_token", "<blank>")}
    if "characters" in obj:
        return set(obj["characters"])
    raise ValueError(f"Unsupported vocab format: {vocab_path}")


def normalize_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    out = dict(row)
    text = str(out.get("text") or out.get("normalized_transcription") or out.get("raw_transcription") or "")
    out["text"] = text
    out["normalized_transcription"] = text
    out["raw_transcription"] = text
    out["split"] = split
    return out


def rows_with_allowed_chars(rows: list[dict[str, Any]], chars: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    rejected = []
    for row in rows:
        text = str(row.get("text") or "")
        missing = sorted(set(text) - chars)
        if missing:
            rejected.append({
                "sample_id": row.get("sample_id"),
                "dataset": row.get("dataset"),
                "text": text,
                "missing_characters": missing,
            })
        else:
            kept.append(row)
    return kept, rejected


def group_by_page(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = page_key(row)
        if key is not None:
            grouped[key].append(row)
    return dict(grouped)


def split_hkr_groups(groups: dict[str, list[dict[str, Any]]], *, seed: int) -> dict[str, list[str]]:
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    if len(keys) < 6:
        raise ValueError(f"Need at least 6 HKR page groups, got {len(keys)}")
    return {
        "test": sorted(keys[:2]),
        "val": sorted(keys[2:4]),
        "train": sorted(keys[4:]),
    }


def split_many_groups(groups: dict[str, list[dict[str, Any]]], *, seed: int, val_frac: float, test_frac: float) -> dict[str, list[str]]:
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac)))
    n_val = max(1, int(round(len(keys) * val_frac)))
    return {
        "test": sorted(keys[:n_test]),
        "val": sorted(keys[n_test:n_test + n_val]),
        "train": sorted(keys[n_test + n_val:]),
    }


def sample_from_groups(
    groups: dict[str, list[dict[str, Any]]],
    keys: list[str],
    *,
    n: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    pool = [row for key in keys for row in groups[key]]
    if len(pool) < n:
        raise ValueError(f"Need {n} rows for {split}, only {len(pool)} available")
    rng = random.Random(seed)
    sampled = rng.sample(pool, n)
    sampled.sort(key=lambda row: str(row.get("sample_id")))
    return [normalize_row(row, split=split) for row in sampled]


def assert_disjoint(split_groups: dict[str, list[str]], *, label: str) -> None:
    seen: dict[str, str] = {}
    for split, keys in split_groups.items():
        for key in keys:
            if key in seen:
                raise AssertionError(f"{label} group {key} in both {seen[key]} and {split}")
            seen[key] = split


def copy_vocab(out_root: Path) -> None:
    obj = json.loads(BASE_VOCAB.read_text(encoding="utf-8"))
    blank = obj.get("blank_token", "<blank>")
    idx_to_char = obj.get("idx_to_char")
    if isinstance(idx_to_char, list):
        obj["characters"] = [str(ch) for ch in idx_to_char if ch != blank]
    elif isinstance(idx_to_char, dict):
        obj["characters"] = [
            str(ch)
            for _, ch in sorted(((int(idx), ch) for idx, ch in idx_to_char.items()))
            if ch != blank
        ]
    else:
        obj["characters"] = sorted(allowed_chars(BASE_VOCAB))
    (out_root / "vocab.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_base(args: argparse.Namespace) -> dict[str, Any]:
    chars = allowed_chars(BASE_VOCAB)
    hkr_rows, hkr_rejected = rows_with_allowed_chars(read_jsonl(HKR_ALL), chars)
    school_rows, school_rejected = rows_with_allowed_chars(read_jsonl(SCHOOL_ALL), chars)

    hkr_groups = group_by_page(hkr_rows)
    school_groups = group_by_page(school_rows)
    hkr_split = split_hkr_groups(hkr_groups, seed=args.seed + 11)
    school_split = split_many_groups(
        school_groups,
        seed=args.seed + 29,
        val_frac=args.val_page_frac,
        test_frac=args.test_page_frac,
    )
    assert_disjoint(hkr_split, label="hkr")
    assert_disjoint(school_split, label="school")

    out_root = Path(args.out_root)
    base_root = out_root / "page_disjoint_hkr_school_base_v1"
    line_root = out_root / "page_disjoint_hkr_school_plus_lines_10k_v1"
    base_root.mkdir(parents=True, exist_ok=True)
    line_root.mkdir(parents=True, exist_ok=True)

    split_rows = {}
    for split in ["train", "val", "test"]:
        hkr_n = args.train_per_dataset if split == "train" else args.eval_per_dataset
        school_n = args.train_per_dataset if split == "train" else args.eval_per_dataset
        rows = []
        rows.extend(
            sample_from_groups(
                hkr_groups,
                hkr_split[split],
                n=hkr_n,
                seed=args.seed + {"train": 101, "val": 102, "test": 103}[split],
                split=split,
            )
        )
        rows.extend(
            sample_from_groups(
                school_groups,
                school_split[split],
                n=school_n,
                seed=args.seed + {"train": 201, "val": 202, "test": 203}[split],
                split=split,
            )
        )
        random.Random(args.seed + {"train": 301, "val": 302, "test": 303}[split]).shuffle(rows)
        split_rows[split] = rows
        write_jsonl(rows, base_root / f"{split}.jsonl")
        write_jsonl(rows, line_root / f"{split}.jsonl")

    copy_vocab(base_root)
    copy_vocab(line_root)

    train_school_pages = set(school_split["train"])
    selected_lines, line_summary = select_line_rows(
        chars=chars,
        train_school_pages=train_school_pages,
        target=args.line_target,
        seed=args.seed + 401,
    )
    line_train = split_rows["train"] + selected_lines
    random.Random(args.seed + 402).shuffle(line_train)
    write_jsonl(line_train, line_root / "train.jsonl")

    base_summary = summarize_variant(
        root=base_root,
        rows_by_split=split_rows,
        hkr_split=hkr_split,
        school_split=school_split,
        hkr_rejected=hkr_rejected,
        school_rejected=school_rejected,
    )
    line_summary_full = {
        **base_summary,
        "root": str(line_root),
        "vocab": str(line_root / "vocab.json"),
        "base_root": str(base_root),
        "line_train": line_summary,
        "train": summarize_rows(line_train),
        "purpose": "page-disjoint hkr+school base plus natural-line augmentation restricted to training pages",
    }
    base_summary["purpose"] = "page-disjoint hkr+school base without cyrillic_handwriting, because cyrillic page/writer metadata is absent"

    (base_root / "summary.json").write_text(json.dumps(base_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (line_root / "summary.json").write_text(json.dumps(line_summary_full, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "base_root": str(base_root),
        "line_root": str(line_root),
        "base_summary": base_summary,
        "line_summary": line_summary_full,
    }


def line_source_key(row: dict[str, Any]) -> str | None:
    sf = row.get("source_image_file")
    if sf:
        return f"school_notebooks|{sf}"
    return None


def select_line_rows(
    *,
    chars: set[str],
    train_school_pages: set[str],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_meta = {str(row["sample_id"]): row for row in read_jsonl(LINE_FULL_TRAIN)}
    rendered = read_jsonl(LINE_RENDERED_10K)
    candidates = []
    rejected = []
    missing_meta = []

    for row in rendered:
        sample_id = str(row.get("sample_id"))
        meta = full_meta.get(sample_id)
        if meta is None:
            missing_meta.append(sample_id)
            continue
        source_key = line_source_key(meta)
        if source_key not in train_school_pages:
            continue
        normalized = normalize_row(row, split="train")
        normalized["source_metadata"] = {
            "page_id": meta.get("page_id"),
            "line_id": meta.get("line_id"),
            "source_image_file": meta.get("source_image_file"),
            "source_line_group_id": meta.get("line_group_id"),
            "transcription_scope": "natural_line_group",
        }
        normalized["augmentation_source"] = "school_full_line_raw_context_page_disjoint_v1"
        normalized["source_type"] = "natural_line_context"
        normalized["source_dataset"] = "school_notebooks"
        text = str(normalized.get("text") or "")
        missing = sorted(set(text) - chars)
        if missing:
            rejected.append({
                "sample_id": sample_id,
                "text": text,
                "missing_characters": missing,
            })
            continue
        candidates.append(normalized)

    if len(candidates) < target:
        selected = candidates
    else:
        selected = random.Random(seed).sample(candidates, target)
    selected.sort(key=lambda row: str(row.get("sample_id")))
    selected_pages = {page_key(row) for row in selected if page_key(row)}
    return selected, {
        "source_full_line_manifest": str(LINE_FULL_TRAIN),
        "source_rendered_manifest": str(LINE_RENDERED_10K),
        "target": target,
        "candidate_n": len(candidates),
        "selected_n": len(selected),
        "selected_unique_page_keys": len(selected_pages),
        "selected_pages_all_in_base_train": sorted(selected_pages - train_school_pages) == [],
        "missing_meta_n": len(missing_meta),
        "oov_rejected_n": len(rejected),
        "oov_rejected_examples": rejected[:20],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "by_dataset": dict(Counter(str(row.get("dataset")) for row in rows)),
        "by_source_dataset": dict(Counter(source_dataset(row) for row in rows)),
        "by_level": dict(Counter(str(row.get("level")) for row in rows)),
        "unique_page_keys": len({page_key(row) for row in rows if page_key(row)}),
        "page_key_rows": sum(1 for row in rows if page_key(row)),
    }


def summarize_variant(
    *,
    root: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    hkr_split: dict[str, list[str]],
    school_split: dict[str, list[str]],
    hkr_rejected: list[dict[str, Any]],
    school_rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    page_overlap = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        a_pages = {page_key(row) for row in rows_by_split[a] if page_key(row)}
        b_pages = {page_key(row) for row in rows_by_split[b] if page_key(row)}
        page_overlap[f"{a}_vs_{b}"] = sorted(a_pages & b_pages)
    return {
        "root": str(root),
        "vocab": str(root / "vocab.json"),
        "source_manifests": {
            "hkr_words": str(HKR_ALL),
            "school_notebooks_clean": str(SCHOOL_ALL),
            "base_vocab": str(BASE_VOCAB),
        },
        "splits": {split: summarize_rows(rows) for split, rows in rows_by_split.items()},
        "hkr_page_groups": {split: keys for split, keys in hkr_split.items()},
        "school_page_group_counts": {split: len(keys) for split, keys in school_split.items()},
        "school_page_group_examples": {split: keys[:10] for split, keys in school_split.items()},
        "page_overlap": page_overlap,
        "oov_rejected": {
            "hkr_words": {"n": len(hkr_rejected), "examples": hkr_rejected[:20]},
            "school_notebooks_clean": {"n": len(school_rejected), "examples": school_rejected[:20]},
        },
        "metadata_limitation": "cyrillic_handwriting is excluded because it has no page_id/source_image_file/writer_id metadata.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="data/experiments/htr_publication_v3")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--train_per_dataset", type=int, default=10000)
    parser.add_argument("--eval_per_dataset", type=int, default=2000)
    parser.add_argument("--line_target", type=int, default=10000)
    parser.add_argument("--val_page_frac", type=float, default=0.10)
    parser.add_argument("--test_page_frac", type=float, default=0.10)
    args = parser.parse_args()

    result = build_base(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
