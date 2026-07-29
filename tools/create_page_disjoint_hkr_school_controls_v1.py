from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


HKR_ALL = Path("data/experiments/htr_baseline_v1_ctc_ready/hkr_words/all.jsonl")
SCHOOL_ALL = Path("data/experiments/htr_baseline_v1_ctc_ready/school_notebooks_clean/all.jsonl")
BASE_ROOT = Path("data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1")
LINE_ROOT = Path("data/experiments/htr_publication_v3/page_disjoint_hkr_school_plus_lines_10k_v1")

CONTROL_ROOTS = {
    "page_random_crops_8k_control": Path(
        "data/experiments/htr_publication_v3/page_disjoint_hkr_school_random_crops_8k_control_v1"
    ),
    "page_school_words_8k_control": Path(
        "data/experiments/htr_publication_v3/page_disjoint_hkr_school_school_words_8k_control_v1"
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def vocab_chars(path: Path) -> set[str]:
    obj = read_json(path)
    blank = str(obj.get("blank_token", "<blank>"))
    if "char_to_idx" in obj:
        chars = set(obj["char_to_idx"])
    elif "idx_to_char" in obj:
        values = obj["idx_to_char"]
        chars = set(values if isinstance(values, list) else values.values())
    elif "characters" in obj:
        chars = set(obj["characters"])
    else:
        raise ValueError(f"Unsupported vocab format: {path}")
    chars.discard(blank)
    return {str(ch) for ch in chars}


def normalize_extra(row: dict[str, Any], *, control_name: str) -> dict[str, Any]:
    out = dict(row)
    text = str(out.get("text") or out.get("normalized_transcription") or out.get("raw_transcription") or "")
    out["text"] = text
    out["normalized_transcription"] = text
    out["raw_transcription"] = text
    out["split"] = "train"
    out["augmentation_source"] = control_name
    out["source_type"] = "page_disjoint_same_size_crop_control"
    out.pop("graph_features", None)
    out.pop("graph_feature_names", None)
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    page_keys = [page_key(row) for row in rows if page_key(row) is not None]
    return {
        "n": len(rows),
        "by_dataset": dict(Counter(str(row.get("dataset")) for row in rows)),
        "by_source_dataset": dict(Counter(source_dataset(row) for row in rows)),
        "by_level": dict(Counter(str(row.get("level")) for row in rows)),
        "by_category": dict(Counter(str(row.get("category")) for row in rows)),
        "unique_page_keys": len(set(page_keys)),
        "page_key_rows": len(page_keys),
    }


def filter_candidates(
    rows: list[dict[str, Any]],
    *,
    train_page_keys: set[str],
    used_ids: set[str],
    allowed_chars: set[str],
    control_name: str,
    level: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    kept = []
    rejected = []
    no_page = 0
    for row in rows:
        key = page_key(row)
        if key is None:
            no_page += 1
            continue
        if key not in train_page_keys:
            continue
        if str(row.get("sample_id")) in used_ids:
            continue
        if level is not None and str(row.get("level")) != level:
            continue
        normalized = normalize_extra(row, control_name=control_name)
        missing = sorted(set(normalized["text"]) - allowed_chars)
        if missing:
            rejected.append({
                "sample_id": normalized.get("sample_id"),
                "dataset": normalized.get("dataset"),
                "text": normalized["text"],
                "missing_characters": missing,
            })
            continue
        kept.append(normalized)
    return kept, rejected, no_page


def sample_rows(rows: list[dict[str, Any]], *, n: int, seed: int, label: str) -> list[dict[str, Any]]:
    if len(rows) < n:
        raise ValueError(f"Need {n} rows for {label}, only {len(rows)} candidates available")
    return random.Random(seed).sample(rows, n)


def split_targets(total: int) -> dict[str, int]:
    hkr = total // 2
    return {
        "hkr_words": hkr,
        "school_notebooks_clean": total - hkr,
    }


def copy_base_eval_files(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for name in ["val.jsonl", "test.jsonl", "vocab.json", "charset.json"]:
        src = BASE_ROOT / name
        if src.exists():
            shutil.copyfile(src, out_root / name)


def page_overlap(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    train = {key for row in train_rows if (key := page_key(row)) is not None}
    val = {key for row in val_rows if (key := page_key(row)) is not None}
    test = {key for row in test_rows if (key := page_key(row)) is not None}
    return {
        "train_vs_val": sorted(train & val),
        "train_vs_test": sorted(train & test),
        "val_vs_test": sorted(val & test),
    }


def build_random_control(*, target_total: int, seed: int) -> dict[str, Any]:
    control_name = "page_random_crops_8k_control"
    out_root = CONTROL_ROOTS[control_name]
    base_train = read_jsonl(BASE_ROOT / "train.jsonl")
    base_val = read_jsonl(BASE_ROOT / "val.jsonl")
    base_test = read_jsonl(BASE_ROOT / "test.jsonl")
    allowed = vocab_chars(BASE_ROOT / "vocab.json")
    used_ids = {str(row.get("sample_id")) for row in base_train}
    train_pages = {key for row in base_train if (key := page_key(row)) is not None}
    targets = split_targets(target_total)

    hkr_candidates, hkr_rejected, hkr_no_page = filter_candidates(
        read_jsonl(HKR_ALL),
        train_page_keys=train_pages,
        used_ids=used_ids,
        allowed_chars=allowed,
        control_name=control_name,
    )
    school_candidates, school_rejected, school_no_page = filter_candidates(
        read_jsonl(SCHOOL_ALL),
        train_page_keys=train_pages,
        used_ids=used_ids,
        allowed_chars=allowed,
        control_name=control_name,
    )

    selected = []
    selected.extend(sample_rows(hkr_candidates, n=targets["hkr_words"], seed=seed + 11, label="hkr random control"))
    selected.extend(
        sample_rows(
            school_candidates,
            n=targets["school_notebooks_clean"],
            seed=seed + 29,
            label="school random control",
        )
    )
    random.Random(seed + 41).shuffle(selected)
    train = base_train + selected
    copy_base_eval_files(out_root)
    write_jsonl(train, out_root / "train.jsonl")

    summary = {
        "control": control_name,
        "root": str(out_root),
        "base_root": str(BASE_ROOT),
        "line_root": str(LINE_ROOT),
        "seed": seed,
        "target_total": target_total,
        "base_train_n": len(base_train),
        "merged_train_n": len(train),
        "extra_selected": summarize_rows(selected),
        "train": summarize_rows(train),
        "val": summarize_rows(base_val),
        "test": summarize_rows(base_test),
        "target_by_dataset": targets,
        "candidate_counts": {
            "hkr_words": len(hkr_candidates),
            "school_notebooks_clean": len(school_candidates),
        },
        "missing_page_rows": {
            "hkr_words": hkr_no_page,
            "school_notebooks_clean": school_no_page,
        },
        "oov_rejected_n": len(hkr_rejected) + len(school_rejected),
        "oov_rejected_examples": (hkr_rejected + school_rejected)[:20],
        "page_overlap": page_overlap(train, base_val, base_test),
        "selected_pages_all_in_base_train": all(page_key(row) in train_pages for row in selected),
        "interpretation": (
            "Page-disjoint same-size ordinary crop control. Extra rows are sampled only from "
            "base train pages and compared against the same validation/test pages as the "
            "page-disjoint base and line variants."
        ),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_school_words_control(*, target_total: int, seed: int) -> dict[str, Any]:
    control_name = "page_school_words_8k_control"
    out_root = CONTROL_ROOTS[control_name]
    base_train = read_jsonl(BASE_ROOT / "train.jsonl")
    base_val = read_jsonl(BASE_ROOT / "val.jsonl")
    base_test = read_jsonl(BASE_ROOT / "test.jsonl")
    allowed = vocab_chars(BASE_ROOT / "vocab.json")
    used_ids = {str(row.get("sample_id")) for row in base_train}
    train_pages = {key for row in base_train if (key := page_key(row)) is not None}

    candidates, rejected, no_page = filter_candidates(
        read_jsonl(SCHOOL_ALL),
        train_page_keys=train_pages,
        used_ids=used_ids,
        allowed_chars=allowed,
        control_name=control_name,
        level="word",
    )
    selected = sample_rows(candidates, n=target_total, seed=seed + 53, label="school words control")
    random.Random(seed + 67).shuffle(selected)
    train = base_train + selected
    copy_base_eval_files(out_root)
    write_jsonl(train, out_root / "train.jsonl")

    summary = {
        "control": control_name,
        "root": str(out_root),
        "base_root": str(BASE_ROOT),
        "line_root": str(LINE_ROOT),
        "seed": seed,
        "target_total": target_total,
        "base_train_n": len(base_train),
        "merged_train_n": len(train),
        "extra_selected": summarize_rows(selected),
        "train": summarize_rows(train),
        "val": summarize_rows(base_val),
        "test": summarize_rows(base_test),
        "target_by_dataset": {"school_notebooks_clean": target_total},
        "candidate_counts": {"school_notebooks_clean_word": len(candidates)},
        "missing_page_rows": {"school_notebooks_clean": no_page},
        "oov_rejected_n": len(rejected),
        "oov_rejected_examples": rejected[:20],
        "page_overlap": page_overlap(train, base_val, base_test),
        "selected_pages_all_in_base_train": all(page_key(row) in train_pages for row in selected),
        "interpretation": (
            "Page-disjoint same-size School-word control. Extra rows are ordinary word crops "
            "from base train pages only, not natural line-context samples."
        ),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", nargs="+", default=list(CONTROL_ROOTS))
    parser.add_argument("--target_total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    line_summary = read_json(LINE_ROOT / "summary.json")
    target_total = args.target_total
    if target_total is None:
        target_total = int(line_summary["line_train"]["selected_n"])

    out = {}
    for control in args.controls:
        if control == "page_random_crops_8k_control":
            out[control] = build_random_control(target_total=target_total, seed=args.seed)
        elif control == "page_school_words_8k_control":
            out[control] = build_school_words_control(target_total=target_total, seed=args.seed)
        else:
            raise ValueError(f"Unknown control: {control}")

    print(json.dumps({
        "target_total": target_total,
        "controls": {
            key: {
                "root": value["root"],
                "merged_train_n": value["merged_train_n"],
                "page_overlap": value["page_overlap"],
                "selected_pages_all_in_base_train": value["selected_pages_all_in_base_train"],
            }
            for key, value in out.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
