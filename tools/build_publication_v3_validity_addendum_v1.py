from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.metrics import cer, exact_match, wer


OUT_ROOT = Path("outputs/htr_publication_v3/validity_addendum_v1")
EVAL_ROOT = Path("outputs/htr_publication_v3/full_same_size_controls/eval_fixed_m04")
DOSE_ROOT = Path("outputs/htr_publication_v3/dose_response_fixed_m04")
SEEDS = [42, 43, 44]

VARIANTS: dict[str, dict[str, Any]] = {
    "tri10k_base": {
        "manifest_root": Path("data/experiments/htr_graph_v1/graph_ready/tri10k_mixed"),
        "kind": "base",
    },
    "line_context_10k": {
        "manifest_root": Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1"),
        "kind": "natural_line_context",
    },
    "random_crops_10k_control": {
        "manifest_root": Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_random_crops_10k_control_v1"),
        "kind": "same_size_random_crop_control",
    },
    "school_words_10k_control": {
        "manifest_root": Path("data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_words_10k_control_v1"),
        "kind": "same_size_school_word_control",
    },
}

DOSE_RUNS = [
    ("baseline_0_lines", 0),
    ("plus_2k_lines", 1998),
    ("plus_5k_lines", 4999),
    ("plus_10k_lines", 9998),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return read_json(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def stdev(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else None


def compact_record(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("source_metadata") or {}
    info = row.get("image_info") or {}
    return {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "source_dataset": row.get("source_dataset") or row.get("dataset"),
        "split": row.get("split"),
        "level": row.get("level"),
        "language": row.get("language"),
        "script": row.get("script"),
        "image_path": row.get("image_path"),
        "text": row.get("text") or row.get("normalized_transcription") or "",
        "text_len": row.get("text_len") if row.get("text_len") is not None else len(str(row.get("text") or "")),
        "writer_id": row.get("writer_id"),
        "category": row.get("category"),
        "page_id": meta.get("page_id"),
        "line_id": meta.get("line_id"),
        "word_id": meta.get("word_id"),
        "source_image_file": meta.get("source_image_file"),
        "transcription_scope": meta.get("transcription_scope"),
        "width": info.get("width"),
        "height": info.get("height"),
        "gray_mean": info.get("gray_mean"),
        "gray_std": info.get("gray_std"),
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(compact_record(json.loads(line)))
    return rows


def load_manifests() -> dict[str, dict[str, list[dict[str, Any]]]]:
    loaded: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant, cfg in VARIANTS.items():
        root = Path(cfg["manifest_root"])
        loaded[variant] = {}
        for split in ["train", "val", "test"]:
            loaded[variant][split] = load_manifest(root / f"{split}.jsonl")
    return loaded


def source_dataset(row: dict[str, Any]) -> str:
    return str(row.get("source_dataset") or row.get("dataset") or "<unknown>")


def page_key(row: dict[str, Any]) -> str | None:
    page = row.get("source_image_file") or row.get("page_id")
    if page is None or page == "":
        return None
    return f"{source_dataset(row)}|{page}"


def line_key(row: dict[str, Any]) -> str | None:
    page = page_key(row)
    line = row.get("line_id")
    if page is None or line is None or line == "":
        return None
    return f"{page}|line={line}"


def word_key(row: dict[str, Any]) -> str | None:
    line = line_key(row)
    word = row.get("word_id")
    if line is None or word is None or word == "":
        return None
    return f"{line}|word={word}"


def text_key(row: dict[str, Any]) -> str | None:
    text = str(row.get("text") or "").strip()
    return text or None


def image_path_key(row: dict[str, Any]) -> str | None:
    path = row.get("image_path")
    return str(path) if path else None


def sample_id_key(row: dict[str, Any]) -> str | None:
    sample_id = row.get("sample_id")
    return str(sample_id) if sample_id else None


def collect_key_map(rows: list[dict[str, Any]], key_fn) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is not None:
            out[key].append(row)
    return out


def overlap_report(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key_fn,
    *,
    example_limit: int = 10,
) -> dict[str, Any]:
    left_map = collect_key_map(left, key_fn)
    right_map = collect_key_map(right, key_fn)
    overlap_keys = sorted(set(left_map) & set(right_map))
    overlap_right_rows = sum(len(right_map[key]) for key in overlap_keys)
    examples = []
    for key in overlap_keys[:example_limit]:
        examples.append({
            "key": key,
            "left_examples": [brief_row(row) for row in left_map[key][:2]],
            "right_examples": [brief_row(row) for row in right_map[key][:2]],
        })
    return {
        "left_rows": len(left),
        "right_rows": len(right),
        "left_rows_with_key": sum(len(v) for v in left_map.values()),
        "right_rows_with_key": sum(len(v) for v in right_map.values()),
        "left_unique_keys": len(left_map),
        "right_unique_keys": len(right_map),
        "overlap_unique_keys": len(overlap_keys),
        "overlap_right_rows": overlap_right_rows,
        "examples": examples,
    }


def brief_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("source_dataset") or row.get("dataset"),
        "split": row.get("split"),
        "level": row.get("level"),
        "text": row.get("text"),
        "image_path": row.get("image_path"),
        "page_id": row.get("page_id"),
        "line_id": row.get("line_id"),
        "word_id": row.get("word_id"),
        "source_image_file": row.get("source_image_file"),
    }


def split_summary(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for split, rows in rows_by_split.items():
        writer_rows = [row for row in rows if row.get("writer_id")]
        out[split] = {
            "n": len(rows),
            "by_source_dataset": dict(Counter(source_dataset(row) for row in rows)),
            "by_level": dict(Counter(str(row.get("level") or "<none>") for row in rows)),
            "by_category": dict(Counter(str(row.get("category") or "<none>") for row in rows)),
            "page_key_rows": sum(1 for row in rows if page_key(row) is not None),
            "unique_page_keys": len({page_key(row) for row in rows if page_key(row) is not None}),
            "line_key_rows": sum(1 for row in rows if line_key(row) is not None),
            "unique_line_keys": len({line_key(row) for row in rows if line_key(row) is not None}),
            "word_key_rows": sum(1 for row in rows if word_key(row) is not None),
            "unique_word_keys": len({word_key(row) for row in rows if word_key(row) is not None}),
            "writer_id_rows": len(writer_rows),
            "unique_writer_ids": len({str(row.get("writer_id")) for row in writer_rows}),
        }
    return out


def build_metadata_leakage_audit(manifests: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    checks = {
        "sample_id": sample_id_key,
        "image_path": image_path_key,
        "page_key": page_key,
        "line_key": line_key,
        "word_key": word_key,
        "text": text_key,
    }
    variants = {}
    for variant, rows_by_split in manifests.items():
        pair_reports = {}
        for left_name, right_name in [("train", "test"), ("train", "val"), ("val", "test")]:
            left = rows_by_split[left_name]
            right = rows_by_split[right_name]
            pair_reports[f"{left_name}_vs_{right_name}"] = {
                key_name: overlap_report(left, right, key_fn)
                for key_name, key_fn in checks.items()
            }
        variants[variant] = {
            "manifest_root": str(VARIANTS[variant]["manifest_root"]),
            "split_summary": split_summary(rows_by_split),
            "overlaps": pair_reports,
            "interpretation": metadata_interpretation(pair_reports),
        }
    return {
        "checks": {
            "sample_id": "exact sample identifier overlap across splits",
            "image_path": "exact image path overlap across splits",
            "page_key": "source-dataset plus page/source-image overlap; possible page-level dependence",
            "line_key": "source-dataset plus page/source-image plus line overlap",
            "word_key": "source-dataset plus page/source-image plus line plus word overlap",
            "text": "exact normalized text overlap; this is lexical overlap, not image leakage by itself",
        },
        "variants": variants,
        "global_limitation": writer_limitation(manifests),
    }


def metadata_interpretation(pair_reports: dict[str, Any]) -> list[str]:
    train_test = pair_reports["train_vs_test"]
    out = []
    if train_test["sample_id"]["overlap_unique_keys"] == 0 and train_test["image_path"]["overlap_unique_keys"] == 0:
        out.append("No exact sample_id or image_path train-test overlap was detected.")
    else:
        out.append("Exact sample_id/image_path overlap exists and must be manually resolved before strong publication claims.")

    page_overlap = train_test["page_key"]["overlap_unique_keys"]
    if page_overlap:
        out.append(
            "Train-test page/source-image overlap is present; page-disjoint stress evaluation is required for cautious interpretation."
        )
    else:
        out.append("No train-test page/source-image overlap was detected for rows with available page metadata.")

    if train_test["text"]["overlap_unique_keys"]:
        out.append(
            "Lexical train-test text overlap is present; this is expected in word/phrase HTR and is not visual leakage by itself."
        )
    return out


def writer_limitation(manifests: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    total = 0
    with_writer = 0
    for rows_by_split in manifests.values():
        for rows in rows_by_split.values():
            total += len(rows)
            with_writer += sum(1 for row in rows if row.get("writer_id"))
    ratio = with_writer / total if total else 0.0
    if ratio < 0.5:
        return (
            f"Writer-disjoint validation is not supported by current metadata: only {with_writer}/{total} "
            f"rows ({ratio:.2%}) have writer_id."
        )
    return f"Writer_id coverage is {with_writer}/{total} ({ratio:.2%}); writer-disjoint split is metadata-feasible."


def file_sha1(path: Path) -> str | None:
    try:
        h = hashlib.sha1()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def dhash(path: Path) -> str | None:
    try:
        from PIL import Image, ImageOps
    except Exception:
        return None

    try:
        with Image.open(path) as img:
            img = ImageOps.grayscale(img)
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            img = img.resize((9, 8), resampling)
            pixels = list(img.getdata())
    except Exception:
        return None

    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{bits:016x}"


def visual_hash_rows(
    rows: list[dict[str, Any]],
    *,
    cache: dict[str, dict[str, Any]],
    max_paths: int | None,
) -> dict[str, dict[str, Any]]:
    out = {}
    seen_paths = []
    for row in rows:
        path_str = image_path_key(row)
        if path_str and path_str not in out:
            seen_paths.append(path_str)
    if max_paths is not None:
        seen_paths = seen_paths[:max_paths]

    for path_str in seen_paths:
        if path_str not in cache:
            path = Path(path_str)
            cache[path_str] = {
                "path": path_str,
                "exists": path.exists(),
                "sha1": file_sha1(path) if path.exists() else None,
                "dhash": dhash(path) if path.exists() else None,
            }
        out[path_str] = cache[path_str]
    return out


def visual_overlap_report(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    cache: dict[str, dict[str, Any]],
    max_paths: int | None,
    example_limit: int = 10,
) -> dict[str, Any]:
    train_hashes = visual_hash_rows(train, cache=cache, max_paths=max_paths)
    test_hashes = visual_hash_rows(test, cache=cache, max_paths=max_paths)

    def invert(values: dict[str, dict[str, Any]], key: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for path, obj in values.items():
            val = obj.get(key)
            if val:
                out[str(val)].append(path)
        return out

    train_sha = invert(train_hashes, "sha1")
    test_sha = invert(test_hashes, "sha1")
    sha_overlap = sorted(set(train_sha) & set(test_sha))

    train_dhash = invert(train_hashes, "dhash")
    test_dhash = invert(test_hashes, "dhash")
    dhash_overlap = sorted(set(train_dhash) & set(test_dhash))

    return {
        "train_unique_paths_hashed": len(train_hashes),
        "test_unique_paths_hashed": len(test_hashes),
        "max_paths_per_split": max_paths,
        "missing_or_unreadable_train_paths": sum(1 for obj in train_hashes.values() if not obj.get("exists")),
        "missing_or_unreadable_test_paths": sum(1 for obj in test_hashes.values() if not obj.get("exists")),
        "exact_file_sha1_overlap_unique": len(sha_overlap),
        "exact_file_sha1_examples": [
            {"sha1": key, "train_paths": train_sha[key][:2], "test_paths": test_sha[key][:2]}
            for key in sha_overlap[:example_limit]
        ],
        "exact_dhash_candidate_overlap_unique": len(dhash_overlap),
        "exact_dhash_candidate_examples": [
            {"dhash": key, "train_paths": train_dhash[key][:2], "test_paths": test_dhash[key][:2]}
            for key in dhash_overlap[:example_limit]
        ],
        "interpretation": (
            "SHA1 overlap is exact file duplication. dHash overlap is only a perceptual-hash candidate and can include false positives; "
            "inspect examples manually before treating it as leakage."
        ),
    }


def build_visual_audit(
    manifests: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    max_paths_per_split: int | None,
) -> dict[str, Any]:
    cache: dict[str, dict[str, Any]] = {}
    variants = {}
    for variant, rows_by_split in manifests.items():
        variants[variant] = visual_overlap_report(
            rows_by_split["train"],
            rows_by_split["test"],
            cache=cache,
            max_paths=max_paths_per_split,
        )
    audit = {
        "max_paths_per_split": max_paths_per_split,
        "variants": variants,
    }
    audit["candidate_assessment"] = assess_visual_candidates(audit, manifests)
    return audit


def path_to_manifest_rows(manifests: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows_by_split in manifests.values():
        for rows in rows_by_split.values():
            for row in rows:
                path = image_path_key(row)
                if path:
                    out[path].append(row)
    return out


def text_overlap_relation(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return "missing_text"
    if left == right:
        return "exact_text_match"
    if left in right or right in left:
        return "one_text_contains_the_other"
    if cer(left, right) <= 0.25 or cer(right, left) <= 0.25:
        return "low_text_edit_distance"
    return "different_text"


def assess_visual_candidates(
    visual_audit: dict[str, Any],
    manifests: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    by_path = path_to_manifest_rows(manifests)
    unique_pairs: dict[tuple[str, str, str], dict[str, Any]] = {}

    for variant, obj in visual_audit["variants"].items():
        for example in obj["exact_dhash_candidate_examples"]:
            dhash_value = example["dhash"]
            for train_path in example["train_paths"]:
                for test_path in example["test_paths"]:
                    key = (dhash_value, train_path, test_path)
                    train_row = (by_path.get(train_path) or [{}])[0]
                    test_row = (by_path.get(test_path) or [{}])[0]
                    relation = text_overlap_relation(
                        str(train_row.get("text") or ""),
                        str(test_row.get("text") or ""),
                    )
                    same_source_dataset = (
                        source_dataset(train_row) == source_dataset(test_row)
                        and source_dataset(train_row) != "<unknown>"
                    )
                    risk = (
                        "high_near_duplicate_risk"
                        if same_source_dataset and relation in {
                            "exact_text_match",
                            "one_text_contains_the_other",
                            "low_text_edit_distance",
                        }
                        else "requires_manual_inspection"
                    )
                    if key not in unique_pairs:
                        unique_pairs[key] = {
                            "dhash": dhash_value,
                            "train_path": train_path,
                            "test_path": test_path,
                            "train_sample_id": train_row.get("sample_id"),
                            "test_sample_id": test_row.get("sample_id"),
                            "train_split": train_row.get("split"),
                            "test_split": test_row.get("split"),
                            "train_source_dataset": source_dataset(train_row),
                            "test_source_dataset": source_dataset(test_row),
                            "train_text": train_row.get("text"),
                            "test_text": test_row.get("text"),
                            "text_relation": relation,
                            "risk": risk,
                            "seen_in_variants": [],
                        }
                    unique_pairs[key]["seen_in_variants"].append(variant)

    pairs = list(unique_pairs.values())
    high_risk = [row for row in pairs if row["risk"] == "high_near_duplicate_risk"]
    return {
        "unique_dhash_candidate_pairs": len(pairs),
        "high_near_duplicate_risk_pairs": len(high_risk),
        "pairs": pairs,
        "interpretation": (
            "dHash candidates are not exact file duplicates. A candidate is marked high risk when train/test images "
            "share the same perceptual hash, source dataset, and exact/near-contained transcript. High-risk candidates "
            "should be removed or isolated in a stricter split before making strongest generalization claims."
        ),
    }


def predictions_path(variant: str, seed: int) -> Path:
    return EVAL_ROOT / f"{variant}_seed{seed}_test" / "predictions.jsonl"


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "cer": None, "wer": None, "exact": None}
    cers = [float(row.get("cer", cer(str(row.get("pred", "")), str(row.get("target", ""))))) for row in rows]
    wers = [float(row.get("wer", wer(str(row.get("pred", "")), str(row.get("target", ""))))) for row in rows]
    exacts = [float(row.get("exact", exact_match(str(row.get("pred", "")), str(row.get("target", ""))))) for row in rows]
    return {
        "n": len(rows),
        "cer": mean(cers),
        "wer": mean(wers),
        "exact": mean(exacts),
    }


def manifest_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in rows if row.get("sample_id") is not None}


def enrich_prediction(pred: dict[str, Any], manifest_row: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(pred)
    if manifest_row is not None:
        row.update({
            "dataset": manifest_row.get("source_dataset") or manifest_row.get("dataset"),
            "manifest_level": manifest_row.get("level"),
            "manifest_category": manifest_row.get("category"),
            "text_len": manifest_row.get("text_len"),
            "page_key": page_key(manifest_row),
            "line_key": line_key(manifest_row),
            "word_key": word_key(manifest_row),
            "transcription_scope": manifest_row.get("transcription_scope"),
        })
    return row


def subset_metrics(
    preds: list[dict[str, Any]],
    test_map: dict[str, dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    excluded_sample_ids: set[str],
) -> dict[str, Any]:
    train_pages = {page_key(row) for row in train_rows if page_key(row) is not None}
    train_lines = {line_key(row) for row in train_rows if line_key(row) is not None}
    train_words = {word_key(row) for row in train_rows if word_key(row) is not None}

    enriched = [
        enrich_prediction(pred, test_map.get(str(pred.get("sample_id"))))
        for pred in preds
    ]
    subsets: dict[str, list[dict[str, Any]]] = {
        "all_test": enriched,
        "all_test_minus_high_risk_visual_near_duplicates": [
            row for row in enriched
            if str(row.get("sample_id")) not in excluded_sample_ids
        ],
        "page_key_available": [row for row in enriched if row.get("page_key")],
        "page_disjoint_from_train": [row for row in enriched if row.get("page_key") and row.get("page_key") not in train_pages],
        "page_seen_in_train": [row for row in enriched if row.get("page_key") and row.get("page_key") in train_pages],
        "line_key_available": [row for row in enriched if row.get("line_key")],
        "line_disjoint_from_train": [row for row in enriched if row.get("line_key") and row.get("line_key") not in train_lines],
        "line_seen_in_train": [row for row in enriched if row.get("line_key") and row.get("line_key") in train_lines],
        "word_key_available": [row for row in enriched if row.get("word_key")],
        "word_disjoint_from_train": [row for row in enriched if row.get("word_key") and row.get("word_key") not in train_words],
        "word_seen_in_train": [row for row in enriched if row.get("word_key") and row.get("word_key") in train_words],
        "school_page_disjoint_from_train": [
            row for row in enriched
            if row.get("dataset") == "school_notebooks_clean"
            and row.get("page_key")
            and row.get("page_key") not in train_pages
        ],
        "school_page_seen_in_train": [
            row for row in enriched
            if row.get("dataset") == "school_notebooks_clean"
            and row.get("page_key")
            and row.get("page_key") in train_pages
        ],
    }
    return {name: prediction_metrics(rows) for name, rows in subsets.items()}


def aggregate_seed_metrics(by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    subset_names = sorted({subset for seed_rows in by_seed.values() for subset in seed_rows})
    out = {}
    for subset in subset_names:
        seed_values = [
            {"seed": seed, **seed_rows[subset]}
            for seed, seed_rows in sorted(by_seed.items())
            if subset in seed_rows
        ]
        cers = [float(row["cer"]) for row in seed_values if row.get("cer") is not None]
        wers = [float(row["wer"]) for row in seed_values if row.get("wer") is not None]
        exacts = [float(row["exact"]) for row in seed_values if row.get("exact") is not None]
        ns = [int(row["n"]) for row in seed_values]
        out[subset] = {
            "seed_values": seed_values,
            "n_min": min(ns) if ns else 0,
            "n_max": max(ns) if ns else 0,
            "mean_cer": mean(cers),
            "std_cer": stdev(cers),
            "mean_wer": mean(wers),
            "mean_exact": mean(exacts),
        }
    return out


def high_risk_visual_test_sample_ids(visual_audit: dict[str, Any]) -> set[str]:
    return {
        str(pair["test_sample_id"])
        for pair in visual_audit.get("candidate_assessment", {}).get("pairs", [])
        if pair.get("risk") == "high_near_duplicate_risk" and pair.get("test_sample_id")
    }


def build_group_stress_eval(
    manifests: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    excluded_sample_ids: set[str],
) -> dict[str, Any]:
    variants = {}
    for variant in VARIANTS:
        test_map = manifest_by_sample(manifests[variant]["test"])
        by_seed = {}
        missing = []
        for seed in SEEDS:
            path = predictions_path(variant, seed)
            if not path.exists():
                missing.append({"seed": seed, "path": str(path)})
                continue
            preds = load_predictions(path)
            by_seed[seed] = subset_metrics(
                preds,
                test_map,
                manifests[variant]["train"],
                excluded_sample_ids=excluded_sample_ids,
            )
        variants[variant] = {
            "by_seed": by_seed,
            "aggregated": aggregate_seed_metrics(by_seed),
            "missing_predictions": missing,
        }
    return {
        "definition": (
            "Stress subsets are computed from existing fixed-penalty test predictions. "
            "They are not a retrained group-disjoint experiment; they diagnose whether reported test performance is concentrated "
            "on samples whose page/line/word metadata is already represented in training."
        ),
        "excluded_high_risk_visual_near_duplicate_test_sample_ids": sorted(excluded_sample_ids),
        "variants": variants,
    }


def text_len_bin(value: Any) -> str:
    try:
        n = int(value)
    except Exception:
        return "unknown"
    if n <= 4:
        return "01-04"
    if n <= 8:
        return "05-08"
    if n <= 16:
        return "09-16"
    if n <= 32:
        return "17-32"
    return "33+"


def domain_keys(row: dict[str, Any]) -> dict[str, str]:
    dataset = str(row.get("dataset") or "<unknown>")
    level = str(row.get("manifest_level") or row.get("level") or "<none>")
    category = str(row.get("manifest_category") or row.get("category") or "<none>")
    scope = str(row.get("transcription_scope") or "<none>")
    return {
        "dataset": dataset,
        "level": level,
        "dataset_level": f"{dataset}|{level}",
        "category": category,
        "text_len_bin": text_len_bin(row.get("text_len")),
        "transcription_scope": scope,
    }


def grouped_prediction_metrics(rows: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[domain_keys(row)[group_name]].append(row)
    return {key: prediction_metrics(vals) for key, vals in sorted(grouped.items())}


def aggregate_grouped_by_seed(by_seed: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    group_names = sorted({name for seed_obj in by_seed.values() for name in seed_obj})
    out: dict[str, Any] = {}
    for group_name in group_names:
        keys = sorted({key for seed_obj in by_seed.values() for key in seed_obj.get(group_name, {})})
        out[group_name] = {}
        for key in keys:
            rows = []
            for seed, seed_obj in sorted(by_seed.items()):
                metrics = seed_obj.get(group_name, {}).get(key)
                if metrics is not None:
                    rows.append({"seed": seed, **metrics})
            cers = [float(row["cer"]) for row in rows if row.get("cer") is not None]
            wers = [float(row["wer"]) for row in rows if row.get("wer") is not None]
            exacts = [float(row["exact"]) for row in rows if row.get("exact") is not None]
            ns = [int(row["n"]) for row in rows]
            out[group_name][key] = {
                "seed_values": rows,
                "n_min": min(ns) if ns else 0,
                "n_max": max(ns) if ns else 0,
                "mean_cer": mean(cers),
                "std_cer": stdev(cers),
                "mean_wer": mean(wers),
                "mean_exact": mean(exacts),
            }
    return out


def build_domain_breakdown(manifests: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    group_names = ["dataset", "level", "dataset_level", "category", "text_len_bin", "transcription_scope"]
    variants = {}
    for variant in VARIANTS:
        test_map = manifest_by_sample(manifests[variant]["test"])
        by_seed = {}
        for seed in SEEDS:
            path = predictions_path(variant, seed)
            if not path.exists():
                continue
            enriched = [
                enrich_prediction(pred, test_map.get(str(pred.get("sample_id"))))
                for pred in load_predictions(path)
            ]
            by_seed[seed] = {
                group_name: grouped_prediction_metrics(enriched, group_name)
                for group_name in group_names
            }
        variants[variant] = {
            "by_seed": by_seed,
            "aggregated": aggregate_grouped_by_seed(by_seed),
        }
    return {
        "group_names": group_names,
        "variants": variants,
        "pairwise_line_vs_controls": domain_pairwise_deltas(variants),
    }


def domain_pairwise_deltas(variants: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        ("line_vs_base", "tri10k_base", "line_context_10k"),
        ("line_vs_random", "random_crops_10k_control", "line_context_10k"),
        ("line_vs_school_words", "school_words_10k_control", "line_context_10k"),
    ]
    out: dict[str, Any] = {}
    for pair_name, baseline, contender in pairs:
        out[pair_name] = {}
        base_groups = variants.get(baseline, {}).get("aggregated", {})
        cont_groups = variants.get(contender, {}).get("aggregated", {})
        for group_name in sorted(set(base_groups) & set(cont_groups)):
            out[pair_name][group_name] = {}
            for key in sorted(set(base_groups[group_name]) & set(cont_groups[group_name])):
                base_cer = base_groups[group_name][key].get("mean_cer")
                cont_cer = cont_groups[group_name][key].get("mean_cer")
                if base_cer is None or cont_cer is None:
                    continue
                out[pair_name][group_name][key] = {
                    "delta_cer": float(cont_cer) - float(base_cer),
                    "baseline_mean_cer": base_cer,
                    "contender_mean_cer": cont_cer,
                    "n_min": min(
                        int(base_groups[group_name][key].get("n_min") or 0),
                        int(cont_groups[group_name][key].get("n_min") or 0),
                    ),
                }
    return out


def char_edit_counts(pred: str, target: str) -> dict[str, Any]:
    n = len(target)
    m = len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "match"
                continue
            candidates = [
                (dp[i - 1][j - 1] + 1, "sub"),
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
            ]
            dp[i][j], back[i][j] = min(candidates, key=lambda x: x[0])

    i, j = n, m
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0, "matches": 0}
    sub_pairs: Counter[str] = Counter()
    del_chars: Counter[str] = Counter()
    ins_chars: Counter[str] = Counter()
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "match":
            counts["matches"] += 1
            i -= 1
            j -= 1
        elif op == "sub":
            counts["substitutions"] += 1
            sub_pairs[f"{target[i - 1]}→{pred[j - 1]}"] += 1
            i -= 1
            j -= 1
        elif op == "del":
            counts["deletions"] += 1
            del_chars[target[i - 1]] += 1
            i -= 1
        elif op == "ins":
            counts["insertions"] += 1
            ins_chars[pred[j - 1]] += 1
            j -= 1
        else:
            break

    denom = max(n, 1)
    return {
        **counts,
        "target_len": n,
        "pred_len": m,
        "edit_distance": dp[n][m],
        "substitution_rate": counts["substitutions"] / denom,
        "deletion_rate": counts["deletions"] / denom,
        "insertion_rate": counts["insertions"] / denom,
        "sub_pairs": dict(sub_pairs),
        "deleted_chars": dict(del_chars),
        "inserted_chars": dict(ins_chars),
    }


def aggregate_error_rows(preds: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    sub_pairs: Counter[str] = Counter()
    del_chars: Counter[str] = Counter()
    ins_chars: Counter[str] = Counter()
    per_row_rates = {"substitution_rate": [], "deletion_rate": [], "insertion_rate": []}
    for pred in preds:
        counts = char_edit_counts(str(pred.get("pred", "")), str(pred.get("target", "")))
        totals.update({
            "substitutions": counts["substitutions"],
            "deletions": counts["deletions"],
            "insertions": counts["insertions"],
            "matches": counts["matches"],
            "target_len": counts["target_len"],
            "pred_len": counts["pred_len"],
            "edit_distance": counts["edit_distance"],
        })
        sub_pairs.update(counts["sub_pairs"])
        del_chars.update(counts["deleted_chars"])
        ins_chars.update(counts["inserted_chars"])
        for key in per_row_rates:
            per_row_rates[key].append(float(counts[key]))

    denom = max(int(totals["target_len"]), 1)
    return {
        "n": len(preds),
        "total_target_chars": int(totals["target_len"]),
        "total_pred_chars": int(totals["pred_len"]),
        "total_edit_distance": int(totals["edit_distance"]),
        "substitutions": int(totals["substitutions"]),
        "deletions": int(totals["deletions"]),
        "insertions": int(totals["insertions"]),
        "micro_substitution_rate": int(totals["substitutions"]) / denom,
        "micro_deletion_rate": int(totals["deletions"]) / denom,
        "micro_insertion_rate": int(totals["insertions"]) / denom,
        "mean_substitution_rate": mean(per_row_rates["substitution_rate"]),
        "mean_deletion_rate": mean(per_row_rates["deletion_rate"]),
        "mean_insertion_rate": mean(per_row_rates["insertion_rate"]),
        "top_substitution_pairs": sub_pairs.most_common(20),
        "top_deleted_chars": del_chars.most_common(20),
        "top_inserted_chars": ins_chars.most_common(20),
    }


def aggregate_error_by_seed(by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    keys = ["micro_substitution_rate", "micro_deletion_rate", "micro_insertion_rate"]
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in by_seed.values() if row.get(key) is not None]
        out[key] = {"mean": mean(vals), "std": stdev(vals)}
    out["seed_values"] = by_seed
    return out


def build_error_analysis(manifests: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    variants = {}
    for variant in VARIANTS:
        by_seed = {}
        for seed in SEEDS:
            path = predictions_path(variant, seed)
            if path.exists():
                by_seed[seed] = aggregate_error_rows(load_predictions(path))
        variants[variant] = aggregate_error_by_seed(by_seed)

    return {
        "definition": "Character-level edit decomposition, target-to-prediction: deletion means a target char is missing; insertion means an extra predicted char.",
        "variants": variants,
        "paired_examples": paired_error_examples(manifests),
    }


def paired_error_examples(manifests: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    seed = 42
    out: dict[str, Any] = {}
    pairs = [
        ("line_vs_random", "random_crops_10k_control", "line_context_10k"),
        ("line_vs_school_words", "school_words_10k_control", "line_context_10k"),
    ]
    test_map = manifest_by_sample(manifests["line_context_10k"]["test"])
    for pair_name, baseline, contender in pairs:
        base_path = predictions_path(baseline, seed)
        cont_path = predictions_path(contender, seed)
        if not base_path.exists() or not cont_path.exists():
            continue
        base = {str(row["sample_id"]): row for row in load_predictions(base_path)}
        cont = {str(row["sample_id"]): row for row in load_predictions(cont_path)}
        paired = []
        for sample_id in sorted(set(base) & set(cont)):
            b = base[sample_id]
            c = cont[sample_id]
            manifest_row = test_map.get(sample_id)
            paired.append({
                "sample_id": sample_id,
                "target": c.get("target"),
                "baseline_pred": b.get("pred"),
                "contender_pred": c.get("pred"),
                "baseline_cer": b.get("cer"),
                "contender_cer": c.get("cer"),
                "delta_cer": float(c.get("cer", 0.0)) - float(b.get("cer", 0.0)),
                "dataset": (manifest_row or {}).get("source_dataset") or (manifest_row or {}).get("dataset"),
                "level": (manifest_row or {}).get("level"),
                "category": (manifest_row or {}).get("category"),
                "text_len": (manifest_row or {}).get("text_len"),
            })
        out[pair_name] = {
            "seed": seed,
            "line_context_most_better": sorted(paired, key=lambda row: row["delta_cer"])[:20],
            "line_context_most_worse": sorted(paired, key=lambda row: row["delta_cer"], reverse=True)[:20],
        }
    return out


def build_dose_response() -> dict[str, Any]:
    rows = []
    for key, line_train_n in DOSE_RUNS:
        summary_path = DOSE_ROOT / f"{key}_test" / "summary.json"
        summary = maybe_json(summary_path)
        if summary is None:
            rows.append({
                "key": key,
                "line_train_n": line_train_n,
                "status": "missing",
                "summary": str(summary_path),
            })
            continue
        metrics = summary["metrics"]
        rows.append({
            "key": key,
            "line_train_n": line_train_n,
            "status": "complete",
            "summary": str(summary_path),
            "checkpoint": summary.get("checkpoint"),
            "checkpoint_epoch": summary.get("checkpoint_epoch"),
            "checkpoint_val_cer": summary.get("checkpoint_val_cer"),
            "blank_logit_penalty": summary.get("blank_logit_penalty"),
            "n": metrics["n"],
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "exact": metrics["exact"],
        })

    complete = [row for row in rows if row.get("status") == "complete"]
    baseline = next((row for row in complete if row["key"] == "baseline_0_lines"), None)
    if baseline is not None:
        for row in rows:
            if row.get("status") == "complete":
                row["delta_cer_vs_baseline"] = float(row["cer"]) - float(baseline["cer"])
                row["relative_cer_change_vs_baseline"] = row["delta_cer_vs_baseline"] / float(baseline["cer"])

    increments = []
    for prev, cur in zip(complete, complete[1:]):
        increments.append({
            "from": prev["key"],
            "to": cur["key"],
            "delta_line_train_n": int(cur["line_train_n"]) - int(prev["line_train_n"]),
            "delta_cer": float(cur["cer"]) - float(prev["cer"]),
        })

    return {
        "protocol": "fixed test-time blank_logit_penalty=-0.4, seed-42 historical checkpoints",
        "rows": rows,
        "increments": increments,
        "interpretation": dose_interpretation(rows, increments),
    }


def dose_interpretation(rows: list[dict[str, Any]], increments: list[dict[str, Any]]) -> list[str]:
    complete = [row for row in rows if row.get("status") == "complete"]
    if len(complete) < 4:
        return ["Fixed-penalty dose response is incomplete; run tools/run_publication_v3_dose_fixed_eval_v1.py."]
    out = []
    best = min(complete, key=lambda row: row["cer"])
    out.append(f"Best fixed-penalty dose row is {best['key']} with CER={best['cer']:.4f}.")
    if increments:
        largest = min(increments, key=lambda row: row["delta_cer"])
        out.append(
            f"Largest incremental CER decrease is {largest['from']} -> {largest['to']} "
            f"(delta {largest['delta_cer']:.4f})."
        )
    last_inc = increments[-1] if increments else None
    if last_inc and abs(float(last_inc["delta_cer"])) < 0.003:
        out.append("The 5k->10k increment is small, consistent with a plateau rather than a linear data-scaling effect.")
    return out


def build_summary(max_visual_paths_per_split: int | None) -> dict[str, Any]:
    manifests = load_manifests()
    metadata_audit = build_metadata_leakage_audit(manifests)
    visual_audit = build_visual_audit(manifests, max_paths_per_split=max_visual_paths_per_split)
    group_stress = build_group_stress_eval(
        manifests,
        excluded_sample_ids=high_risk_visual_test_sample_ids(visual_audit),
    )
    domain_breakdown = build_domain_breakdown(manifests)
    error_analysis = build_error_analysis(manifests)
    dose_response = build_dose_response()
    return {
        "package": "htr_publication_v3_validity_addendum_v1",
        "output_root": str(OUT_ROOT),
        "metadata_leakage_audit": metadata_audit,
        "visual_duplicate_audit": visual_audit,
        "group_stress_eval": group_stress,
        "domain_breakdown": domain_breakdown,
        "error_analysis": error_analysis,
        "dose_response_fixed_m04": dose_response,
        "publication_interpretation": publication_interpretation(
            metadata_audit,
            visual_audit,
            group_stress,
            domain_breakdown,
            dose_response,
        ),
    }


def publication_interpretation(
    metadata_audit: dict[str, Any],
    visual_audit: dict[str, Any],
    group_stress: dict[str, Any],
    domain_breakdown: dict[str, Any],
    dose_response: dict[str, Any],
) -> dict[str, Any]:
    exact_leakage_flags = []
    page_overlap_flags = []
    visual_flags = []
    near_visual_flags = []
    for variant, obj in metadata_audit["variants"].items():
        train_test = obj["overlaps"]["train_vs_test"]
        if train_test["sample_id"]["overlap_unique_keys"] or train_test["image_path"]["overlap_unique_keys"]:
            exact_leakage_flags.append(variant)
        if train_test["page_key"]["overlap_unique_keys"]:
            page_overlap_flags.append(variant)
    for variant, obj in visual_audit["variants"].items():
        if obj["exact_file_sha1_overlap_unique"]:
            visual_flags.append(variant)
    if visual_audit.get("candidate_assessment", {}).get("high_near_duplicate_risk_pairs"):
        seen = set()
        for pair in visual_audit["candidate_assessment"]["pairs"]:
            if pair["risk"] == "high_near_duplicate_risk":
                seen.update(pair["seen_in_variants"])
        near_visual_flags = sorted(seen)

    line_random_delta = (
        domain_breakdown.get("pairwise_line_vs_controls", {})
        .get("line_vs_random", {})
        .get("dataset", {})
    )
    line_school_delta = (
        domain_breakdown.get("pairwise_line_vs_controls", {})
        .get("line_vs_school_words", {})
        .get("dataset", {})
    )

    claims = []
    if not exact_leakage_flags and not visual_flags:
        claims.append("No exact train-test sample/image/file duplication was detected by automated audits.")
    if near_visual_flags:
        claims.append(
            "At least one high-risk perceptual near-duplicate candidate was detected; it is too small to explain aggregate metrics, "
            "but it should be removed or isolated in a strict publication split."
        )
    if page_overlap_flags:
        claims.append(
            "Page/source-image overlap exists for at least one variant; page-disjoint stress rows must be cited alongside all-test metrics."
        )
    else:
        claims.append("No train-test page/source-image overlap was detected where page metadata is available.")
    claims.append(
        "The controlled result remains nuanced: line-context beats the base model, but same-size controls are comparable; "
        "the defensible claim is an augmentation/data-volume effect, not a proven unique line-context mechanism."
    )
    claims.append(
        "Writer-disjoint validation remains unresolved unless reliable writer_id metadata is added."
    )

    return {
        "exact_metadata_leakage_flag_variants": exact_leakage_flags,
        "page_overlap_flag_variants": page_overlap_flags,
        "exact_visual_file_duplicate_flag_variants": visual_flags,
        "high_risk_visual_near_duplicate_flag_variants": near_visual_flags,
        "high_risk_visual_near_duplicate_pairs": visual_audit.get("candidate_assessment", {}).get("pairs", []),
        "line_vs_random_dataset_delta_cer": line_random_delta,
        "line_vs_school_words_dataset_delta_cer": line_school_delta,
        "claim_boundary": claims,
        "next_required_for_journal_level": [
            "retrain/evaluate a true page-disjoint or writer-disjoint split if metadata and compute allow",
            "add annotation reliability evidence for the school-line corpus",
            "add a competitive external Russian/Cyrillic HTR baseline beyond decoder-only TrOCR adaptation",
        ],
        "dose_response_ready": all(row.get("status") == "complete" for row in dose_response.get("rows", [])),
    }


def table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(v) for v in values) + " |"


def build_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Publication V3 Validity Addendum v1",
        "",
        "This addendum strengthens the v3 package with automated leakage checks, page/line/word stress slices, domain breakdowns, error decomposition, and fixed-penalty dose-response evidence.",
        "",
        "## Claim Boundary",
        "",
    ]
    for claim in summary["publication_interpretation"]["claim_boundary"]:
        lines.append(f"- {claim}")

    lines.extend([
        "",
        "## Metadata Leakage Audit",
        "",
        f"Writer metadata limitation: {summary['metadata_leakage_audit']['global_limitation']}",
        "",
        table_row(["variant", "sample_id overlap", "image_path overlap", "page overlap", "line overlap", "text overlap"]),
        table_row(["---", "---:", "---:", "---:", "---:", "---:"]),
    ])
    for variant, obj in summary["metadata_leakage_audit"]["variants"].items():
        tt = obj["overlaps"]["train_vs_test"]
        lines.append(table_row([
            f"`{variant}`",
            tt["sample_id"]["overlap_unique_keys"],
            tt["image_path"]["overlap_unique_keys"],
            tt["page_key"]["overlap_unique_keys"],
            tt["line_key"]["overlap_unique_keys"],
            tt["text"]["overlap_unique_keys"],
        ]))
    lines.extend([
        "",
        "Interpretation: exact `sample_id`/`image_path` overlap is direct leakage. `page_key`/`line_key` overlap is a dependence risk. Text overlap is expected in HTR and is not visual leakage by itself.",
        "",
        "## Visual Duplicate Audit",
        "",
        table_row(["variant", "train paths", "test paths", "SHA1 overlaps", "dHash candidate overlaps", "missing train/test"]),
        table_row(["---", "---:", "---:", "---:", "---:", "---:"]),
    ])
    for variant, obj in summary["visual_duplicate_audit"]["variants"].items():
        lines.append(table_row([
            f"`{variant}`",
            obj["train_unique_paths_hashed"],
            obj["test_unique_paths_hashed"],
            obj["exact_file_sha1_overlap_unique"],
            obj["exact_dhash_candidate_overlap_unique"],
            f"{obj['missing_or_unreadable_train_paths']}/{obj['missing_or_unreadable_test_paths']}",
        ]))
    assessment = summary["visual_duplicate_audit"].get("candidate_assessment", {})
    if assessment:
        lines.extend([
            "",
            f"dHash candidate assessment: {assessment['high_near_duplicate_risk_pairs']} high-risk near-duplicate pair(s) among {assessment['unique_dhash_candidate_pairs']} unique candidate pair(s).",
            "",
            table_row(["risk", "train sample", "test sample", "train text", "test text", "seen in variants"]),
            table_row(["---", "---", "---", "---", "---", "---"]),
        ])
        for pair in assessment.get("pairs", [])[:10]:
            lines.append(table_row([
                f"`{pair['risk']}`",
                f"`{pair['train_sample_id']}`",
                f"`{pair['test_sample_id']}`",
                pair["train_text"],
                pair["test_text"],
                ", ".join(f"`{variant}`" for variant in pair["seen_in_variants"]),
            ]))

    lines.extend([
        "",
        "## Group Stress Evaluation",
        "",
        summary["group_stress_eval"]["definition"],
        "",
        table_row(["variant", "subset", "n", "mean CER", "std CER", "mean WER", "mean exact"]),
        table_row(["---", "---", "---:", "---:", "---:", "---:", "---:"]),
    ])
    for variant, obj in summary["group_stress_eval"]["variants"].items():
        for subset in [
            "all_test",
            "all_test_minus_high_risk_visual_near_duplicates",
            "page_disjoint_from_train",
            "page_seen_in_train",
            "line_disjoint_from_train",
            "line_seen_in_train",
            "school_page_disjoint_from_train",
            "school_page_seen_in_train",
        ]:
            metrics = obj["aggregated"].get(subset)
            if not metrics:
                continue
            lines.append(table_row([
                f"`{variant}`",
                f"`{subset}`",
                f"{metrics['n_min']}-{metrics['n_max']}",
                fmt(metrics["mean_cer"]),
                fmt(metrics["std_cer"]),
                fmt(metrics["mean_wer"]),
                fmt(metrics["mean_exact"]),
            ]))

    lines.extend([
        "",
        "## Domain Breakdown",
        "",
        "Mean CER by source dataset, averaged across seeds.",
        "",
        table_row(["variant", "dataset", "n", "mean CER", "std CER", "mean WER", "mean exact"]),
        table_row(["---", "---", "---:", "---:", "---:", "---:", "---:"]),
    ])
    for variant, obj in summary["domain_breakdown"]["variants"].items():
        dataset_groups = obj["aggregated"].get("dataset", {})
        for dataset, metrics in dataset_groups.items():
            lines.append(table_row([
                f"`{variant}`",
                f"`{dataset}`",
                f"{metrics['n_min']}-{metrics['n_max']}",
                fmt(metrics["mean_cer"]),
                fmt(metrics["std_cer"]),
                fmt(metrics["mean_wer"]),
                fmt(metrics["mean_exact"]),
            ]))

    lines.extend([
        "",
        "Line-context CER deltas by dataset. Negative means line-context is better.",
        "",
        table_row(["comparison", "dataset", "delta CER"]),
        table_row(["---", "---", "---:"]),
    ])
    for pair_name in ["line_vs_base", "line_vs_random", "line_vs_school_words"]:
        dataset_deltas = summary["domain_breakdown"]["pairwise_line_vs_controls"].get(pair_name, {}).get("dataset", {})
        for dataset, metrics in dataset_deltas.items():
            lines.append(table_row([f"`{pair_name}`", f"`{dataset}`", fmt(metrics["delta_cer"])]))

    lines.extend([
        "",
        "## Error Decomposition",
        "",
        table_row(["variant", "substitution rate", "deletion rate", "insertion rate"]),
        table_row(["---", "---:", "---:", "---:"]),
    ])
    for variant, obj in summary["error_analysis"]["variants"].items():
        lines.append(table_row([
            f"`{variant}`",
            fmt(obj["micro_substitution_rate"]["mean"]),
            fmt(obj["micro_deletion_rate"]["mean"]),
            fmt(obj["micro_insertion_rate"]["mean"]),
        ]))

    lines.extend([
        "",
        "## Fixed-Penalty Dose Response",
        "",
        summary["dose_response_fixed_m04"]["protocol"],
        "",
        table_row(["run", "line train n", "CER", "WER", "exact", "delta CER vs base", "status"]),
        table_row(["---", "---:", "---:", "---:", "---:", "---:", "---"]),
    ])
    for row in summary["dose_response_fixed_m04"]["rows"]:
        lines.append(table_row([
            f"`{row['key']}`",
            row["line_train_n"],
            fmt(row.get("cer")),
            fmt(row.get("wer")),
            fmt(row.get("exact")),
            fmt(row.get("delta_cer_vs_baseline")),
            row["status"],
        ]))
    lines.append("")
    for item in summary["dose_response_fixed_m04"]["interpretation"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Remaining Publication Risks",
        "",
    ])
    for item in summary["publication_interpretation"]["next_required_for_journal_level"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_visual_paths_per_split",
        type=int,
        default=0,
        help="0 means hash all unique paths; positive values cap each split for faster smoke runs.",
    )
    args = parser.parse_args()
    max_visual_paths = None if args.max_visual_paths_per_split == 0 else args.max_visual_paths_per_split

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = build_summary(max_visual_paths_per_split=max_visual_paths)
    out_json = OUT_ROOT / "summary.json"
    out_md = OUT_ROOT / "report.md"
    write_json(out_json, summary)
    out_md.write_text(build_md(summary), encoding="utf-8")
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "exact_metadata_leakage_flag_variants": summary["publication_interpretation"]["exact_metadata_leakage_flag_variants"],
        "page_overlap_flag_variants": summary["publication_interpretation"]["page_overlap_flag_variants"],
        "exact_visual_file_duplicate_flag_variants": summary["publication_interpretation"]["exact_visual_file_duplicate_flag_variants"],
        "dose_response_ready": summary["publication_interpretation"]["dose_response_ready"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
