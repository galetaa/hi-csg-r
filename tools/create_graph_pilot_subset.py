from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.datasets.metadata import read_jsonl, write_jsonl


SEED = 42

OUT_PATH = Path("data/pilot/graph_pilot_v1.jsonl")
SUMMARY_PATH = Path("data/pilot/graph_pilot_v1_summary.json")

DATASETS = {
    "iam": Path("data/processed/iam/metadata.preprocessed.jsonl"),
    "cyrillic_handwriting": Path("data/processed/cyrillic_handwriting/metadata.preprocessed.jsonl"),
    "hwr200": Path("data/processed/hwr200/metadata.preprocessed.jsonl"),
    "hkr_forms": Path("data/processed/hkr_forms/metadata.preprocessed.jsonl"),
}


def is_cleanish_for_pilot(record: dict[str, Any]) -> bool:
    flags = set(record.get("metadata", {}).get("quality_flags", []))

    hard_bad = {
        "missing_image",
        "broken_image",
        "empty_raw_transcription",
        "empty_normalized_transcription",
        "ocr_preprocess_failed",
        "feature_preprocess_failed",
        "hwr200_page_preprocess_failed",
    }

    return not bool(flags & hard_bad)


def has_existing_path(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).exists()


def sample_random(records: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(records) <= n:
        return list(records)
    return rng.sample(records, n)


def make_pilot_record(
    pilot_id: str,
    record: dict[str, Any],
    input_image_path: str,
    source_metadata_path: Path,
    selection_reason: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = record.get("metadata", {})

    out = {
        "pilot_id": pilot_id,
        "sample_id": record["sample_id"],
        "dataset": record["dataset"],
        "source_metadata_path": str(source_metadata_path),
        "input_image_path": input_image_path,
        "original_image_path": record.get("image_path"),
        "ocr_image_path": record.get("ocr_image_path"),
        "feature_image_path": record.get("feature_image_path"),
        "level": record.get("level"),
        "split": record.get("split"),
        "language": record.get("language"),
        "script": record.get("script"),
        "writer_id": record.get("writer_id"),
        "raw_transcription": record.get("raw_transcription"),
        "normalized_transcription": record.get("normalized_transcription"),
        "selection_reason": selection_reason,
        "metadata": {
            "source_flags": metadata.get("quality_flags", []),
            "usable_for_htr": metadata.get("usable_for_htr"),
            "usable_for_graph": metadata.get("usable_for_graph"),
            "usable_for_robustness": metadata.get("usable_for_robustness"),
        },
    }

    if extra:
        out.update(extra)

    return out


def create_iam_pilot(rng: random.Random) -> list[dict[str, Any]]:
    path = DATASETS["iam"]
    records = read_jsonl(path)

    candidates = [
        r for r in records
        if r.get("split") in {"train", "val"}
        and r.get("level") == "line"
        and r.get("metadata", {}).get("usable_for_graph") is True
        and r.get("metadata", {}).get("segmentation_status") == "ok"
        and has_existing_path(r.get("feature_image_path"))
        and is_cleanish_for_pilot(r)
    ]

    selected = sample_random(candidates, 100, rng)

    out = []
    for i, r in enumerate(selected, start=1):
        out.append(
            make_pilot_record(
                pilot_id=f"pilot_iam_{i:04d}",
                record=r,
                input_image_path=r["feature_image_path"],
                source_metadata_path=path,
                selection_reason=["iam", "line", "random_train_val", "segmentation_ok"],
            )
        )

    return out


def create_cyrillic_pilot(rng: random.Random) -> list[dict[str, Any]]:
    path = DATASETS["cyrillic_handwriting"]
    records = read_jsonl(path)

    base = [
        r for r in records
        if r.get("split") in {"train", "val"}
        and r.get("metadata", {}).get("usable_for_graph") is True
        and has_existing_path(r.get("feature_image_path"))
        and is_cleanish_for_pilot(r)
    ]

    words = [r for r in base if r.get("level") == "word"]
    phrases = [r for r in base if r.get("level") == "phrase"]

    selected_words = sample_random(words, 100, rng)
    selected_phrases = sample_random(phrases, 50, rng)

    out = []

    for i, r in enumerate(selected_words, start=1):
        out.append(
            make_pilot_record(
                pilot_id=f"pilot_cyr_word_{i:04d}",
                record=r,
                input_image_path=r["feature_image_path"],
                source_metadata_path=path,
                selection_reason=["cyrillic_handwriting", "word", "random_train_val"],
            )
        )

    for i, r in enumerate(selected_phrases, start=1):
        out.append(
            make_pilot_record(
                pilot_id=f"pilot_cyr_phrase_{i:04d}",
                record=r,
                input_image_path=r["feature_image_path"],
                source_metadata_path=path,
                selection_reason=["cyrillic_handwriting", "phrase", "random_train_val"],
            )
        )

    return out


def group_hwr200_triplets(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for r in records:
        metadata = r.get("metadata", {})
        writer_id = r.get("writer_id")
        document_id = metadata.get("document_id")
        condition = metadata.get("acquisition_condition")

        if not writer_id or not document_id or not condition:
            continue

        key = f"{writer_id}::{document_id}"
        groups[key][condition] = r

    return groups


def choose_hwr200_page(record: dict[str, Any]) -> tuple[str | None, int | None]:
    """
    На первом pilot берём первую успешно обработанную feature page.
    Позже можно заменить на more-informative page selection.
    """
    paths = record.get("metadata", {}).get("page_feature_image_paths", [])
    if not paths:
        return None, None

    for idx, p in enumerate(paths, start=1):
        if Path(p).exists():
            return p, idx

    return None, None


def create_hwr200_pilot(rng: random.Random) -> list[dict[str, Any]]:
    path = DATASETS["hwr200"]
    records = read_jsonl(path)

    candidates = [
        r for r in records
        if r.get("split") in {"train", "val"}
        and r.get("metadata", {}).get("usable_for_graph") is True
        and r.get("metadata", {}).get("usable_for_robustness") is True
        and is_cleanish_for_pilot(r)
    ]

    groups = group_hwr200_triplets(candidates)

    complete_keys = [
        key for key, conds in groups.items()
        if {"scan", "photo_light", "photo_dark"} <= set(conds)
    ]

    selected_keys = sample_random(complete_keys, 30, rng)

    out = []
    counter = 0

    for key in selected_keys:
        for condition in ["scan", "photo_light", "photo_dark"]:
            r = groups[key][condition]
            input_path, page_idx = choose_hwr200_page(r)

            if input_path is None:
                continue

            counter += 1

            out.append(
                make_pilot_record(
                    pilot_id=f"pilot_hwr200_{counter:04d}",
                    record=r,
                    input_image_path=input_path,
                    source_metadata_path=path,
                    selection_reason=["hwr200", "document_triplet", condition, "first_valid_page"],
                    extra={
                        "condition": condition,
                        "paired_group": key,
                        "page_idx": page_idx,
                        "level": "page_from_document_condition",
                        "document_id": r.get("metadata", {}).get("document_id"),
                        "source_group": r.get("metadata", {}).get("source_group"),
                    },
                )
            )

    return out


def create_hkr_forms_pilot(rng: random.Random) -> list[dict[str, Any]]:
    path = DATASETS["hkr_forms"]
    records = read_jsonl(path)

    candidates = [
        r for r in records
        if r.get("split") in {"train", "val"}
        and r.get("metadata", {}).get("usable_for_graph") is True
        and has_existing_path(r.get("feature_image_path"))
        and is_cleanish_for_pilot(r)
    ]

    selected = sample_random(candidates, 30, rng)

    out = []

    for i, r in enumerate(selected, start=1):
        out.append(
            make_pilot_record(
                pilot_id=f"pilot_hkr_forms_{i:04d}",
                record=r,
                input_image_path=r["feature_image_path"],
                source_metadata_path=path,
                selection_reason=["hkr_forms", "form_page", "random_train_val", "page_stress"],
                extra={
                    "form_id": r.get("metadata", {}).get("form_id"),
                    "level": "form_page",
                },
            )
        )

    return out


def validate_pilot_records(records: list[dict[str, Any]]) -> None:
    seen = set()

    for r in records:
        pid = r["pilot_id"]
        if pid in seen:
            raise RuntimeError(f"Duplicate pilot_id: {pid}")
        seen.add(pid)

        p = Path(r["input_image_path"])
        if not p.exists():
            raise FileNotFoundError(f"Missing input image for {pid}: {p}")


def main() -> None:
    rng = random.Random(SEED)

    missing = [str(path) for path in DATASETS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing metadata files: {missing}")

    pilot_records: list[dict[str, Any]] = []
    pilot_records.extend(create_iam_pilot(rng))
    pilot_records.extend(create_cyrillic_pilot(rng))
    pilot_records.extend(create_hwr200_pilot(rng))
    pilot_records.extend(create_hkr_forms_pilot(rng))

    validate_pilot_records(pilot_records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(pilot_records, OUT_PATH)

    summary = {
        "seed": SEED,
        "num_records": len(pilot_records),
        "by_dataset": dict(Counter(r["dataset"] for r in pilot_records)),
        "by_level": dict(Counter(r["level"] for r in pilot_records)),
        "by_split": dict(Counter(r["split"] for r in pilot_records)),
        "hwr200_conditions": dict(Counter(r.get("condition") for r in pilot_records if r["dataset"] == "hwr200")),
        "selection_reasons": dict(Counter(reason for r in pilot_records for reason in r["selection_reason"])),
        "output_path": str(OUT_PATH),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote pilot subset: {OUT_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()