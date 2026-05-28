from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from src.datasets.metadata import read_jsonl, write_jsonl


MIN_WIDTH = 8
MIN_HEIGHT = 8
MAX_ASPECT_RATIO = 80.0
LOW_CONTRAST_STD_THRESHOLD = 5.0

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_image(path: Path) -> tuple[list[str], dict[str, Any]]:
    flags: list[str] = []
    info: dict[str, Any] = {}

    if not path.exists():
        return ["missing_image"], info

    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode

            info["width"] = width
            info["height"] = height
            info["mode"] = mode

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                flags.append("too_small")

            aspect = width / max(height, 1)
            info["aspect_ratio"] = aspect

            if aspect > MAX_ASPECT_RATIO or aspect < 1 / MAX_ASPECT_RATIO:
                flags.append("extreme_aspect_ratio")

            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            mean = float(stat.mean[0])
            std = float(stat.stddev[0])

            info["gray_mean"] = mean
            info["gray_std"] = std

            if std < LOW_CONTRAST_STD_THRESHOLD:
                flags.append("low_contrast")

            if mean > 250 and std < 3:
                flags.append("mostly_blank")

            if mean < 5 and std < 3:
                flags.append("mostly_black")

    except Exception as exc:
        flags.append("broken_image")
        info["image_error"] = repr(exc)

    return flags, info


def validate_text(raw: str, normalized: str) -> list[str]:
    flags: list[str] = []

    if raw is None or raw == "":
        flags.append("empty_raw_transcription")

    if normalized is None or normalized == "":
        flags.append("empty_normalized_transcription")

    if raw and CONTROL_CHARS_RE.search(raw):
        flags.append("control_characters")

    if normalized and CONTROL_CHARS_RE.search(normalized):
        flags.append("control_characters_normalized")

    return flags


def detect_script_flags(text: str) -> list[str]:
    has_cyr = any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
    has_lat = any("a" <= ch.lower() <= "z" for ch in text)

    flags: list[str] = []
    if has_cyr and has_lat:
        flags.append("mixed_script")

    return flags


def build_alphabet_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, Counter] = defaultdict(Counter)

    for r in records:
        dataset = r["dataset"]
        text = r.get("normalized_transcription", "") or ""
        by_dataset[dataset].update(text)

    report = {}
    for dataset, counter in by_dataset.items():
        report[dataset] = {
            "num_unique_chars": len(counter),
            "chars": [
                {"char": ch, "count": count}
                for ch, count in counter.most_common()
            ],
        }

    return report


def validate_metadata(
    metadata_path: str | Path,
    report_dir: str | Path,
    rewrite_metadata_with_flags: bool = True,
) -> None:
    metadata_path = Path(metadata_path)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(metadata_path)

    issues: list[dict[str, Any]] = []
    dataset_counter = Counter()
    level_counter = Counter()
    image_hashes: dict[str, str] = {}

    seen_sample_ids = set()

    image_widths = []
    image_heights = []
    text_lengths = []

    for idx, record in enumerate(records):
        sample_id = record.get("sample_id")
        dataset = record.get("dataset", "unknown")
        level = record.get("level", "unknown")

        dataset_counter[dataset] += 1
        level_counter[(dataset, level)] += 1

        flags = list(record.get("metadata", {}).get("quality_flags", []))

        if not sample_id:
            flags.append("missing_sample_id")
        elif sample_id in seen_sample_ids:
            flags.append("duplicate_sample_id")
        else:
            seen_sample_ids.add(sample_id)

        image_path = Path(record.get("image_path", ""))
        image_flags, image_info = validate_image(image_path)
        flags.extend(image_flags)

        if "width" in image_info:
            image_widths.append(image_info["width"])
        if "height" in image_info:
            image_heights.append(image_info["height"])

        if image_path.exists() and "broken_image" not in image_flags:
            try:
                file_hash = sha256_file(image_path)
                if file_hash in image_hashes:
                    flags.append("duplicate_exact")
                else:
                    image_hashes[file_hash] = str(image_path)
            except Exception:
                flags.append("hash_failed")

        raw = record.get("raw_transcription", "") or ""
        normalized = record.get("normalized_transcription", "") or ""

        flags.extend(validate_text(raw, normalized))
        flags.extend(detect_script_flags(normalized))

        text_lengths.append(len(normalized))

        flags = sorted(set(flags))

        record.setdefault("metadata", {})
        record["metadata"]["quality_flags"] = flags
        record["metadata"]["image_info"] = image_info

        # Usability rules.
        if any(f in flags for f in ["missing_image", "broken_image", "empty_normalized_transcription"]):
            record["metadata"]["usable_for_htr"] = False

        if any(f in flags for f in ["missing_image", "broken_image", "too_small", "mostly_blank", "mostly_black"]):
            record["metadata"]["usable_for_graph"] = False

        for flag in flags:
            severity = "warning"
            if flag in {"missing_image", "broken_image", "empty_normalized_transcription"}:
                severity = "error"

            issues.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "level": level,
                    "issue_type": flag,
                    "severity": severity,
                    "message": flag,
                    "image_path": str(image_path),
                    "transcription": raw,
                    "suggested_action": "manual_review" if severity == "warning" else "exclude_or_fix",
                }
            )

    if rewrite_metadata_with_flags:
        flagged_path = metadata_path.with_name("metadata.validated.jsonl")
        write_jsonl(records, flagged_path)
        print(f"Wrote validated metadata: {flagged_path}")

    issues_path = report_dir / "dataset_issues.csv"
    with issues_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "dataset",
            "level",
            "issue_type",
            "severity",
            "message",
            "image_path",
            "transcription",
            "suggested_action",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issues)

    stats = {
        "num_samples": len(records),
        "datasets": dict(dataset_counter),
        "levels": {f"{k[0]}::{k[1]}": v for k, v in level_counter.items()},
        "num_issues": len(issues),
        "issue_counts": dict(Counter(i["issue_type"] for i in issues).most_common()),
        "image_width": summarize_numeric(image_widths),
        "image_height": summarize_numeric(image_heights),
        "text_length": summarize_numeric(text_lengths),
    }

    stats_path = report_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    alphabet_report = build_alphabet_report(records)
    alphabet_path = report_dir / "alphabet_report.json"
    alphabet_path.write_text(json.dumps(alphabet_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote issues: {issues_path}")
    print(f"Wrote stats: {stats_path}")
    print(f"Wrote alphabet report: {alphabet_path}")


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--no_rewrite", action="store_true")
    args = parser.parse_args()

    validate_metadata(
        metadata_path=args.metadata,
        report_dir=args.report_dir,
        rewrite_metadata_with_flags=not args.no_rewrite,
    )


if __name__ == "__main__":
    main()