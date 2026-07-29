from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in ["metadata", "source_metadata"]:
        value = row.get(key)

        if isinstance(value, dict):
            result.update(value)

    return result


def numeric(value: Any) -> tuple[int, str]:
    text = str(value or "")

    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def bbox_from_raw(
    row: dict[str, Any] | None,
) -> list[float] | None:
    if row is None:
        return None

    value = row.get("bbox")

    if (
        isinstance(value, list)
        and len(value) >= 4
    ):
        return [
            float(x)
            for x in value[:4]
        ]

    meta = metadata(row)
    value = meta.get("crop_bbox_xyxy")

    if (
        isinstance(value, list)
        and len(value) >= 4
    ):
        return [
            float(x)
            for x in value[:4]
        ]

    return None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ctc_root",
        default=(
            "data/experiments/"
            "htr_baseline_v1_ctc_ready/"
            "school_notebooks_clean"
        ),
    )
    parser.add_argument(
        "--raw_root",
        default=(
            "data/experiments/"
            "htr_baseline_v1/"
            "school_notebooks_clean"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=(
            "outputs/iter2_data_audit/"
            "school_notebooks_v1/"
            "line_groups"
        ),
    )

    args = parser.parse_args()

    ctc_root = Path(args.ctc_root)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)

    all_items = []
    join_missing = []

    for split in ["train", "val", "test"]:
        ctc_path = ctc_root / f"{split}.jsonl"
        raw_path = raw_root / f"{split}.jsonl"

        ctc_rows = read_jsonl(ctc_path)
        raw_rows = read_jsonl(raw_path)

        raw_by_id = {
            str(row["sample_id"]): row
            for row in raw_rows
        }

        for row in ctc_rows:
            sample_id = str(row["sample_id"])
            raw = raw_by_id.get(sample_id)

            if raw is None:
                join_missing.append({
                    "sample_id": sample_id,
                    "split": split,
                })

            ctc_meta = metadata(row)
            raw_meta = metadata(raw or {})

            merged_meta = {
                **raw_meta,
                **ctc_meta,
            }

            bbox = bbox_from_raw(raw)

            page_id = str(
                merged_meta.get("page_id", "")
            )
            line_id = str(
                merged_meta.get("line_id", "")
            )
            word_id = str(
                merged_meta.get("word_id", "")
            )
            source_image_file = str(
                merged_meta.get(
                    "source_image_file",
                    "",
                )
            )

            page_key = (
                source_image_file
                or page_id
            )

            all_items.append({
                "sample_id": sample_id,
                "split": split,
                "text": str(
                    row.get("text", "")
                ),
                "text_len": int(
                    row.get(
                        "text_len",
                        len(str(row.get("text", ""))),
                    )
                ),
                "image_path": str(
                    row.get("image_path", "")
                ),
                "page_id": page_id,
                "page_key": page_key,
                "line_id": line_id,
                "word_id": word_id,
                "bbox": bbox,
                "source_image_file": (
                    source_image_file
                ),
            })

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for item in all_items:
        if not item["page_key"]:
            continue
        if not item["line_id"]:
            continue

        grouped[
            (
                item["split"],
                item["page_key"],
                item["line_id"],
            )
        ].append(item)

    groups = []

    for (
        split,
        page_key,
        line_id,
    ), items in grouped.items():

        def order_key(
            item: dict[str, Any],
        ) -> tuple[float, int, str]:
            bbox = item.get("bbox")

            if bbox:
                return (
                    float(bbox[0]),
                    *numeric(item["word_id"]),
                )

            word_num, word_text = numeric(
                item["word_id"]
            )

            return (
                float(word_num),
                word_num,
                word_text,
            )

        ordered = sorted(
            items,
            key=order_key,
        )

        bboxes = [
            item["bbox"]
            for item in ordered
            if item["bbox"]
        ]

        combined_bbox = None

        if bboxes:
            combined_bbox = [
                min(b[0] for b in bboxes),
                min(b[1] for b in bboxes),
                max(b[2] for b in bboxes),
                max(b[3] for b in bboxes),
            ]

        groups.append({
            "split": split,
            "page_key": page_key,
            "line_id": line_id,
            "sample_count": len(ordered),
            "sample_ids": [
                item["sample_id"]
                for item in ordered
            ],
            "texts": [
                item["text"]
                for item in ordered
            ],
            "combined_text": " ".join(
                item["text"]
                for item in ordered
            ),
            "contains_short_sample": any(
                item["text_len"] <= 2
                for item in ordered
            ),
            "combined_bbox": combined_bbox,
            "items": ordered,
        })

    groups.sort(
        key=lambda row: (
            -row["sample_count"],
            not row["contains_short_sample"],
        )
    )

    multi_groups = [
        row
        for row in groups
        if row["sample_count"] >= 2
    ]

    page_splits: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for item in all_items:
        if item["page_key"]:
            page_splits[
                item["page_key"]
            ].add(item["split"])

    cross_split_pages = {
        page: sorted(splits)
        for page, splits in page_splits.items()
        if len(splits) > 1
    }

    summary = {
        "sample_count": len(all_items),
        "raw_join_missing": len(
            join_missing
        ),
        "group_count_all": len(groups),
        "multi_sample_group_count": len(
            multi_groups
        ),
        "samples_in_multi_groups": sum(
            row["sample_count"]
            for row in multi_groups
        ),
        "multi_groups_with_short_sample": sum(
            row["contains_short_sample"]
            for row in multi_groups
        ),
        "group_size_distribution": dict(
            Counter(
                row["sample_count"]
                for row in groups
            )
        ),
        "cross_split_page_count": len(
            cross_split_pages
        ),
        "cross_split_pages": (
            cross_split_pages
        ),
    }

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        multi_groups,
        out_dir / "natural_line_groups.jsonl",
    )

    write_jsonl(
        join_missing,
        out_dir / "join_missing.jsonl",
    )

    (
        out_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()