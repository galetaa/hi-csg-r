from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.datasets.metadata import read_jsonl, write_jsonl
from src.preprocessing.ocr_preprocess import preprocess_ocr_image
from src.preprocessing.feature_preprocess import preprocess_feature_image


def _preprocess_one_hwr200_record(
    record: dict[str, Any],
    out_dir: str,
    ocr_target_height: int,
) -> tuple[dict[str, Any], int, int]:
    out_dir_path = Path(out_dir)

    ocr_root = out_dir_path / "ocr_pages"
    feature_root = out_dir_path / "feature_pages"

    sample_id = str(record["sample_id"])
    metadata = record.setdefault("metadata", {})
    page_paths = metadata.get("page_image_paths", [])

    page_ocr_paths: list[str] = []
    page_feature_paths: list[str] = []

    processed_pages = 0
    failed_pages = 0

    sample_ocr_dir = ocr_root / sample_id
    sample_feature_dir = feature_root / sample_id

    sample_ocr_dir.mkdir(parents=True, exist_ok=True)
    sample_feature_dir.mkdir(parents=True, exist_ok=True)

    for page_idx, page_path in enumerate(page_paths, start=1):
        suffix = ".png"

        ocr_dst = sample_ocr_dir / f"page_{page_idx:03d}{suffix}"
        feature_dst = sample_feature_dir / f"page_{page_idx:03d}{suffix}"

        try:
            ocr_img = preprocess_ocr_image(
                page_path,
                target_height=ocr_target_height,
                min_width=ocr_target_height,
                autocontrast=True,
                median_denoise=False,
            )

            # Быстрее, чем дефолтный PNG compression.
            # Если размер файлов важнее скорости — убери compress_level.
            ocr_img.save(ocr_dst, compress_level=1)

            feature_img = preprocess_feature_image(
                page_path,
                autocontrast=False,
                weak_denoise=False,
            )
            feature_img.save(feature_dst, compress_level=1)

            page_ocr_paths.append(str(ocr_dst))
            page_feature_paths.append(str(feature_dst))
            processed_pages += 1

        except Exception as exc:
            failed_pages += 1

            metadata.setdefault("quality_flags", [])
            metadata["quality_flags"] = sorted(
                set(metadata["quality_flags"] + ["hwr200_page_preprocess_failed"])
            )

            metadata.setdefault("page_preprocess_errors", []).append(
                {
                    "page_idx": page_idx,
                    "page_path": page_path,
                    "error": repr(exc),
                }
            )

    metadata["page_ocr_image_paths"] = page_ocr_paths
    metadata["page_feature_image_paths"] = page_feature_paths
    metadata["num_page_ocr_images"] = len(page_ocr_paths)
    metadata["num_page_feature_images"] = len(page_feature_paths)

    return record, processed_pages, failed_pages

def preprocess_hwr200_pages(
    metadata_path: str | Path,
    out_dir: str | Path,
    metadata_out: str | Path,
    limit_samples: int | None = None,
    ocr_target_height: int = 128,
    num_workers: int | None = None,
) -> None:
    records = read_jsonl(metadata_path)

    if limit_samples is not None:
        records = records[:limit_samples]

    out_dir = Path(out_dir)

    ocr_root = out_dir / "ocr_pages"
    feature_root = out_dir / "feature_pages"

    ocr_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    processed: list[dict[str, Any] | None] = [None] * len(records)

    total_pages = 0
    failed_pages = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _preprocess_one_hwr200_record,
                record,
                str(out_dir),
                ocr_target_height,
            ): idx
            for idx, record in enumerate(records)
        }

        for done_idx, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]

            try:
                record, ok_pages, bad_pages = future.result()
            except Exception as exc:
                record = records[idx]
                metadata = record.setdefault("metadata", {})
                metadata.setdefault("quality_flags", [])
                metadata["quality_flags"] = sorted(
                    set(metadata["quality_flags"] + ["hwr200_record_preprocess_failed"])
                )
                metadata["record_preprocess_error"] = repr(exc)

                ok_pages = 0
                bad_pages = len(metadata.get("page_image_paths", []))

            processed[idx] = record
            total_pages += ok_pages
            failed_pages += bad_pages

            if done_idx % 500 == 0:
                print(
                    f"processed samples: {done_idx}/{len(records)} "
                    f"pages={total_pages} failed={failed_pages}"
                )

    final_records = [record for record in processed if record is not None]

    write_jsonl(final_records, metadata_out)

    print(f"Wrote metadata: {metadata_out}")
    print(f"Processed records: {len(final_records)}")
    print(f"Processed pages: {total_pages}")
    print(f"Failed pages: {failed_pages}")
    print(f"Workers: {num_workers}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out_dir", default="data/processed/hwr200")
    parser.add_argument("--metadata_out", default="data/processed/hwr200/metadata.preprocessed.jsonl")
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--ocr_target_height", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    preprocess_hwr200_pages(
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        metadata_out=args.metadata_out,
        limit_samples=args.limit_samples,
        ocr_target_height=args.ocr_target_height,
    )


if __name__ == "__main__":
    main()