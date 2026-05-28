from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, Any

from src.datasets.metadata import SampleMetadata, write_jsonl
from src.datasets.text_normalization import normalize_text_en


DATASET_NAME = "iam"


def _safe_symlink_or_copy(src: Path, dst: Path, mode: Literal["symlink", "copy", "none"]) -> str:
    src = src.resolve()

    if mode == "none":
        return str(src)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return str(dst)

    if mode == "symlink":
        dst.symlink_to(src)
        return str(dst)

    if mode == "copy":
        shutil.copy2(src, dst)
        return str(dst)

    raise ValueError(f"Unknown link mode: {mode}")


def parse_writer_ids_from_xml(xml_dir: Path) -> dict[str, str]:
    """
    Возвращает mapping:
      form_id -> writer_id

    IAM XML root выглядит как:
      <form id="a01-000u" writer-id="000" ...>
    """
    form_to_writer: dict[str, str] = {}

    for xml_path in sorted(xml_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            continue

        form_id = root.attrib.get("id") or xml_path.stem
        writer_id = root.attrib.get("writer-id")

        if writer_id:
            form_to_writer[form_id] = writer_id

    return form_to_writer


def parse_iam_lines_txt(lines_txt: Path) -> list[dict[str, Any]]:
    """
    IAM lines.txt формат:

      a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|...

    Поля:
      0 line_id
      1 segmentation_status
      2 graylevel
      3 num_components
      4 bbox_x
      5 bbox_y
      6 bbox_w
      7 bbox_h
      8 transcription, tokens separated by |
    """
    rows: list[dict[str, Any]] = []

    with lines_txt.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")

            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=8)

            if len(parts) < 9:
                raise ValueError(f"Bad IAM lines.txt row at line {line_no}: {line!r}")

            line_id = parts[0]
            segmentation_status = parts[1]

            try:
                graylevel = int(parts[2])
                num_components = int(parts[3])
                bbox = [int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])]
            except ValueError as exc:
                raise ValueError(f"Bad numeric fields at line {line_no}: {line!r}") from exc

            transcription_pipe = parts[8]
            transcription = transcription_pipe.replace("|", " ")

            form_id = line_id.rsplit("-", 1)[0]

            rows.append(
                {
                    "line_id": line_id,
                    "form_id": form_id,
                    "segmentation_status": segmentation_status,
                    "graylevel": graylevel,
                    "num_components": num_components,
                    "bbox": bbox,
                    "raw_transcription": transcription,
                    "raw_transcription_pipe": transcription_pipe,
                }
            )

    return rows


def image_path_for_line(lines_dir: Path, line_id: str) -> Path:
    form_id = line_id.rsplit("-", 1)[0]
    prefix = form_id[:3]
    return lines_dir / prefix / form_id / f"{line_id}.png"


def convert_iam(
    raw_dir: str | Path,
    out_dir: str | Path,
    link_mode: Literal["symlink", "copy", "none"] = "symlink",
) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    lines_txt = raw_dir / "ascii" / "lines.txt"
    xml_dir = raw_dir / "xml"
    lines_dir = raw_dir / "lines"

    required = [lines_txt, xml_dir, lines_dir]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required IAM paths: {missing}")

    form_to_writer = parse_writer_ids_from_xml(xml_dir)
    rows = parse_iam_lines_txt(lines_txt)

    processed_images_dir = out_dir / "images"
    records: list[dict[str, Any]] = []

    missing_images = 0
    missing_writers = 0

    for idx, row in enumerate(rows, start=1):
        line_id = row["line_id"]
        form_id = row["form_id"]
        writer_id = form_to_writer.get(form_id)

        if writer_id is None:
            missing_writers += 1

        src_img = image_path_for_line(lines_dir, line_id)

        if src_img.exists():
            dst = processed_images_dir / f"{line_id}.png"
            image_path = _safe_symlink_or_copy(src_img, dst, link_mode)
        else:
            missing_images += 1
            image_path = str(src_img)

        norm = normalize_text_en(row["raw_transcription"])

        transcription_modes = {
            "raw": norm.raw,
            "nfc": norm.nfc,
            "lower": norm.lower,
            "no_punct": norm.no_punct,
            "ctc_default": norm.ctc_default,
            "ctc_no_punct": norm.ctc_no_punct,
        }

        sample = SampleMetadata(
            sample_id=f"iam_line_{idx:06d}",
            dataset=DATASET_NAME,
            source_id=line_id,
            language="en",
            script="latin",
            level="line",
            writer_id=writer_id,
            image_path=image_path,
            raw_transcription=norm.raw,
            normalized_transcription=norm.ctc_default,
            bbox=row["bbox"],
            transcription_modes=transcription_modes,
            split=None,
            metadata={
                "source_split": None,
                "scan_type": "scan",
                "acquisition_condition": "scan",
                "page_id": form_id,
                "line_id": line_id,
                "word_id": None,
                "quality_flags": [],
                "usable_for_htr": True,
                "usable_for_graph": True,
                "usable_for_gold_subset": True,
                "usable_for_robustness": True,
                "form_id": form_id,
                "original_line_id": line_id,
                "segmentation_status": row["segmentation_status"],
                "iam_graylevel": row["graylevel"],
                "iam_num_components": row["num_components"],
                "raw_transcription_pipe": row["raw_transcription_pipe"],
            },
        )

        records.append(sample.to_dict())

    metadata_path = out_dir / "metadata.jsonl"
    write_jsonl(records, metadata_path)

    print(f"Converted IAM lines: {len(records)}")
    print(f"Missing images: {missing_images}")
    print(f"Missing writer_id: {missing_writers}")
    print(f"Wrote metadata: {metadata_path}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        default="data/interim/iam",
        help="Path containing ascii/, xml/, lines/",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/iam",
    )
    parser.add_argument(
        "--link_mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
    )
    args = parser.parse_args()

    convert_iam(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()