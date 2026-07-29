from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.metrics import cer, exact_match, wer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def target_text(row: dict[str, Any]) -> str:
    return str(
        row.get("text")
        or row.get("normalized_transcription")
        or row.get("raw_transcription")
        or ""
    )


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def chunks(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    cers = [float(row["cer"]) for row in predictions]
    wers = [float(row["wer"]) for row in predictions]
    exacts = [float(row["exact"]) for row in predictions]
    pred_lens = [float(len(str(row["pred"]))) for row in predictions]
    empty = [1.0 if str(row["pred"]) == "" else 0.0 for row in predictions]

    grouped = defaultdict(lambda: {"cer": [], "wer": [], "exact": []})
    for row in predictions:
        key = f"{row.get('dataset')}|{row.get('level')}|{row.get('category')}"
        grouped[key]["cer"].append(float(row["cer"]))
        grouped[key]["wer"].append(float(row["wer"]))
        grouped[key]["exact"].append(float(row["exact"]))

    return {
        "n": len(predictions),
        "cer": mean(cers),
        "wer": mean(wers),
        "exact": mean(exacts),
        "pred_len_mean": mean(pred_lens),
        "pred_empty_ratio": mean(empty),
        "grouped": {
            key: {
                "n": len(values["cer"]),
                "cer": mean(values["cer"]),
                "wer": mean(values["wer"]),
                "exact": mean(values["exact"]),
            }
            for key, values in grouped.items()
        },
        "examples": predictions[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model_id", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = read_jsonl(Path(args.manifest))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    processor = TrOCRProcessor.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )

    tokenizer = processor.tokenizer
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id

    model.to(device)
    if args.fp16 and device.type == "cuda":
        model.half()
    model.eval()

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_rows in chunks(rows, args.batch_size):
            images = [load_image(Path(row["image_path"])) for row in batch_rows]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            if args.fp16 and device.type == "cuda":
                pixel_values = pixel_values.half()

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for row, pred in zip(batch_rows, texts):
                target = target_text(row)
                pred = str(pred).strip()
                predictions.append({
                    "sample_id": row.get("sample_id"),
                    "dataset": row.get("dataset"),
                    "target": target,
                    "pred": pred,
                    "cer": cer(pred, target),
                    "wer": wer(pred, target),
                    "exact": exact_match(pred, target),
                    "level": row.get("level"),
                    "category": row.get("category"),
                })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline": "TrOCR",
        "model_id": args.model_id,
        "protocol": "pretrained zero-shot generation baseline",
        "manifest": args.manifest,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "fp16": bool(args.fp16 and device.type == "cuda"),
        "local_files_only": args.local_files_only,
        "metrics": summarize(predictions),
        "publication_limitation": (
            "This is an external pretrained zero-shot baseline. It is useful as an "
            "off-the-shelf reference, but it is not a substitute for a fine-tuned "
            "transformer HTR baseline under the same train/test protocol."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "out_dir": str(out_dir),
        "n": summary["metrics"]["n"],
        "cer": summary["metrics"]["cer"],
        "wer": summary["metrics"]["wer"],
        "exact": summary["metrics"]["exact"],
        "protocol": summary["protocol"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
