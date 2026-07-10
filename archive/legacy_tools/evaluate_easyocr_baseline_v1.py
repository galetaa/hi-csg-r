from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.htr.metrics import cer, exact_match, wer


SPACE_RE = re.compile(r"\s+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def normalize_prediction(text: str, mode: str) -> str:
    text = SPACE_RE.sub(" ", text.strip())
    if mode == "lower":
        return text.lower()
    if mode == "none":
        return text
    raise ValueError(f"Unsupported normalization mode: {mode}")


def summarize_group(values: dict[str, list[float]]) -> dict[str, float | int]:
    return {
        "n": len(values["cer"]),
        "cer": mean(values["cer"]),
        "wer": mean(values["wer"]),
        "exact": mean(values["exact"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--languages", nargs="+", default=["ru", "en"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--paragraph", action="store_true")
    parser.add_argument("--normalize_prediction", choices=["lower", "none"], default="lower")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    try:
        import easyocr
    except Exception as exc:
        raise SystemExit(
            "easyocr is not installed. Install it in an isolated environment before running this baseline."
        ) from exc

    rows = read_jsonl(Path(args.manifest))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    reader = easyocr.Reader(args.languages, gpu=args.gpu)

    cers: list[float] = []
    wers: list[float] = []
    exacts: list[float] = []
    pred_lens: list[float] = []
    empty_preds: list[float] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"cer": [], "wer": [], "exact": []})
    predictions = []
    examples = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        target = str(row.get("text") or row.get("normalized_transcription") or "")
        raw_parts = reader.readtext(str(image_path), detail=0, paragraph=args.paragraph)
        raw_pred = " ".join(str(part) for part in raw_parts)
        pred = normalize_prediction(raw_pred, args.normalize_prediction)

        c = cer(pred, target)
        w = wer(pred, target)
        e = exact_match(pred, target)
        dataset = str(row.get("dataset") or row.get("source_dataset") or "")
        level = str(row.get("level") or "")
        category = str(row.get("category") or "")
        key = f"{dataset}|{level}|{category}"

        cers.append(c)
        wers.append(w)
        exacts.append(e)
        pred_lens.append(float(len(pred)))
        empty_preds.append(1.0 if pred == "" else 0.0)
        grouped[key]["cer"].append(c)
        grouped[key]["wer"].append(w)
        grouped[key]["exact"].append(e)

        item = {
            "sample_id": row.get("sample_id"),
            "target": target,
            "pred": pred,
            "raw_pred": raw_pred,
            "cer": c,
            "wer": w,
            "exact": e,
            "level": level,
            "category": category,
        }
        predictions.append(item)
        if len(examples) < 50:
            examples.append(item)

        if args.log_every and index % args.log_every == 0:
            print(json.dumps({
                "processed": index,
                "n": len(rows),
                "running_cer": mean(cers),
                "running_wer": mean(wers),
                "running_exact": mean(exacts),
            }, ensure_ascii=False), flush=True)

    summary = {
        "package": "easyocr_baseline_v1",
        "manifest": args.manifest,
        "languages": args.languages,
        "gpu": args.gpu,
        "paragraph": args.paragraph,
        "normalize_prediction": args.normalize_prediction,
        "max_samples": args.max_samples,
        "metrics": {
            "n": len(cers),
            "cer": mean(cers),
            "wer": mean(wers),
            "exact": mean(exacts),
            "pred_len_mean": mean(pred_lens),
            "pred_empty_ratio": mean(empty_preds),
            "grouped": {key: summarize_group(values) for key, values in grouped.items()},
            "examples": examples,
        },
        "publication_limitation": (
            "EasyOCR is a general OCR baseline, not a specialized Russian handwriting model. "
            "Report it as an external OCR reference, not as a strong HTR/SOTA comparison."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out_summary": str(out_dir / "summary.json"),
        "out_predictions": str(out_dir / "predictions.jsonl"),
        "metrics": summary["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
