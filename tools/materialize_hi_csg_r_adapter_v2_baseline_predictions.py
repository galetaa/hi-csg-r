from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.htr.metrics import edit_distance
from src.htr.xaligned_hi_csg_r import read_jsonl


def baseline_row(row: dict[str, object]) -> dict[str, object]:
    prediction = str(row["baseline_prediction"])
    target = str(row["target"])
    target_words = max(len(target.split()), 1)
    word_edits = edit_distance(prediction.split(), target.split())
    return {
        **row,
        "prediction": prediction,
        "char_edits": int(row["baseline_char_edits"]),
        "sample_cer": int(row["baseline_char_edits"]) / max(len(target), 1),
        "word_edits": word_edits,
        "target_words": target_words,
        "sample_wer": word_edits / target_words,
        "exact": bool(row["baseline_exact"]),
        "prediction_length": len(prediction),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [baseline_row(row) for row in read_jsonl(args.v2_predictions)]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows: {output}")


if __name__ == "__main__":
    main()
