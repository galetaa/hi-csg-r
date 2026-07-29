from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.datasets.metadata import read_jsonl


METADATA = Path("data/processed/school_notebooks/metadata.preprocessed.jsonl")
OUT = Path("data/reports/school_notebooks/text_leakage_report.json")


def main() -> None:
    records = read_jsonl(METADATA)

    text_to_splits = defaultdict(set)
    text_to_count = Counter()
    text_to_category = defaultdict(Counter)

    for r in records:
        text = r.get("transcription_modes", {}).get("ctc_default") or r.get("normalized_transcription")
        split = r.get("split")
        category = r.get("metadata", {}).get("category")

        if not text or not split:
            continue

        text_to_splits[text].add(split)
        text_to_count[text] += 1
        text_to_category[text][category] += 1

    leakage = {
        text: sorted(splits)
        for text, splits in text_to_splits.items()
        if len(splits) > 1
    }

    report = {
        "num_records": len(records),
        "num_unique_texts": len(text_to_splits),
        "num_leakage_texts": len(leakage),
        "leakage_ratio_by_unique_text": len(leakage) / max(len(text_to_splits), 1),
        "top_repeated_texts": text_to_count.most_common(100),
        "leakage_preview": dict(list(leakage.items())[:100]),
        "top_repeated_with_categories": [
            {
                "text": text,
                "count": count,
                "splits": sorted(text_to_splits[text]),
                "categories": dict(text_to_category[text]),
            }
            for text, count in text_to_count.most_common(100)
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("records:", report["num_records"])
    print("unique texts:", report["num_unique_texts"])
    print("leakage texts:", report["num_leakage_texts"])
    print("leakage ratio:", report["leakage_ratio_by_unique_text"])
    print("top repeated:")
    for text, count in report["top_repeated_texts"][:30]:
        print(repr(text), count)
    print("wrote:", OUT)


if __name__ == "__main__":
    main()