# Cyrillic Handwriting Dataset — audit summary

Status

Dataset is converted, validated, deduplicated by split, and preprocessed.

Final metadata:
data/processed/cyrillic_handwriting/metadata.preprocessed.jsonl

Counts
total: 73830

Split counts
{
  "train": 65033,
  "val": 7232,
  "test": 1563,
  "excluded": 2
}

Level counts
{
  "word": 66060,
  "phrase": 7770
}

Quality flags
{
  "duplicate_exact": 993,
  "mixed_script": 4,
  "empty_normalized_transcription": 2,
  "empty_raw_transcription": 2
}

Text length stats (for non-excluded records)
{
  "count": 73828,
  "min": 1,
  "max": 40,
  "mean": 7.517811670368966
}

Image width stats
{
  "count": 73830,
  "min": 30,
  "max": 1348,
  "mean": 213.58369226601653
}

Image height stats
{
  "count": 73830,
  "min": 15,
  "max": 908,
  "mean": 66.98450494378979
}
