# HWR200 — audit summary

## Status

HWR200 is converted as a document-condition robustness dataset.

Final metadata:


data/processed/hwr200/metadata.preprocessed.jsonl

Counts
document-condition samples: 9000
total pages: 33030
ocr pages: 33026
feature pages: 33026

Split counts
{
  "train": 6300,
  "test": 1800,
  "val": 900
}

Condition counts
{
  "scan": 3000,
  "photo_light": 3000,
  "photo_dark": 3000
}

Source group counts
{
  "FPR": 525,
  "Originals": 525,
  "Reuse": 7950
}

Quality flags
{
  "mixed_script": 594,
  "duplicate_exact": 63,
  "hwr200_page_preprocess_failed": 4,
  "broken_image": 1
}

Paired robustness index
{
  "num_document_groups": 3000,
  "num_complete_triplets": 3000,
  "num_incomplete_groups": 0
}

Failed pages report
data/reports/hwr200/hwr200_failed_pages_report.csv
