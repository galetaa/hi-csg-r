# School Notebooks — audit summary

## Status

School Notebooks is converted, validated, and preprocessed as a polygon crop-level handwriting dataset.

Final metadata:

data/processed/school_notebooks/metadata.preprocessed.jsonl

Counts
total samples: 324312

Split counts
{
  "train": 268350,
  "val": 27853,
  "test": 28109
}

Level counts
{
  "word": 320422,
  "phrase": 3890
}

Category counts
{
  "pupil_text": 294123,
  "pupil_comment": 17386,
  "teacher_comment": 12803
}

Quality flags
{
  "polygon_masked_crop": 324312,
  "single_character_or_mark": 40641,
  "duplicate_exact": 2,
  "mixed_script": 77,
  "occluded": 1
}

Image width stats
{
  "count": 324312,
  "min": 26,
  "max": 1835,
  "mean": 255.99082981819976
}

Image height stats
{
  "count": 324312,
  "min": 25,
  "max": 1990,
  "mean": 106.50203507733293
}

Text length stats
{
  "count": 324312,
  "min": 1,
  "max": 23,
  "mean": 5.269545992747725
}

Text leakage check
{
  "num_unique_texts": 59777,
  "num_leakage_texts": 10230,
  "leakage_ratio_by_unique_text": 0.17113605567358683
}
