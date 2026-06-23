# Structural Gold Annotation Summary

## Overall Acceptance

| metric | rate | threshold | passed |
|---|---:|---:|---|
| `structural_usable` | 1.000 | 0.850 | yes |
| `foreground_ok` | 1.000 | 0.850 | yes |
| `skeleton_ok` | 1.000 | 0.800 | yes |
| `graph_ok` | 1.000 | 0.750 | yes |

Completed rows: 200 / 200
All acceptance passed: yes

## By Stratum

| stratum | n | usable | foreground | skeleton | graph |
|---|---:|---:|---:|---:|---:|
| `clean_core_correct` | 40 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hard_real_correct` | 40 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hard_real_error` | 40 | 1.000 | 1.000 | 1.000 | 1.000 |
| `high_confidence_error` | 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| `numeric_mixed_rare_format` | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| `rejected_correct_low_confidence` | 30 | 1.000 | 1.000 | 1.000 | 1.000 |

## By Dataset

| dataset | n | usable | foreground | skeleton | graph |
|---|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 18 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hkr_words` | 26 | 1.000 | 1.000 | 1.000 | 1.000 |
| `school_notebooks_clean` | 156 | 1.000 | 1.000 | 1.000 | 1.000 |

## HTR Error Explained By Structure

| value | count |
|---|---:|
| `no` | 84 |
| `not_applicable` | 116 |

## Severe/Dominant Issue Counts

| issue | severe_or_dominant_count |
|---|---:|
| `line_residual` | 23 |
| `neighbor_noise` | 0 |
| `missed_ink` | 0 |
| `false_ink` | 0 |
| `false_branches` | 0 |
| `broken_strokes` | 0 |
| `overconnected` | 0 |
| `segmentation_issue` | 0 |
