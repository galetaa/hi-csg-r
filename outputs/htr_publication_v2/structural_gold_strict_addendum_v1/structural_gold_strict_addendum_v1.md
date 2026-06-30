# Structural Gold Strict Addendum v1

This addendum tightens the reporting around the existing 200-sample structural gold subset.
It does not convert the subset into a topology benchmark.

## Sample Composition

- n: 200
- datasets: {'school_notebooks_clean': 156, 'cyrillic_handwriting': 18, 'hkr_words': 26}
- strata: {'clean_core_correct': 40, 'hard_real_correct': 40, 'hard_real_error': 40, 'high_confidence_error': 30, 'rejected_correct_low_confidence': 30, 'numeric_mixed_rare_format': 20}
- token types: {'alpha': 123, 'mixed': 74, 'numeric': 3}

## Binary Diagnostic Usability Metrics

| field | count | n | rate | Wilson 95% CI |
|---|---:|---:|---:|---:|
| `structural_usable` | 200 | 200 | 1.000 | [0.981, 1.000] |
| `foreground_ok` | 200 | 200 | 1.000 | [0.981, 1.000] |
| `skeleton_ok` | 200 | 200 | 1.000 | [0.981, 1.000] |
| `graph_ok` | 200 | 200 | 1.000 | [0.981, 1.000] |

## Severity Profile

| issue | none | minor | severe/dominant | minor+ rate | severe rate |
|---|---:|---:|---:|---:|---:|
| `line_residual` | 156 | 21 | 23 | 0.220 | 0.115 |
| `neighbor_noise` | 197 | 3 | 0 | 0.015 | 0.000 |
| `missed_ink` | 183 | 17 | 0 | 0.085 | 0.000 |
| `false_ink` | 199 | 1 | 0 | 0.005 | 0.000 |
| `false_branches` | 200 | 0 | 0 | 0.000 | 0.000 |
| `broken_strokes` | 200 | 0 | 0 | 0.000 | 0.000 |
| `overconnected` | 199 | 1 | 0 | 0.005 | 0.000 |
| `segmentation_issue` | 200 | 0 | 0 | 0.000 | 0.000 |

## Dataset-Level Caution

| dataset | n | structural usable | foreground ok | skeleton ok | graph ok |
|---|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 18 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hkr_words` | 26 | 1.000 | 1.000 | 1.000 | 1.000 |
| `school_notebooks_clean` | 156 | 1.000 | 1.000 | 1.000 | 1.000 |

## HTR Error Attribution

`htr_error_explained_by_structure`: {'not_applicable': 116, 'no': 84}

## Strict Interpretation

- The subset supports diagnostic usability on the sampled cases.
- It does not prove exact topology recovery, endpoint/junction correctness, stroke order, or pen trajectory.
- No inter-annotator agreement is available; this remains a major publication limitation.
- The subset is dominated by School Notebooks, so HKR/Cyrillic conclusions are weak at dataset level.
