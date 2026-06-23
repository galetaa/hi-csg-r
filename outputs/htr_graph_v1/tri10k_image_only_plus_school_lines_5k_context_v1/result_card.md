# tri10k image-only + School natural lines 5k context v1

## Setup

Baseline:
- train root: data/experiments/htr_graph_v1/graph_ready/tri10k_mixed
- output: outputs/htr_graph_v1/tri10k_image_only_v1

Augmented:
- train root: data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_5k_context_v1
- output: outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_5k_context_v1
- added samples: 4999 used from 5000 sampled line crops
- source: School full natural-line groups
- renderer: school_full_line_raw_rgb_v1
- crop note: raw contextual line crops, not clean isolated lines

One sampled line was filtered before training because it contained an out-of-vocabulary `_` character:
`school_full_line_train_1097_46`, text `проволочную поднялся 2_?`.

## Word-level test results

| model | overall CER | overall WER | exact | HKR CER | Cyrillic CER | School CER |
|---|---:|---:|---:|---:|---:|---:|
| baseline image-only | 0.1453 | 0.5134 | 0.4411 | 0.0956 | 0.1932 | 0.1575 |
| +5k School lines | 0.1360 | 0.4907 | 0.4605 | 0.0910 | 0.1879 | 0.1403 |
| delta | -0.0093 | -0.0227 | +0.0194 | -0.0046 | -0.0053 | -0.0171 |

## Paired Comparison

Common test samples: 5563.

| scope | wins | losses | ties | mean delta CER | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| overall | 1416 | 1199 | 2948 | -0.0093 | [-0.0132, -0.0056] |
| School Notebooks | 474 | 336 | 1190 | -0.0171 | [-0.0244, -0.0097] |

## Dataset Details

| dataset | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|
| hkr_words | 0.4000 | 0.3865 | -0.0135 | 0.5010 | 0.5145 | +0.0135 |
| cyrillic_handwriting | 0.6733 | 0.6570 | -0.0163 | 0.2815 | 0.2866 | +0.0051 |
| school_notebooks_clean | 0.5018 | 0.4650 | -0.0368 | 0.5060 | 0.5425 | +0.0365 |

## Interpretation

Adding 5k contextual natural-line School crops improves word-level image-only CTC performance on the unchanged tri10k mixed test set. The gain is strongest on School Notebooks but also appears on HKR and Cyrillic, suggesting the augmentation did not merely overfit to School.

The paired bootstrap confirms the CER reduction overall and on School Notebooks. The improvement is concentrated in text lengths 4+; samples with length 1-3 are neutral.
