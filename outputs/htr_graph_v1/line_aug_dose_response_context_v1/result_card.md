# School Natural-Line Context Augmentation Dose Response

## Setup

Task: word-level HTR evaluation on the unchanged tri10k mixed test set.

Baseline:
- output: `outputs/htr_graph_v1/tri10k_image_only_v1`
- train samples: 30000

Augmentation:
- source: School full natural-line groups
- renderer: `school_full_line_raw_rgb_v1`
- crop type: raw contextual line crops, not isolated clean line crops
- trainer/config: same image-only CRNN CTC setup as baseline

Runs:
- A0: baseline image-only
- A2: +1998 usable School contextual line samples
- A5: +4999 usable School contextual line samples
- A10: +9998 usable School contextual line samples

## Test Results

| model | train_n | line_n | overall CER | overall WER | exact | HKR CER | Cyrillic CER | School CER | School WER | School exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 30000 | 0 | 0.1453 | 0.5134 | 0.4411 | 0.0956 | 0.1932 | 0.1575 | 0.5018 | 0.5060 |
| +2k lines | 31998 | 1998 | 0.1448 | 0.5108 | 0.4440 | 0.0974 | 0.1934 | 0.1541 | 0.4983 | 0.5120 |
| +5k lines | 34999 | 4999 | 0.1360 | 0.4907 | 0.4605 | 0.0910 | 0.1879 | 0.1403 | 0.4650 | 0.5425 |
| +10k lines | 39998 | 9998 | 0.1351 | 0.4924 | 0.4636 | 0.0917 | 0.1858 | 0.1389 | 0.4718 | 0.5390 |

## Paired CER Delta vs Baseline

| model | scope | mean delta CER | 95% CI | interpretation |
|---|---|---:|---:|---|
| +2k lines | overall | -0.0005 | [-0.0043, 0.0031] | neutral |
| +2k lines | School | -0.0034 | [-0.0108, 0.0039] | neutral |
| +5k lines | overall | -0.0093 | [-0.0132, -0.0056] | significant gain |
| +5k lines | School | -0.0171 | [-0.0244, -0.0099] | significant gain |
| +10k lines | overall | -0.0102 | [-0.0141, -0.0061] | significant gain |
| +10k lines | School | -0.0186 | [-0.0262, -0.0112] | significant gain |

## Interpretation

The effect is dose-dependent. The +2k augmentation is not enough to produce a statistically clear improvement. Both +5k and +10k improve overall CER and School CER with paired bootstrap confidence intervals fully below zero.

+10k gives the best overall CER and School CER, while +5k gives slightly better School WER/exact. There is no clear HKR/Cyrillic degradation: their CER deltas are small and their confidence intervals include zero.

Current recommendation: keep +10k as the best CER-oriented candidate and +5k as the conservative candidate with stronger School exact/WER. The next decision should depend on whether CER or exact/WER is the primary selection metric.

Operating-point stratification shows that confidence_graph is effective as a risk ranking method but not coverage-fair under a single global threshold. Strict/balanced thresholds accept School samples much more often than HKR/Cyrillic and reject numeric/mixed tokens more often than alphabetic tokens. Very short samples are accepted more readily, so the remaining short-token errors are better described as overconfident ambiguity errors rather than low-confidence failures.