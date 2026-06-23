# School Natural-Line Augmentation Dose Response

| model | train_n | line_n | overall CER | overall WER | exact | HKR CER | Cyrillic CER | School CER | School WER | School exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 30000 | 0 | 0.1453 | 0.5134 | 0.4411 | 0.0956 | 0.1932 | 0.1575 | 0.5018 | 0.5060 |
| +2k lines | 31998 | 1998 | 0.1448 | 0.5108 | 0.4440 | 0.0974 | 0.1934 | 0.1541 | 0.4983 | 0.5120 |
| +5k lines | 34999 | 4999 | 0.1360 | 0.4907 | 0.4605 | 0.0910 | 0.1879 | 0.1403 | 0.4650 | 0.5425 |
| +10k lines | 39998 | 9998 | 0.1351 | 0.4924 | 0.4636 | 0.0917 | 0.1858 | 0.1389 | 0.4718 | 0.5390 |

## Paired CER Delta vs Baseline

| model | scope | n | wins | losses | ties | mean delta CER | CI95 low | CI95 high |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| +2k lines | overall | 5563 | 1306 | 1236 | 3021 | -0.0005 | -0.0043 | 0.0031 |
| +2k lines | hkr_words | 2000 | 442 | 445 | 1113 | 0.0018 | -0.0029 | 0.0066 |
| +2k lines | cyrillic_handwriting | 1563 | 453 | 438 | 672 | 0.0002 | -0.0074 | 0.0078 |
| +2k lines | school_notebooks_clean | 2000 | 411 | 353 | 1236 | -0.0034 | -0.0108 | 0.0039 |
| +5k lines | overall | 5563 | 1416 | 1199 | 2948 | -0.0093 | -0.0132 | -0.0056 |
| +5k lines | hkr_words | 2000 | 467 | 420 | 1113 | -0.0046 | -0.0093 | 0.0001 |
| +5k lines | cyrillic_handwriting | 1563 | 475 | 443 | 645 | -0.0053 | -0.0127 | 0.0018 |
| +5k lines | school_notebooks_clean | 2000 | 474 | 336 | 1190 | -0.0171 | -0.0244 | -0.0099 |
| +10k lines | overall | 5563 | 1479 | 1161 | 2923 | -0.0102 | -0.0141 | -0.0061 |
| +10k lines | hkr_words | 2000 | 487 | 415 | 1098 | -0.0039 | -0.0087 | 0.0010 |
| +10k lines | cyrillic_handwriting | 1563 | 491 | 411 | 661 | -0.0074 | -0.0153 | 0.0005 |
| +10k lines | school_notebooks_clean | 2000 | 501 | 335 | 1164 | -0.0186 | -0.0262 | -0.0112 |

## Training Notes

- +2k lines: line input 2000, used 1998, OOV filtered 2, best epoch 71, val CER 0.1003, blank penalty -0.5823.
- +5k lines: line input 5000, used 4999, OOV filtered 1, best epoch 79, val CER 0.0985, blank penalty -0.4203.
- +10k lines: line input 10000, used 9998, OOV filtered 2, best epoch 73, val CER 0.0950, blank penalty -0.5418.
