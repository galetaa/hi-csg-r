# Seed Confirmation - Iteration 2 +10k Context

| run | overall CER | WER | exact | HKR CER | Cyrillic CER | School CER | School WER | School exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline seed42 | 0.1453 | 0.5134 | 0.4411 | 0.0956 | 0.1932 | 0.1575 | 0.5018 | 0.5060 |
| +10k seed42 | 0.1351 | 0.4924 | 0.4636 | 0.0917 | 0.1858 | 0.1389 | 0.4718 | 0.5390 |
| baseline seed43 | 0.1489 | 0.5152 | 0.4357 | 0.0962 | 0.1991 | 0.1624 | 0.5112 | 0.4965 |
| +10k seed43 | 0.1371 | 0.4933 | 0.4587 | 0.0966 | 0.1913 | 0.1354 | 0.4470 | 0.5600 |

## Paired Bootstrap vs Same-Seed Baseline

| seed | scope | mean delta CER | CI95 | wins | losses | ties |
|---|---|---:|---:|---:|---:|---:|
| seed42 | `overall` | -0.0102 | [-0.0141, -0.0061] | 1479 | 1161 | 2923 |
| seed42 | `hkr_words` | -0.0039 | [-0.0087, 0.0009] | 487 | 415 | 1098 |
| seed42 | `cyrillic_handwriting` | -0.0074 | [-0.0150, 0.0002] | 491 | 411 | 661 |
| seed42 | `school_notebooks_clean` | -0.0186 | [-0.0261, -0.0111] | 501 | 335 | 1164 |
| seed43 | `overall` | -0.0118 | [-0.0158, -0.0078] | 1471 | 1167 | 2925 |
| seed43 | `hkr_words` | 0.0003 | [-0.0046, 0.0052] | 460 | 458 | 1082 |
| seed43 | `cyrillic_handwriting` | -0.0078 | [-0.0158, -0.0001] | 499 | 425 | 639 |
| seed43 | `school_notebooks_clean` | -0.0270 | [-0.0348, -0.0192] | 512 | 284 | 1204 |

## Conclusion

The +10k contextual School line augmentation improves CER for seed43 as well as seed42. The seed43 overall and School deltas are statistically supported by paired bootstrap confidence intervals below zero.
