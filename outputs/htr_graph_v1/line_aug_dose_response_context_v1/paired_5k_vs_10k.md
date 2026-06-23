# Paired Comparison: +5k vs +10k School Context Lines

Delta is `10k - 5k`; negative CER delta means +10k is better.

## Overall

| n | wins 10k | wins 5k | ties | 5k CER | 10k CER | delta CER | CI95 | 5k WER | 10k WER | delta WER | 5k exact | 10k exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1309 | 1212 | 3042 | 0.1360 | 0.1351 | -0.0008 | [-0.0044, 0.0030] | 0.4907 | 0.4924 | 0.0017 | 0.4605 | 0.4636 | 0.0031 |

## By Dataset

| dataset | n | wins 10k | wins 5k | ties | 5k CER | 10k CER | delta CER | CI95 | 5k WER | 10k WER | delta WER | 5k exact | 10k exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 439 | 429 | 1132 | 0.0910 | 0.0917 | 0.0007 | [-0.0040, 0.0055] | 0.3865 | 0.3798 | -0.0067 | 0.5145 | 0.5245 | 0.0100 |
| `cyrillic_handwriting` | 1563 | 463 | 417 | 683 | 0.1879 | 0.1858 | -0.0021 | [-0.0093, 0.0054] | 0.6570 | 0.6630 | 0.0060 | 0.2866 | 0.2892 | 0.0026 |
| `school_notebooks_clean` | 2000 | 407 | 366 | 1227 | 0.1403 | 0.1389 | -0.0014 | [-0.0083, 0.0059] | 0.4650 | 0.4718 | 0.0067 | 0.5425 | 0.5390 | -0.0035 |

## By Text Length

| text_len | n | wins 10k | wins 5k | ties | 5k CER | 10k CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 48 | 66 | 379 | 0.1494 | 0.1650 | 0.0156 |
| `4-6` | 1538 | 288 | 324 | 926 | 0.1436 | 0.1485 | 0.0048 |
| `7-10` | 1767 | 497 | 391 | 879 | 0.1467 | 0.1390 | -0.0077 |
| `11+` | 1765 | 476 | 431 | 858 | 0.1148 | 0.1113 | -0.0035 |

## Interpretation

The +10k run has slightly lower overall CER and School CER than +5k, but the paired 5k-vs-10k CER confidence intervals include zero overall and by dataset. +5k keeps slightly better School WER/exact, while +10k is the CER-oriented choice.
