# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1741 | 1034 | 2788 | 0.1629 | 0.1365 | -0.0264 | [-0.0306, -0.0223] | 0.5575 | 0.4913 | -0.0662 | 0.3960 | 0.4634 | 0.0674 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 607 | 344 | 612 | 0.2187 | 0.1872 | -0.0315 | 0.7239 | 0.6604 | -0.0636 | 0.2348 | 0.2885 | 0.0537 |
| `hkr_words` | 2000 | 627 | 361 | 1012 | 0.1113 | 0.0870 | -0.0244 | 0.4575 | 0.3685 | -0.0889 | 0.4355 | 0.5380 | 0.1025 |
| `school_notebooks_clean` | 2000 | 507 | 329 | 1164 | 0.1708 | 0.1463 | -0.0245 | 0.5275 | 0.4820 | -0.0455 | 0.4825 | 0.5255 | 0.0430 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 76 | 59 | 358 | 0.1707 | 0.1542 | -0.0166 |
| `4-6` | 1538 | 429 | 240 | 869 | 0.1769 | 0.1415 | -0.0354 |
| `7-10` | 1767 | 593 | 361 | 813 | 0.1762 | 0.1497 | -0.0265 |
| `11+` | 1765 | 643 | 374 | 748 | 0.1352 | 0.1139 | -0.0213 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0322, -0.0168]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 507 | 329 | 1164 | 0.1708 | 0.1463 | -0.0245 | 0.4825 | 0.5255 | 0.0430 |
