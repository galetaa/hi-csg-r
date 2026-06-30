# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1697 | 1046 | 2820 | 0.1629 | 0.1378 | -0.0251 | [-0.0289, -0.0211] | 0.5575 | 0.4854 | -0.0721 | 0.3960 | 0.4683 | 0.0723 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 557 | 376 | 630 | 0.2187 | 0.1953 | -0.0234 | 0.7239 | 0.6742 | -0.0498 | 0.2348 | 0.2796 | 0.0448 |
| `hkr_words` | 2000 | 585 | 380 | 1035 | 0.1113 | 0.0940 | -0.0174 | 0.4575 | 0.3797 | -0.0778 | 0.4355 | 0.5250 | 0.0895 |
| `school_notebooks_clean` | 2000 | 555 | 290 | 1155 | 0.1708 | 0.1366 | -0.0342 | 0.5275 | 0.4437 | -0.0837 | 0.4825 | 0.5590 | 0.0765 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 68 | 57 | 368 | 0.1707 | 0.1623 | -0.0085 |
| `4-6` | 1538 | 421 | 245 | 872 | 0.1769 | 0.1416 | -0.0353 |
| `7-10` | 1767 | 620 | 345 | 802 | 0.1762 | 0.1454 | -0.0308 |
| `11+` | 1765 | 588 | 399 | 778 | 0.1352 | 0.1200 | -0.0152 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0422, -0.0264]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 555 | 290 | 1155 | 0.1708 | 0.1366 | -0.0342 | 0.4825 | 0.5590 | 0.0765 |
