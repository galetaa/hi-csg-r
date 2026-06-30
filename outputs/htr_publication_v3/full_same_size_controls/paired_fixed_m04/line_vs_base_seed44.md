# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1788 | 1022 | 2753 | 0.1629 | 0.1348 | -0.0281 | [-0.0323, -0.0239] | 0.5575 | 0.4828 | -0.0747 | 0.3960 | 0.4710 | 0.0750 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 599 | 367 | 597 | 0.2187 | 0.1930 | -0.0257 | 0.7239 | 0.6713 | -0.0526 | 0.2348 | 0.2809 | 0.0461 |
| `hkr_words` | 2000 | 624 | 359 | 1017 | 0.1113 | 0.0875 | -0.0238 | 0.4575 | 0.3670 | -0.0905 | 0.4355 | 0.5330 | 0.0975 |
| `school_notebooks_clean` | 2000 | 565 | 296 | 1139 | 0.1708 | 0.1367 | -0.0341 | 0.5275 | 0.4512 | -0.0762 | 0.4825 | 0.5575 | 0.0750 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 80 | 57 | 356 | 0.1707 | 0.1528 | -0.0179 |
| `4-6` | 1538 | 431 | 238 | 869 | 0.1769 | 0.1416 | -0.0353 |
| `7-10` | 1767 | 663 | 336 | 768 | 0.1762 | 0.1407 | -0.0355 |
| `11+` | 1765 | 614 | 391 | 760 | 0.1352 | 0.1181 | -0.0171 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0425, -0.0260]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 565 | 296 | 1139 | 0.1708 | 0.1367 | -0.0341 | 0.4825 | 0.5575 | 0.0750 |
