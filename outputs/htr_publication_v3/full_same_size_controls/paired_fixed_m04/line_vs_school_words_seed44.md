# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1325 | 1247 | 2991 | 0.1378 | 0.1348 | -0.0029 | [-0.0067, 0.0008] | 0.4854 | 0.4828 | -0.0027 | 0.4683 | 0.4710 | 0.0027 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 482 | 461 | 620 | 0.1953 | 0.1930 | -0.0024 | 0.6742 | 0.6713 | -0.0028 | 0.2796 | 0.2809 | 0.0013 |
| `hkr_words` | 2000 | 464 | 409 | 1127 | 0.0940 | 0.0875 | -0.0064 | 0.3797 | 0.3670 | -0.0127 | 0.5250 | 0.5330 | 0.0080 |
| `school_notebooks_clean` | 2000 | 379 | 377 | 1244 | 0.1366 | 0.1367 | 0.0001 | 0.4437 | 0.4512 | 0.0075 | 0.5590 | 0.5575 | -0.0015 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 58 | 51 | 384 | 0.1623 | 0.1528 | -0.0095 |
| `4-6` | 1538 | 297 | 297 | 944 | 0.1416 | 0.1416 | 0.0001 |
| `7-10` | 1767 | 476 | 418 | 873 | 0.1454 | 0.1407 | -0.0047 |
| `11+` | 1765 | 494 | 481 | 790 | 0.1200 | 0.1181 | -0.0019 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0070, 0.0072]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 379 | 377 | 1244 | 0.1366 | 0.1367 | 0.0001 | 0.5590 | 0.5575 | -0.0015 |
