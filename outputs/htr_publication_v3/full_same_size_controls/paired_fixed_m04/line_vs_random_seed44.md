# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1285 | 1256 | 3022 | 0.1365 | 0.1348 | -0.0016 | [-0.0054, 0.0023] | 0.4913 | 0.4828 | -0.0085 | 0.4634 | 0.4710 | 0.0075 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 434 | 482 | 647 | 0.1872 | 0.1930 | 0.0057 | 0.6604 | 0.6713 | 0.0109 | 0.2885 | 0.2809 | -0.0077 |
| `hkr_words` | 2000 | 425 | 428 | 1147 | 0.0870 | 0.0875 | 0.0005 | 0.3685 | 0.3670 | -0.0015 | 0.5380 | 0.5330 | -0.0050 |
| `school_notebooks_clean` | 2000 | 426 | 346 | 1228 | 0.1463 | 0.1367 | -0.0096 | 0.4820 | 0.4512 | -0.0307 | 0.5255 | 0.5575 | 0.0320 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 50 | 56 | 387 | 0.1542 | 0.1528 | -0.0014 |
| `4-6` | 1538 | 298 | 288 | 952 | 0.1415 | 0.1416 | 0.0001 |
| `7-10` | 1767 | 496 | 417 | 854 | 0.1497 | 0.1407 | -0.0090 |
| `11+` | 1765 | 441 | 495 | 829 | 0.1139 | 0.1181 | 0.0041 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0168, -0.0023]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 426 | 346 | 1228 | 0.1463 | 0.1367 | -0.0096 | 0.5255 | 0.5575 | 0.0320 |
