# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1297 | 1304 | 2962 | 0.1365 | 0.1372 | 0.0007 | [-0.0033, 0.0046] | 0.4888 | 0.4935 | 0.0047 | 0.4634 | 0.4584 | -0.0050 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 446 | 477 | 640 | 0.1898 | 0.1917 | 0.0019 | 0.6581 | 0.6598 | 0.0017 | 0.2949 | 0.2898 | -0.0051 |
| `hkr_words` | 2000 | 450 | 449 | 1101 | 0.0939 | 0.0955 | 0.0016 | 0.3942 | 0.4087 | 0.0145 | 0.5080 | 0.4900 | -0.0180 |
| `school_notebooks_clean` | 2000 | 401 | 378 | 1221 | 0.1374 | 0.1363 | -0.0011 | 0.4512 | 0.4485 | -0.0027 | 0.5505 | 0.5585 | 0.0080 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 55 | 76 | 362 | 0.1508 | 0.1599 | 0.0091 |
| `4-6` | 1538 | 287 | 330 | 921 | 0.1391 | 0.1470 | 0.0079 |
| `7-10` | 1767 | 464 | 437 | 866 | 0.1470 | 0.1424 | -0.0045 |
| `11+` | 1765 | 491 | 461 | 813 | 0.1198 | 0.1171 | -0.0027 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0087, 0.0066]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 401 | 378 | 1221 | 0.1374 | 0.1363 | -0.0011 | 0.5505 | 0.5585 | 0.0080 |
