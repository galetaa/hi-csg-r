# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1267 | 1300 | 2996 | 0.1372 | 0.1372 | 0.0000 | [-0.0038, 0.0037] | 0.4880 | 0.4935 | 0.0056 | 0.4636 | 0.4584 | -0.0052 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 431 | 451 | 681 | 0.1882 | 0.1917 | 0.0035 | 0.6571 | 0.6598 | 0.0028 | 0.2975 | 0.2898 | -0.0077 |
| `hkr_words` | 2000 | 399 | 501 | 1100 | 0.0881 | 0.0955 | 0.0075 | 0.3670 | 0.4087 | 0.0417 | 0.5315 | 0.4900 | -0.0415 |
| `school_notebooks_clean` | 2000 | 437 | 348 | 1215 | 0.1464 | 0.1363 | -0.0101 | 0.4768 | 0.4485 | -0.0282 | 0.5255 | 0.5585 | 0.0330 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 65 | 61 | 367 | 0.1717 | 0.1599 | -0.0118 |
| `4-6` | 1538 | 294 | 329 | 915 | 0.1410 | 0.1470 | 0.0060 |
| `7-10` | 1767 | 447 | 413 | 907 | 0.1451 | 0.1424 | -0.0026 |
| `11+` | 1765 | 461 | 497 | 807 | 0.1163 | 0.1171 | 0.0008 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0178, -0.0028]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 437 | 348 | 1215 | 0.1464 | 0.1363 | -0.0101 | 0.5255 | 0.5585 | 0.0330 |
