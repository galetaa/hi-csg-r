# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1087 | 1142 | 3334 | 0.1476 | 0.1487 | 0.0011 | [-0.0023, 0.0045] | 0.5214 | 0.5188 | -0.0026 | 0.4341 | 0.4343 | 0.0002 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 383 | 390 | 790 | 0.1990 | 0.1984 | -0.0006 | 0.6898 | 0.6835 | -0.0063 | 0.2674 | 0.2783 | 0.0109 |
| `hkr_words` | 2000 | 363 | 389 | 1248 | 0.0919 | 0.0931 | 0.0012 | 0.3895 | 0.3964 | 0.0069 | 0.5085 | 0.4970 | -0.0115 |
| `school_notebooks_clean` | 2000 | 341 | 363 | 1296 | 0.1631 | 0.1655 | 0.0024 | 0.5218 | 0.5125 | -0.0093 | 0.4900 | 0.4935 | 0.0035 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 50 | 41 | 402 | 0.1653 | 0.1575 | -0.0078 |
| `4-6` | 1538 | 257 | 248 | 1033 | 0.1577 | 0.1561 | -0.0016 |
| `7-10` | 1767 | 374 | 415 | 978 | 0.1581 | 0.1632 | 0.0052 |
| `11+` | 1765 | 406 | 438 | 921 | 0.1233 | 0.1253 | 0.0020 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0043, 0.0092]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 1764 | 304 | 297 | 1163 | 0.1552 | 0.1534 | -0.0018 | 0.5034 | 0.5102 | 0.0068 |
| `hard_real` | 236 | 37 | 66 | 133 | 0.2219 | 0.2560 | 0.0341 | 0.3898 | 0.3686 | -0.0212 |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
