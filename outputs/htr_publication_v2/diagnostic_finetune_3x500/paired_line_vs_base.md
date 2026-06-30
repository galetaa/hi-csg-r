# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1143 | 1018 | 3402 | 0.1476 | 0.1437 | -0.0039 | [-0.0071, -0.0006] | 0.5214 | 0.5176 | -0.0038 | 0.4341 | 0.4418 | 0.0077 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 376 | 395 | 792 | 0.1990 | 0.2020 | 0.0030 | 0.6898 | 0.6991 | 0.0094 | 0.2674 | 0.2700 | 0.0026 |
| `hkr_words` | 2000 | 349 | 347 | 1304 | 0.0919 | 0.0925 | 0.0006 | 0.3895 | 0.3918 | 0.0023 | 0.5085 | 0.5015 | -0.0070 |
| `school_notebooks_clean` | 2000 | 418 | 276 | 1306 | 0.1631 | 0.1494 | -0.0137 | 0.5218 | 0.5015 | -0.0203 | 0.4900 | 0.5165 | 0.0265 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 60 | 43 | 390 | 0.1653 | 0.1538 | -0.0115 |
| `4-6` | 1538 | 281 | 222 | 1035 | 0.1577 | 0.1494 | -0.0083 |
| `7-10` | 1767 | 400 | 354 | 1013 | 0.1581 | 0.1554 | -0.0027 |
| `11+` | 1765 | 402 | 399 | 964 | 0.1233 | 0.1243 | 0.0009 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0201, -0.0072]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 1764 | 371 | 217 | 1176 | 0.1552 | 0.1370 | -0.0182 | 0.5034 | 0.5374 | 0.0340 |
| `hard_real` | 236 | 47 | 59 | 130 | 0.2219 | 0.2420 | 0.0201 | 0.3898 | 0.3602 | -0.0297 |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
