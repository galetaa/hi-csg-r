# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1419 | 1168 | 2976 | 0.1461 | 0.1354 | -0.0107 | [-0.0145, -0.0067] | 0.5155 | 0.4864 | -0.0292 | 0.4383 | 0.4665 | 0.0282 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 473 | 450 | 640 | 0.1946 | 0.1887 | -0.0059 | 0.6771 | 0.6711 | -0.0060 | 0.2790 | 0.2834 | 0.0045 |
| `hkr_words` | 2000 | 471 | 396 | 1133 | 0.0961 | 0.0920 | -0.0042 | 0.4044 | 0.3781 | -0.0262 | 0.4940 | 0.5240 | 0.0300 |
| `school_notebooks_clean` | 2000 | 475 | 322 | 1203 | 0.1582 | 0.1372 | -0.0210 | 0.5005 | 0.4502 | -0.0502 | 0.5070 | 0.5520 | 0.0450 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 64 | 49 | 380 | 0.1528 | 0.1396 | -0.0132 |
| `4-6` | 1538 | 356 | 275 | 907 | 0.1597 | 0.1424 | -0.0173 |
| `7-10` | 1767 | 482 | 431 | 854 | 0.1564 | 0.1491 | -0.0073 |
| `11+` | 1765 | 517 | 413 | 835 | 0.1221 | 0.1145 | -0.0076 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0285, -0.0134]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 475 | 322 | 1203 | 0.1582 | 0.1372 | -0.0210 | 0.5070 | 0.5520 | 0.0450 |
