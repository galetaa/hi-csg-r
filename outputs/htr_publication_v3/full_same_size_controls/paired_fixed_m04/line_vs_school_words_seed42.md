# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1311 | 1226 | 3026 | 0.1354 | 0.1353 | -0.0001 | [-0.0040, 0.0039] | 0.4864 | 0.4929 | 0.0066 | 0.4665 | 0.4629 | -0.0036 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 479 | 431 | 653 | 0.1887 | 0.1864 | -0.0024 | 0.6711 | 0.6670 | -0.0041 | 0.2834 | 0.2853 | 0.0019 |
| `hkr_words` | 2000 | 436 | 427 | 1137 | 0.0920 | 0.0916 | -0.0004 | 0.3781 | 0.3786 | 0.0005 | 0.5240 | 0.5255 | 0.0015 |
| `school_notebooks_clean` | 2000 | 396 | 368 | 1236 | 0.1372 | 0.1391 | 0.0019 | 0.4502 | 0.4713 | 0.0210 | 0.5520 | 0.5390 | -0.0130 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 39 | 68 | 386 | 0.1396 | 0.1670 | 0.0274 |
| `4-6` | 1538 | 289 | 314 | 935 | 0.1424 | 0.1467 | 0.0043 |
| `7-10` | 1767 | 504 | 385 | 878 | 0.1491 | 0.1402 | -0.0089 |
| `11+` | 1765 | 479 | 459 | 827 | 0.1145 | 0.1116 | -0.0028 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0055, 0.0095]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 396 | 368 | 1236 | 0.1372 | 0.1391 | 0.0019 | 0.5520 | 0.5390 | -0.0130 |
