# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1509 | 1158 | 2896 | 0.1461 | 0.1353 | -0.0108 | [-0.0147, -0.0067] | 0.5155 | 0.4929 | -0.0226 | 0.4383 | 0.4629 | 0.0246 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 494 | 401 | 668 | 0.1946 | 0.1864 | -0.0082 | 0.6771 | 0.6670 | -0.0101 | 0.2790 | 0.2853 | 0.0064 |
| `hkr_words` | 2000 | 496 | 404 | 1100 | 0.0961 | 0.0916 | -0.0046 | 0.4044 | 0.3786 | -0.0258 | 0.4940 | 0.5255 | 0.0315 |
| `school_notebooks_clean` | 2000 | 519 | 353 | 1128 | 0.1582 | 0.1391 | -0.0190 | 0.5005 | 0.4713 | -0.0292 | 0.5070 | 0.5390 | 0.0320 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 56 | 72 | 365 | 0.1528 | 0.1670 | 0.0142 |
| `4-6` | 1538 | 360 | 286 | 892 | 0.1597 | 0.1467 | -0.0130 |
| `7-10` | 1767 | 545 | 378 | 844 | 0.1564 | 0.1402 | -0.0162 |
| `11+` | 1765 | 548 | 422 | 795 | 0.1221 | 0.1116 | -0.0104 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0267, -0.0114]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 519 | 353 | 1128 | 0.1582 | 0.1391 | -0.0190 | 0.5070 | 0.5390 | 0.0320 |
