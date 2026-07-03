# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1372 | 729 | 1899 | 0.1620 | 0.1332 | -0.0288 | [-0.0335, -0.0242] | 0.5112 | 0.4403 | -0.0710 | 0.3573 | 0.4325 | 0.0752 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 896 | 395 | 709 | 0.1601 | 0.1226 | -0.0375 | 0.5245 | 0.4226 | -0.1019 | 0.2070 | 0.3170 | 0.1100 |
| `school_notebooks_clean` | 2000 | 476 | 334 | 1190 | 0.1639 | 0.1439 | -0.0200 | 0.4980 | 0.4580 | -0.0400 | 0.5075 | 0.5480 | 0.0405 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 74 | 49 | 312 | 0.1854 | 0.1628 | -0.0226 |
| `4-6` | 849 | 210 | 141 | 498 | 0.1782 | 0.1519 | -0.0264 |
| `7-10` | 1006 | 324 | 187 | 495 | 0.1625 | 0.1332 | -0.0292 |
| `11+` | 1710 | 764 | 352 | 594 | 0.1476 | 0.1164 | -0.0312 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0277, -0.0120]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 476 | 334 | 1190 | 0.1639 | 0.1439 | -0.0200 | 0.5075 | 0.5480 | 0.0405 |
