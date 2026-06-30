# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1205 | 1037 | 3321 | 0.1487 | 0.1437 | -0.0050 | [-0.0084, -0.0016] | 0.5188 | 0.5176 | -0.0012 | 0.4343 | 0.4418 | 0.0075 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 389 | 380 | 794 | 0.1984 | 0.2020 | 0.0036 | 0.6835 | 0.6991 | 0.0156 | 0.2783 | 0.2700 | -0.0083 |
| `hkr_words` | 2000 | 383 | 367 | 1250 | 0.0931 | 0.0925 | -0.0006 | 0.3964 | 0.3918 | -0.0046 | 0.4970 | 0.5015 | 0.0045 |
| `school_notebooks_clean` | 2000 | 433 | 290 | 1277 | 0.1655 | 0.1494 | -0.0162 | 0.5125 | 0.5015 | -0.0110 | 0.4935 | 0.5165 | 0.0230 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 48 | 44 | 401 | 0.1575 | 0.1538 | -0.0037 |
| `4-6` | 1538 | 287 | 249 | 1002 | 0.1561 | 0.1494 | -0.0067 |
| `7-10` | 1767 | 438 | 353 | 976 | 0.1632 | 0.1554 | -0.0079 |
| `11+` | 1765 | 432 | 391 | 942 | 0.1253 | 0.1243 | -0.0011 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0226, -0.0095]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 1764 | 373 | 242 | 1149 | 0.1534 | 0.1370 | -0.0164 | 0.5102 | 0.5374 | 0.0272 |
| `hard_real` | 236 | 60 | 48 | 128 | 0.2560 | 0.2420 | -0.0141 | 0.3686 | 0.3602 | -0.0085 |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
