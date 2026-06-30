# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1430 | 1091 | 3042 | 0.1461 | 0.1339 | -0.0122 | [-0.0160, -0.0084] | 0.5155 | 0.4890 | -0.0265 | 0.4383 | 0.4661 | 0.0279 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 482 | 390 | 691 | 0.1946 | 0.1808 | -0.0138 | 0.6771 | 0.6551 | -0.0219 | 0.2790 | 0.2994 | 0.0205 |
| `hkr_words` | 2000 | 493 | 375 | 1132 | 0.0961 | 0.0895 | -0.0067 | 0.4044 | 0.3755 | -0.0289 | 0.4940 | 0.5255 | 0.0315 |
| `school_notebooks_clean` | 2000 | 455 | 326 | 1219 | 0.1582 | 0.1417 | -0.0165 | 0.5005 | 0.4728 | -0.0277 | 0.5070 | 0.5370 | 0.0300 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 62 | 56 | 375 | 0.1528 | 0.1515 | -0.0014 |
| `4-6` | 1538 | 362 | 242 | 934 | 0.1597 | 0.1391 | -0.0206 |
| `7-10` | 1767 | 476 | 400 | 891 | 0.1564 | 0.1475 | -0.0089 |
| `11+` | 1765 | 530 | 393 | 842 | 0.1221 | 0.1108 | -0.0112 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0240, -0.0091]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 455 | 326 | 1219 | 0.1582 | 0.1417 | -0.0165 | 0.5070 | 0.5370 | 0.0300 |
