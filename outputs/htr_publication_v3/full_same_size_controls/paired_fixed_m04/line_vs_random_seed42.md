# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1258 | 1253 | 3052 | 0.1339 | 0.1353 | 0.0014 | [-0.0024, 0.0053] | 0.4890 | 0.4929 | 0.0039 | 0.4661 | 0.4629 | -0.0032 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 433 | 461 | 669 | 0.1808 | 0.1864 | 0.0056 | 0.6551 | 0.6670 | 0.0119 | 0.2994 | 0.2853 | -0.0141 |
| `hkr_words` | 2000 | 412 | 431 | 1157 | 0.0895 | 0.0916 | 0.0021 | 0.3755 | 0.3786 | 0.0031 | 0.5255 | 0.5255 | 0.0000 |
| `school_notebooks_clean` | 2000 | 413 | 361 | 1226 | 0.1417 | 0.1391 | -0.0025 | 0.4728 | 0.4713 | -0.0015 | 0.5370 | 0.5390 | 0.0020 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 50 | 71 | 372 | 0.1515 | 0.1670 | 0.0156 |
| `4-6` | 1538 | 284 | 328 | 926 | 0.1391 | 0.1467 | 0.0076 |
| `7-10` | 1767 | 476 | 387 | 904 | 0.1475 | 0.1402 | -0.0073 |
| `11+` | 1765 | 448 | 467 | 850 | 0.1108 | 0.1116 | 0.0008 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0101, 0.0047]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 413 | 361 | 1226 | 0.1417 | 0.1391 | -0.0025 | 0.5370 | 0.5390 | 0.0020 |
