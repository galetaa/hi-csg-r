# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1078 | 948 | 1974 | 0.1400 | 0.1367 | -0.0033 | [-0.0081, 0.0014] | 0.4562 | 0.4467 | -0.0095 | 0.4130 | 0.4253 | 0.0123 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 652 | 534 | 814 | 0.1287 | 0.1229 | -0.0058 | 0.4434 | 0.4240 | -0.0194 | 0.2900 | 0.3110 | 0.0210 |
| `school_notebooks_clean` | 2000 | 426 | 414 | 1160 | 0.1513 | 0.1504 | -0.0009 | 0.4690 | 0.4695 | 0.0005 | 0.5360 | 0.5395 | 0.0035 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 57 | 67 | 311 | 0.1816 | 0.1958 | 0.0142 |
| `4-6` | 849 | 176 | 180 | 493 | 0.1511 | 0.1499 | -0.0013 |
| `7-10` | 1006 | 253 | 237 | 516 | 0.1408 | 0.1355 | -0.0053 |
| `11+` | 1710 | 592 | 464 | 654 | 0.1234 | 0.1157 | -0.0077 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0091, 0.0075]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 426 | 414 | 1160 | 0.1513 | 0.1504 | -0.0009 | 0.5360 | 0.5395 | 0.0035 |
