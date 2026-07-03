# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1129 | 843 | 2028 | 0.1429 | 0.1253 | -0.0175 | [-0.0224, -0.0128] | 0.4617 | 0.4214 | -0.0403 | 0.4135 | 0.4510 | 0.0375 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 658 | 511 | 831 | 0.1269 | 0.1165 | -0.0104 | 0.4410 | 0.4124 | -0.0286 | 0.3025 | 0.3295 | 0.0270 |
| `school_notebooks_clean` | 2000 | 471 | 332 | 1197 | 0.1589 | 0.1342 | -0.0247 | 0.4825 | 0.4305 | -0.0520 | 0.5245 | 0.5725 | 0.0480 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 92 | 47 | 296 | 0.2172 | 0.1548 | -0.0625 |
| `4-6` | 849 | 207 | 134 | 508 | 0.1613 | 0.1390 | -0.0223 |
| `7-10` | 1006 | 255 | 199 | 552 | 0.1375 | 0.1274 | -0.0102 |
| `11+` | 1710 | 575 | 463 | 672 | 0.1180 | 0.1099 | -0.0081 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0332, -0.0159]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 471 | 332 | 1197 | 0.1589 | 0.1342 | -0.0247 | 0.5245 | 0.5725 | 0.0480 |
