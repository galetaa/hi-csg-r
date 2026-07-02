# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1157 | 832 | 2011 | 0.1400 | 0.1265 | -0.0135 | [-0.0181, -0.0089] | 0.4562 | 0.4193 | -0.0369 | 0.4130 | 0.4552 | 0.0423 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 678 | 509 | 813 | 0.1287 | 0.1193 | -0.0094 | 0.4434 | 0.4131 | -0.0303 | 0.2900 | 0.3305 | 0.0405 |
| `school_notebooks_clean` | 2000 | 479 | 323 | 1198 | 0.1513 | 0.1336 | -0.0177 | 0.4690 | 0.4255 | -0.0435 | 0.5360 | 0.5800 | 0.0440 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 65 | 55 | 315 | 0.1816 | 0.1709 | -0.0107 |
| `4-6` | 849 | 180 | 156 | 513 | 0.1511 | 0.1440 | -0.0071 |
| `7-10` | 1006 | 288 | 194 | 524 | 0.1408 | 0.1209 | -0.0199 |
| `11+` | 1710 | 624 | 427 | 659 | 0.1234 | 0.1097 | -0.0137 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0255, -0.0097]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 479 | 323 | 1198 | 0.1513 | 0.1336 | -0.0177 | 0.5360 | 0.5800 | 0.0440 |
