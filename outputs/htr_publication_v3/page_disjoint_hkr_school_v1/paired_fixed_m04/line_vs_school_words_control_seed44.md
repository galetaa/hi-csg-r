# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 908 | 960 | 2132 | 0.1202 | 0.1265 | 0.0063 | [0.0019, 0.0106] | 0.4086 | 0.4193 | 0.0107 | 0.4600 | 0.4552 | -0.0048 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 545 | 581 | 874 | 0.1149 | 0.1193 | 0.0044 | 0.4078 | 0.4131 | 0.0053 | 0.3260 | 0.3305 | 0.0045 |
| `school_notebooks_clean` | 2000 | 363 | 379 | 1258 | 0.1255 | 0.1336 | 0.0081 | 0.4095 | 0.4255 | 0.0160 | 0.5940 | 0.5800 | -0.0140 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 46 | 65 | 324 | 0.1464 | 0.1709 | 0.0245 |
| `4-6` | 849 | 155 | 171 | 523 | 0.1374 | 0.1440 | 0.0067 |
| `7-10` | 1006 | 222 | 220 | 564 | 0.1176 | 0.1209 | 0.0033 |
| `11+` | 1710 | 485 | 504 | 721 | 0.1066 | 0.1097 | 0.0032 |

## School Quality Buckets

School CER delta bootstrap CI95: [0.0004, 0.0158]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 363 | 379 | 1258 | 0.1255 | 0.1336 | 0.0081 | 0.5940 | 0.5800 | -0.0140 |
