# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 940 | 1028 | 2032 | 0.1253 | 0.1330 | 0.0077 | [0.0031, 0.0126] | 0.4214 | 0.4365 | 0.0150 | 0.4510 | 0.4373 | -0.0137 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 542 | 646 | 812 | 0.1165 | 0.1252 | 0.0087 | 0.4124 | 0.4370 | 0.0246 | 0.3295 | 0.3080 | -0.0215 |
| `school_notebooks_clean` | 2000 | 398 | 382 | 1220 | 0.1342 | 0.1408 | 0.0066 | 0.4305 | 0.4360 | 0.0055 | 0.5725 | 0.5665 | -0.0060 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 46 | 83 | 306 | 0.1548 | 0.1962 | 0.0414 |
| `4-6` | 849 | 161 | 176 | 512 | 0.1390 | 0.1471 | 0.0081 |
| `7-10` | 1006 | 233 | 219 | 554 | 0.1274 | 0.1272 | -0.0002 |
| `11+` | 1710 | 500 | 550 | 660 | 0.1099 | 0.1135 | 0.0036 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0017, 0.0149]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 398 | 382 | 1220 | 0.1342 | 0.1408 | 0.0066 | 0.5725 | 0.5665 | -0.0060 |
