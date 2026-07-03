# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1084 | 880 | 2036 | 0.1367 | 0.1265 | -0.0102 | [-0.0149, -0.0054] | 0.4467 | 0.4193 | -0.0274 | 0.4253 | 0.4552 | 0.0300 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 602 | 533 | 865 | 0.1229 | 0.1193 | -0.0036 | 0.4240 | 0.4131 | -0.0109 | 0.3110 | 0.3305 | 0.0195 |
| `school_notebooks_clean` | 2000 | 482 | 347 | 1171 | 0.1504 | 0.1336 | -0.0168 | 0.4695 | 0.4255 | -0.0440 | 0.5395 | 0.5800 | 0.0405 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 82 | 56 | 297 | 0.1958 | 0.1709 | -0.0249 |
| `4-6` | 849 | 185 | 164 | 500 | 0.1499 | 0.1440 | -0.0059 |
| `7-10` | 1006 | 272 | 201 | 533 | 0.1355 | 0.1209 | -0.0146 |
| `11+` | 1710 | 545 | 459 | 706 | 0.1157 | 0.1097 | -0.0060 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0253, -0.0088]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 482 | 347 | 1171 | 0.1504 | 0.1336 | -0.0168 | 0.5395 | 0.5800 | 0.0405 |
