# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 883 | 1042 | 2075 | 0.1211 | 0.1330 | 0.0119 | [0.0073, 0.0168] | 0.4126 | 0.4365 | 0.0238 | 0.4585 | 0.4373 | -0.0212 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 531 | 625 | 844 | 0.1182 | 0.1252 | 0.0070 | 0.4085 | 0.4370 | 0.0284 | 0.3280 | 0.3080 | -0.0200 |
| `school_notebooks_clean` | 2000 | 352 | 417 | 1231 | 0.1241 | 0.1408 | 0.0168 | 0.4168 | 0.4360 | 0.0192 | 0.5890 | 0.5665 | -0.0225 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 41 | 79 | 315 | 0.1464 | 0.1962 | 0.0498 |
| `4-6` | 849 | 134 | 199 | 516 | 0.1254 | 0.1471 | 0.0217 |
| `7-10` | 1006 | 228 | 231 | 547 | 0.1249 | 0.1272 | 0.0023 |
| `11+` | 1710 | 480 | 533 | 697 | 0.1104 | 0.1135 | 0.0031 |

## School Quality Buckets

School CER delta bootstrap CI95: [0.0084, 0.0252]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 352 | 417 | 1231 | 0.1241 | 0.1408 | 0.0168 | 0.5890 | 0.5665 | -0.0225 |
