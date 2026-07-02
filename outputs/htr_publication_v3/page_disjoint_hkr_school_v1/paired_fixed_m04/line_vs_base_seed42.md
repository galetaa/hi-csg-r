# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1088 | 891 | 2021 | 0.1429 | 0.1330 | -0.0099 | [-0.0147, -0.0051] | 0.4617 | 0.4365 | -0.0253 | 0.4135 | 0.4373 | 0.0238 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 617 | 565 | 818 | 0.1269 | 0.1252 | -0.0017 | 0.4410 | 0.4370 | -0.0040 | 0.3025 | 0.3080 | 0.0055 |
| `school_notebooks_clean` | 2000 | 471 | 326 | 1203 | 0.1589 | 0.1408 | -0.0180 | 0.4825 | 0.4360 | -0.0465 | 0.5245 | 0.5665 | 0.0420 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 74 | 64 | 297 | 0.2172 | 0.1962 | -0.0211 |
| `4-6` | 849 | 205 | 146 | 498 | 0.1613 | 0.1471 | -0.0142 |
| `7-10` | 1006 | 255 | 198 | 553 | 0.1375 | 0.1272 | -0.0103 |
| `11+` | 1710 | 554 | 483 | 673 | 0.1180 | 0.1135 | -0.0045 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0263, -0.0098]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 471 | 326 | 1203 | 0.1589 | 0.1408 | -0.0180 | 0.5245 | 0.5665 | 0.0420 |
