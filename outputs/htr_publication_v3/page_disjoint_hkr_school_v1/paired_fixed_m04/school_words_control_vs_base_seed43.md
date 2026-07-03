# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1509 | 628 | 1863 | 0.1620 | 0.1217 | -0.0403 | [-0.0452, -0.0355] | 0.5112 | 0.4117 | -0.0995 | 0.3573 | 0.4627 | 0.1055 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 932 | 364 | 704 | 0.1601 | 0.1202 | -0.0399 | 0.5245 | 0.4194 | -0.1051 | 0.2070 | 0.3270 | 0.1200 |
| `school_notebooks_clean` | 2000 | 577 | 264 | 1159 | 0.1639 | 0.1231 | -0.0407 | 0.4980 | 0.4040 | -0.0940 | 0.5075 | 0.5985 | 0.0910 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 81 | 44 | 310 | 0.1854 | 0.1429 | -0.0425 |
| `4-6` | 849 | 258 | 110 | 481 | 0.1782 | 0.1273 | -0.0509 |
| `7-10` | 1006 | 358 | 162 | 486 | 0.1625 | 0.1245 | -0.0380 |
| `11+` | 1710 | 812 | 312 | 586 | 0.1476 | 0.1118 | -0.0358 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0491, -0.0323]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 577 | 264 | 1159 | 0.1639 | 0.1231 | -0.0407 | 0.5075 | 0.5985 | 0.0910 |
