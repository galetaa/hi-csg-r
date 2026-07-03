# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 980 | 942 | 2078 | 0.1217 | 0.1217 | 0.0001 | [-0.0045, 0.0046] | 0.4117 | 0.4123 | 0.0005 | 0.4627 | 0.4640 | 0.0013 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 602 | 586 | 812 | 0.1202 | 0.1192 | -0.0010 | 0.4194 | 0.4230 | 0.0036 | 0.3270 | 0.3240 | -0.0030 |
| `school_notebooks_clean` | 2000 | 378 | 356 | 1266 | 0.1231 | 0.1242 | 0.0011 | 0.4040 | 0.4015 | -0.0025 | 0.5985 | 0.6040 | 0.0055 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 47 | 60 | 328 | 0.1429 | 0.1598 | 0.0169 |
| `4-6` | 849 | 148 | 166 | 535 | 0.1273 | 0.1315 | 0.0042 |
| `7-10` | 1006 | 263 | 205 | 538 | 0.1245 | 0.1150 | -0.0095 |
| `11+` | 1710 | 522 | 511 | 677 | 0.1118 | 0.1111 | -0.0006 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0073, 0.0094]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 378 | 356 | 1266 | 0.1231 | 0.1242 | 0.0011 | 0.5985 | 0.6040 | 0.0055 |
