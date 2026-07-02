# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1566 | 647 | 1787 | 0.1620 | 0.1217 | -0.0403 | [-0.0450, -0.0357] | 0.5112 | 0.4123 | -0.0990 | 0.3573 | 0.4640 | 0.1068 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 965 | 376 | 659 | 0.1601 | 0.1192 | -0.0409 | 0.5245 | 0.4230 | -0.1015 | 0.2070 | 0.3240 | 0.1170 |
| `school_notebooks_clean` | 2000 | 601 | 271 | 1128 | 0.1639 | 0.1242 | -0.0396 | 0.4980 | 0.4015 | -0.0965 | 0.5075 | 0.6040 | 0.0965 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 74 | 49 | 312 | 0.1854 | 0.1598 | -0.0257 |
| `4-6` | 849 | 254 | 122 | 473 | 0.1782 | 0.1315 | -0.0467 |
| `7-10` | 1006 | 395 | 154 | 457 | 0.1625 | 0.1150 | -0.0475 |
| `11+` | 1710 | 843 | 322 | 545 | 0.1476 | 0.1111 | -0.0365 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0482, -0.0316]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 601 | 271 | 1128 | 0.1639 | 0.1242 | -0.0396 | 0.5075 | 0.6040 | 0.0965 |
