# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1100 | 838 | 2062 | 0.1332 | 0.1217 | -0.0115 | [-0.0159, -0.0070] | 0.4403 | 0.4123 | -0.0280 | 0.4325 | 0.4640 | 0.0315 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 624 | 553 | 823 | 0.1226 | 0.1192 | -0.0033 | 0.4226 | 0.4230 | 0.0004 | 0.3170 | 0.3240 | 0.0070 |
| `school_notebooks_clean` | 2000 | 476 | 285 | 1239 | 0.1439 | 0.1242 | -0.0196 | 0.4580 | 0.4015 | -0.0565 | 0.5480 | 0.6040 | 0.0560 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 54 | 53 | 328 | 0.1628 | 0.1598 | -0.0031 |
| `4-6` | 849 | 197 | 133 | 519 | 0.1519 | 0.1315 | -0.0204 |
| `7-10` | 1006 | 287 | 173 | 546 | 0.1332 | 0.1150 | -0.0182 |
| `11+` | 1710 | 562 | 479 | 669 | 0.1164 | 0.1111 | -0.0053 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0273, -0.0119]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 476 | 285 | 1239 | 0.1439 | 0.1242 | -0.0196 | 0.5480 | 0.6040 | 0.0560 |
