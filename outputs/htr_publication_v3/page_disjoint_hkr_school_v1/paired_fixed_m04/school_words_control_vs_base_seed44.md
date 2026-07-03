# Paired HTR Comparison - Iteration 2

Common samples: 4000

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1186 | 773 | 2041 | 0.1400 | 0.1202 | -0.0198 | [-0.0242, -0.0153] | 0.4562 | 0.4086 | -0.0476 | 0.4130 | 0.4600 | 0.0470 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 709 | 492 | 799 | 0.1287 | 0.1149 | -0.0138 | 0.4434 | 0.4078 | -0.0356 | 0.2900 | 0.3260 | 0.0360 |
| `school_notebooks_clean` | 2000 | 477 | 281 | 1242 | 0.1513 | 0.1255 | -0.0258 | 0.4690 | 0.4095 | -0.0595 | 0.5360 | 0.5940 | 0.0580 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 435 | 69 | 37 | 329 | 0.1816 | 0.1464 | -0.0352 |
| `4-6` | 849 | 192 | 146 | 511 | 0.1511 | 0.1374 | -0.0138 |
| `7-10` | 1006 | 276 | 173 | 557 | 0.1408 | 0.1176 | -0.0232 |
| `11+` | 1710 | 649 | 417 | 644 | 0.1234 | 0.1066 | -0.0169 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0336, -0.0179]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 477 | 281 | 1242 | 0.1513 | 0.1255 | -0.0258 | 0.5360 | 0.5940 | 0.0580 |
