# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1439 | 1130 | 2994 | 0.1494 | 0.1372 | -0.0122 | [-0.0160, -0.0085] | 0.5147 | 0.4880 | -0.0267 | 0.4354 | 0.4636 | 0.0282 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 499 | 418 | 646 | 0.1997 | 0.1882 | -0.0115 | 0.6610 | 0.6571 | -0.0039 | 0.2777 | 0.2975 | 0.0198 |
| `hkr_words` | 2000 | 491 | 387 | 1122 | 0.0960 | 0.0881 | -0.0079 | 0.4033 | 0.3670 | -0.0363 | 0.4990 | 0.5315 | 0.0325 |
| `school_notebooks_clean` | 2000 | 449 | 325 | 1226 | 0.1634 | 0.1464 | -0.0170 | 0.5118 | 0.4768 | -0.0350 | 0.4950 | 0.5255 | 0.0305 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 57 | 59 | 377 | 0.1758 | 0.1717 | -0.0041 |
| `4-6` | 1538 | 354 | 253 | 931 | 0.1576 | 0.1410 | -0.0166 |
| `7-10` | 1767 | 504 | 403 | 860 | 0.1584 | 0.1451 | -0.0134 |
| `11+` | 1765 | 524 | 415 | 826 | 0.1257 | 0.1163 | -0.0095 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0243, -0.0096]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 449 | 325 | 1226 | 0.1634 | 0.1464 | -0.0170 | 0.4950 | 0.5255 | 0.0305 |
