# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1421 | 1143 | 2999 | 0.1494 | 0.1365 | -0.0129 | [-0.0170, -0.0088] | 0.5147 | 0.4888 | -0.0259 | 0.4354 | 0.4634 | 0.0280 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 484 | 415 | 664 | 0.1997 | 0.1898 | -0.0099 | 0.6610 | 0.6581 | -0.0029 | 0.2777 | 0.2949 | 0.0173 |
| `hkr_words` | 2000 | 454 | 430 | 1116 | 0.0960 | 0.0939 | -0.0021 | 0.4033 | 0.3942 | -0.0092 | 0.4990 | 0.5080 | 0.0090 |
| `school_notebooks_clean` | 2000 | 483 | 298 | 1219 | 0.1634 | 0.1374 | -0.0261 | 0.5118 | 0.4512 | -0.0605 | 0.4950 | 0.5505 | 0.0555 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 68 | 53 | 372 | 0.1758 | 0.1508 | -0.0250 |
| `4-6` | 1538 | 353 | 247 | 938 | 0.1576 | 0.1391 | -0.0186 |
| `7-10` | 1767 | 486 | 397 | 884 | 0.1584 | 0.1470 | -0.0115 |
| `11+` | 1765 | 514 | 446 | 805 | 0.1257 | 0.1198 | -0.0060 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0339, -0.0184]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 483 | 298 | 1219 | 0.1634 | 0.1374 | -0.0261 | 0.4950 | 0.5505 | 0.0555 |
