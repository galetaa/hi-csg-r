# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1474 | 1167 | 2922 | 0.1494 | 0.1372 | -0.0122 | [-0.0163, -0.0081] | 0.5147 | 0.4935 | -0.0212 | 0.4354 | 0.4584 | 0.0230 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 504 | 427 | 632 | 0.1997 | 0.1917 | -0.0080 | 0.6610 | 0.6598 | -0.0012 | 0.2777 | 0.2898 | 0.0122 |
| `hkr_words` | 2000 | 459 | 454 | 1087 | 0.0960 | 0.0955 | -0.0005 | 0.4033 | 0.4087 | 0.0053 | 0.4990 | 0.4900 | -0.0090 |
| `school_notebooks_clean` | 2000 | 511 | 286 | 1203 | 0.1634 | 0.1363 | -0.0272 | 0.5118 | 0.4485 | -0.0633 | 0.4950 | 0.5585 | 0.0635 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 61 | 59 | 373 | 0.1758 | 0.1599 | -0.0159 |
| `4-6` | 1538 | 355 | 291 | 892 | 0.1576 | 0.1470 | -0.0106 |
| `7-10` | 1767 | 518 | 392 | 857 | 0.1584 | 0.1424 | -0.0160 |
| `11+` | 1765 | 540 | 425 | 800 | 0.1257 | 0.1171 | -0.0087 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0351, -0.0195]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `hard_real` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 2000 | 511 | 286 | 1203 | 0.1634 | 0.1363 | -0.0272 | 0.4950 | 0.5585 | 0.0635 |
