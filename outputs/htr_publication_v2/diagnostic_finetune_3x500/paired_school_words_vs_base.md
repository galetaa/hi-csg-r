# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1076 | 1271 | 3216 | 0.1476 | 0.1554 | 0.0078 | [0.0041, 0.0114] | 0.5214 | 0.5304 | 0.0090 | 0.4341 | 0.4246 | -0.0095 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 361 | 459 | 743 | 0.1990 | 0.2110 | 0.0120 | 0.6898 | 0.6992 | 0.0094 | 0.2674 | 0.2610 | -0.0064 |
| `hkr_words` | 2000 | 349 | 440 | 1211 | 0.0919 | 0.1034 | 0.0115 | 0.3895 | 0.4133 | 0.0238 | 0.5085 | 0.4865 | -0.0220 |
| `school_notebooks_clean` | 2000 | 366 | 372 | 1262 | 0.1631 | 0.1639 | 0.0008 | 0.5218 | 0.5155 | -0.0063 | 0.4900 | 0.4905 | 0.0005 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 55 | 62 | 376 | 0.1653 | 0.1711 | 0.0057 |
| `4-6` | 1538 | 242 | 294 | 1002 | 0.1577 | 0.1665 | 0.0088 |
| `7-10` | 1767 | 373 | 465 | 929 | 0.1581 | 0.1686 | 0.0105 |
| `11+` | 1765 | 406 | 450 | 909 | 0.1233 | 0.1280 | 0.0047 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0064, 0.0077]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 1764 | 308 | 316 | 1140 | 0.1552 | 0.1551 | -0.0001 | 0.5034 | 0.5000 | -0.0034 |
| `hard_real` | 236 | 58 | 56 | 122 | 0.2219 | 0.2291 | 0.0072 | 0.3898 | 0.4195 | 0.0297 |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
