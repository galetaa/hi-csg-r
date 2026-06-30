# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1281 | 964 | 3318 | 0.1554 | 0.1437 | -0.0117 | [-0.0151, -0.0083] | 0.5304 | 0.5176 | -0.0128 | 0.4246 | 0.4418 | 0.0173 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 451 | 356 | 756 | 0.2110 | 0.2020 | -0.0090 | 0.6992 | 0.6991 | -0.0001 | 0.2610 | 0.2700 | 0.0090 |
| `hkr_words` | 2000 | 428 | 317 | 1255 | 0.1034 | 0.0925 | -0.0109 | 0.4133 | 0.3918 | -0.0215 | 0.4865 | 0.5015 | 0.0150 |
| `school_notebooks_clean` | 2000 | 402 | 291 | 1307 | 0.1639 | 0.1494 | -0.0145 | 0.5155 | 0.5015 | -0.0140 | 0.4905 | 0.5165 | 0.0260 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 57 | 39 | 397 | 0.1711 | 0.1538 | -0.0172 |
| `4-6` | 1538 | 314 | 204 | 1020 | 0.1665 | 0.1494 | -0.0171 |
| `7-10` | 1767 | 471 | 332 | 964 | 0.1686 | 0.1554 | -0.0132 |
| `11+` | 1765 | 439 | 389 | 937 | 0.1280 | 0.1243 | -0.0038 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0215, -0.0077]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 1764 | 361 | 227 | 1176 | 0.1551 | 0.1370 | -0.0181 | 0.5000 | 0.5374 | 0.0374 |
| `hard_real` | 236 | 41 | 64 | 131 | 0.2291 | 0.2420 | 0.0129 | 0.4195 | 0.3602 | -0.0593 |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
