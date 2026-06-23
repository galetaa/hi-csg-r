# Paired HTR Comparison - Iteration 2

Common samples: 5563

## Overall

| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5563 | 1416 | 1199 | 2948 | 0.1453 | 0.1360 | -0.0093 | [-0.0132, -0.0056] | 0.5134 | 0.4907 | -0.0227 | 0.4411 | 0.4605 | 0.0194 |

## By Dataset

| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 475 | 443 | 645 | 0.1932 | 0.1879 | -0.0053 | 0.6733 | 0.6570 | -0.0163 | 0.2815 | 0.2866 | 0.0051 |
| `hkr_words` | 2000 | 467 | 420 | 1113 | 0.0956 | 0.0910 | -0.0046 | 0.4000 | 0.3865 | -0.0135 | 0.5010 | 0.5145 | 0.0135 |
| `school_notebooks_clean` | 2000 | 474 | 336 | 1190 | 0.1575 | 0.1403 | -0.0171 | 0.5018 | 0.4650 | -0.0368 | 0.5060 | 0.5425 | 0.0365 |

## By Text Length

| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 63 | 60 | 370 | 0.1494 | 0.1494 | 0.0000 |
| `4-6` | 1538 | 338 | 275 | 925 | 0.1576 | 0.1436 | -0.0140 |
| `7-10` | 1767 | 500 | 405 | 862 | 0.1578 | 0.1467 | -0.0111 |
| `11+` | 1765 | 515 | 459 | 791 | 0.1208 | 0.1148 | -0.0060 |

## School Quality Buckets

School CER delta bootstrap CI95: [-0.0244, -0.0097]

| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 1764 | 406 | 288 | 1070 | 0.1477 | 0.1312 | -0.0165 | 0.5193 | 0.5629 | 0.0437 |
| `hard_real` | 236 | 68 | 48 | 120 | 0.2307 | 0.2088 | -0.0219 | 0.4068 | 0.3898 | -0.0169 |
| `invalid_or_review` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `unknown` | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
