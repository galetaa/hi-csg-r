# Graph Fusion vs Image-only +10k Context

Common samples: 5563

## Paired Bootstrap

| scope | n | mean per-sample ΔCER | 95% CI | aggregate ΔCER | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| `overall` | 5563 | 0.0036 | [-0.0003, 0.0076] | 0.0066 | 1228 | 1372 | 2963 |
| `hkr_words` | 2000 | 0.0123 | [0.0070, 0.0176] | 0.0081 | 407 | 527 | 1066 |
| `cyrillic_handwriting` | 1563 | 0.0101 | [0.0023, 0.0181] | 0.0162 | 405 | 492 | 666 |
| `school_notebooks_clean` | 2000 | -0.0100 | [-0.0172, -0.0027] | -0.0080 | 416 | 353 | 1231 |

## By Dataset

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 0.0839 | 0.0920 | 0.0081 | 0.3473 | 0.3830 | 0.5245 | 0.4755 | 407 | 527 | 1066 |
| `cyrillic_handwriting` | 1563 | 0.1836 | 0.1997 | 0.0162 | 0.6474 | 0.6625 | 0.2892 | 0.2821 | 405 | 492 | 666 |
| `school_notebooks_clean` | 2000 | 0.1333 | 0.1253 | -0.0080 | 0.4732 | 0.4355 | 0.5390 | 0.5730 | 416 | 353 | 1231 |

## School Quality

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 1764 | 0.1233 | 0.1178 | -0.0055 | 0.4553 | 0.4188 | 0.5573 | 0.5907 | 345 | 308 | 1111 |
| `hard_real` | 236 | 0.2057 | 0.1794 | -0.0262 | 0.6076 | 0.5612 | 0.4025 | 0.4407 | 71 | 45 | 120 |
| `invalid_or_review` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 |
| `unknown` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 |

## By Text Length

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 0.1663 | 0.1494 | -0.0169 | 0.3151 | 0.2661 | 0.6815 | 0.7323 | 75 | 56 | 362 |
| `4-6` | 1538 | 0.1450 | 0.1469 | 0.0019 | 0.4926 | 0.4939 | 0.5143 | 0.5085 | 309 | 324 | 905 |
| `7-10` | 1767 | 0.1395 | 0.1456 | 0.0060 | 0.5255 | 0.5477 | 0.4335 | 0.4126 | 416 | 477 | 874 |
| `11+` | 1765 | 0.1117 | 0.1215 | 0.0098 | 0.4457 | 0.4611 | 0.3887 | 0.3773 | 428 | 515 | 822 |

## By Token Type

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `alpha` | 3729 | 0.1214 | 0.1259 | 0.0045 | 0.4645 | 0.4848 | 0.4647 | 0.4430 | 834 | 949 | 1946 |
| `mixed` | 1316 | 0.1359 | 0.1449 | 0.0090 | 0.5096 | 0.5086 | 0.3845 | 0.3974 | 316 | 350 | 650 |
| `numeric` | 25 | 0.2103 | 0.4483 | 0.2379 | 0.8125 | 0.9375 | 0.1600 | 0.0800 | 3 | 17 | 5 |
| `punctuation` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 |
| `short_1_3` | 493 | 0.1663 | 0.1494 | -0.0169 | 0.3151 | 0.2661 | 0.6815 | 0.7323 | 75 | 56 | 362 |

## By Graph Valid

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `True` | 5563 | 0.1272 | 0.1338 | 0.0066 | 0.4687 | 0.4786 | 0.4636 | 0.4562 | 1228 | 1372 | 2963 |
| `False` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 |
