# Graph Fusion vs Baseline Image-only

Common samples: 5563

## Paired Bootstrap

| scope | n | mean per-sample ΔCER | 95% CI | aggregate ΔCER | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| `overall` | 5563 | -0.0065 | [-0.0104, -0.0028] | -0.0053 | 1389 | 1211 | 2963 |
| `hkr_words` | 2000 | 0.0084 | [0.0035, 0.0133] | 0.0028 | 437 | 477 | 1086 |
| `cyrillic_handwriting` | 1563 | 0.0027 | [-0.0047, 0.0103] | 0.0059 | 434 | 455 | 674 |
| `school_notebooks_clean` | 2000 | -0.0286 | [-0.0361, -0.0214] | -0.0337 | 518 | 279 | 1203 |

## By Dataset

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkr_words` | 2000 | 0.0892 | 0.0920 | 0.0028 | 0.3641 | 0.3830 | 0.5010 | 0.4755 | 437 | 477 | 1086 |
| `cyrillic_handwriting` | 1563 | 0.1939 | 0.1997 | 0.0059 | 0.6625 | 0.6625 | 0.2815 | 0.2821 | 434 | 455 | 674 |
| `school_notebooks_clean` | 2000 | 0.1590 | 0.1253 | -0.0337 | 0.5045 | 0.4355 | 0.5060 | 0.5730 | 518 | 279 | 1203 |

## School Quality

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_core` | 1764 | 0.1493 | 0.1178 | -0.0315 | 0.4924 | 0.4188 | 0.5193 | 0.5907 | 444 | 237 | 1083 |
| `hard_real` | 236 | 0.2291 | 0.1794 | -0.0497 | 0.5949 | 0.5612 | 0.4068 | 0.4407 | 74 | 42 | 120 |
| `invalid_or_review` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 |
| `unknown` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 |

## By Text Length

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1-3` | 493 | 0.1510 | 0.1494 | -0.0016 | 0.2838 | 0.2661 | 0.7099 | 0.7323 | 56 | 53 | 384 |
| `4-6` | 1538 | 0.1553 | 0.1469 | -0.0084 | 0.5003 | 0.4939 | 0.5026 | 0.5085 | 335 | 306 | 897 |
| `7-10` | 1767 | 0.1583 | 0.1456 | -0.0128 | 0.5747 | 0.5477 | 0.3831 | 0.4126 | 499 | 393 | 875 |
| `11+` | 1765 | 0.1213 | 0.1215 | 0.0002 | 0.4617 | 0.4611 | 0.3705 | 0.3773 | 499 | 459 | 807 |

## By Token Type

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `alpha` | 3729 | 0.1305 | 0.1259 | -0.0046 | 0.4807 | 0.4848 | 0.4401 | 0.4430 | 933 | 868 | 1928 |
| `mixed` | 1316 | 0.1566 | 0.1449 | -0.0117 | 0.5516 | 0.5086 | 0.3488 | 0.3974 | 396 | 275 | 645 |
| `numeric` | 25 | 0.2448 | 0.4483 | 0.2034 | 0.8125 | 0.9375 | 0.1600 | 0.0800 | 4 | 15 | 6 |
| `punctuation` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 |
| `short_1_3` | 493 | 0.1510 | 0.1494 | -0.0016 | 0.2838 | 0.2661 | 0.7099 | 0.7323 | 56 | 53 | 384 |

## By Graph Valid

| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `True` | 5563 | 0.1391 | 0.1338 | -0.0053 | 0.4889 | 0.4786 | 0.4411 | 0.4562 | 1389 | 1211 | 2963 |
| `False` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 |
