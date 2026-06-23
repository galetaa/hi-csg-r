# Operating Point Stratification

Rows are test samples for `plus_10k_context` with `confidence_graph` risk.

## strict

| stratum type | stratum | n | accepted | coverage | accepted CER | accepted exact | mean risk |
|---|---|---:|---:|---:|---:|---:|---:|
| `dataset` | `cyrillic_handwriting` | 1563 | 187 | 0.1196 | 0.0802 | 0.6684 | 0.6367 |
| `dataset` | `hkr_words` | 2000 | 308 | 0.1540 | 0.0419 | 0.8019 | 0.5389 |
| `dataset` | `school_notebooks_clean` | 2000 | 781 | 0.3905 | 0.0491 | 0.8143 | 0.4607 |
| `short_flag` | `not_short_1_3` | 5070 | 964 | 0.1901 | 0.0494 | 0.7635 | 0.5568 |
| `short_flag` | `short_1_3` | 493 | 312 | 0.6329 | 0.0598 | 0.8718 | 0.3484 |
| `text_len` | `1-3` | 493 | 312 | 0.6329 | 0.0598 | 0.8718 | 0.3484 |
| `text_len` | `11+` | 1765 | 104 | 0.0589 | 0.0208 | 0.8077 | 0.6369 |
| `text_len` | `4-6` | 1538 | 539 | 0.3505 | 0.0618 | 0.7570 | 0.4642 |
| `text_len` | `7-10` | 1767 | 321 | 0.1817 | 0.0378 | 0.7601 | 0.5573 |
| `token_type` | `alpha` | 4113 | 985 | 0.2395 | 0.0466 | 0.8152 | 0.5273 |
| `token_type` | `mixed` | 1413 | 282 | 0.1996 | 0.0712 | 0.6986 | 0.5677 |
| `token_type` | `numeric` | 37 | 9 | 0.2432 | 0.0370 | 0.8889 | 0.6380 |

## balanced

| stratum type | stratum | n | accepted | coverage | accepted CER | accepted exact | mean risk |
|---|---|---:|---:|---:|---:|---:|---:|
| `dataset` | `cyrillic_handwriting` | 1563 | 629 | 0.4024 | 0.0984 | 0.5199 | 0.6367 |
| `dataset` | `hkr_words` | 2000 | 1151 | 0.5755 | 0.0499 | 0.7159 | 0.5389 |
| `dataset` | `school_notebooks_clean` | 2000 | 1323 | 0.6615 | 0.0685 | 0.7249 | 0.4607 |
| `short_flag` | `not_short_1_3` | 5070 | 2724 | 0.5373 | 0.0659 | 0.6590 | 0.5568 |
| `short_flag` | `short_1_3` | 493 | 379 | 0.7688 | 0.0800 | 0.8311 | 0.3484 |
| `text_len` | `1-3` | 493 | 379 | 0.7688 | 0.0800 | 0.8311 | 0.3484 |
| `text_len` | `11+` | 1765 | 721 | 0.4085 | 0.0413 | 0.6602 | 0.6369 |
| `text_len` | `4-6` | 1538 | 1028 | 0.6684 | 0.0870 | 0.6644 | 0.4642 |
| `text_len` | `7-10` | 1767 | 975 | 0.5518 | 0.0620 | 0.6523 | 0.5573 |
| `token_type` | `alpha` | 4113 | 2373 | 0.5770 | 0.0633 | 0.7033 | 0.5273 |
| `token_type` | `mixed` | 1413 | 718 | 0.5081 | 0.0821 | 0.6031 | 0.5677 |
| `token_type` | `numeric` | 37 | 12 | 0.3243 | 0.0687 | 0.6667 | 0.6380 |

## broad

| stratum type | stratum | n | accepted | coverage | accepted CER | accepted exact | mean risk |
|---|---|---:|---:|---:|---:|---:|---:|
| `dataset` | `cyrillic_handwriting` | 1563 | 1000 | 0.6398 | 0.1232 | 0.4090 | 0.6367 |
| `dataset` | `hkr_words` | 2000 | 1655 | 0.8275 | 0.0658 | 0.6060 | 0.5389 |
| `dataset` | `school_notebooks_clean` | 2000 | 1559 | 0.7795 | 0.0845 | 0.6652 | 0.4607 |
| `short_flag` | `not_short_1_3` | 5070 | 3807 | 0.7509 | 0.0850 | 0.5587 | 0.5568 |
| `short_flag` | `short_1_3` | 493 | 407 | 0.8256 | 0.0987 | 0.7912 | 0.3484 |
| `text_len` | `1-3` | 493 | 407 | 0.8256 | 0.0987 | 0.7912 | 0.3484 |
| `text_len` | `11+` | 1765 | 1244 | 0.7048 | 0.0636 | 0.5161 | 0.6369 |
| `text_len` | `4-6` | 1538 | 1244 | 0.8088 | 0.1044 | 0.6053 | 0.4642 |
| `text_len` | `7-10` | 1767 | 1319 | 0.7465 | 0.0870 | 0.5550 | 0.5573 |
| `token_type` | `alpha` | 4113 | 3170 | 0.7707 | 0.0842 | 0.6025 | 0.5273 |
| `token_type` | `mixed` | 1413 | 1024 | 0.7247 | 0.0923 | 0.5166 | 0.5677 |
| `token_type` | `numeric` | 37 | 20 | 0.5405 | 0.1182 | 0.5000 | 0.6380 |
