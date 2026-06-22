# Natural Line Validation - Iteration 2

## Overall

| metric | value |
|---|---:|
| `n` | 120 |
| `completed` | 120 |
| `valid_line_rate` | 0.983 |
| `correct_order_rate` | 1.000 |
| `missing_words_rate` | 0.617 |
| `neighbor_noise_rate` | 0.000 |
| `good_for_train_aug_rate` | 1.000 |
| `accepted` | True |

## By Stratum

| stratum | n | valid | order | missing | noise | aug | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_clean_core` | 40 | 1.000 | 1.000 | 0.575 | 0.000 | 1.000 | True |
| `has_hard_real` | 30 | 0.967 | 1.000 | 0.500 | 0.000 | 1.000 | True |
| `random` | 20 | 1.000 | 1.000 | 0.650 | 0.000 | 1.000 | True |
| `short_group` | 30 | 0.967 | 1.000 | 0.767 | 0.000 | 1.000 | True |
