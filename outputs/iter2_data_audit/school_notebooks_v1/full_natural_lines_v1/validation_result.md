# Full Natural Line Validation - Iteration 2

## Overall

| metric | value |
|---|---:|
| `n` | 150 |
| `completed` | 150 |
| `valid_line_rate` | 0.967 |
| `correct_order_rate` | 0.993 |
| `complete_enough_rate` | 0.833 |
| `neighbor_noise_rate` | 0.053 |
| `good_for_line_train_rate` | 0.993 |
| `accepted` | False |

## By Stratum

| stratum | n | valid | order | complete | noise | train | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `groups_2_words` | 30 | 1.000 | 1.000 | 0.967 | 0.000 | 1.000 | True |
| `groups_3_words` | 30 | 1.000 | 0.967 | 0.867 | 0.000 | 1.000 | False |
| `groups_4plus_words` | 40 | 0.975 | 1.000 | 1.000 | 0.000 | 0.975 | True |
| `groups_8plus_words` | 25 | 1.000 | 1.000 | 1.000 | 0.120 | 1.000 | False |
| `large_x_gap` | 25 | 0.840 | 1.000 | 0.200 | 0.200 | 1.000 | False |
