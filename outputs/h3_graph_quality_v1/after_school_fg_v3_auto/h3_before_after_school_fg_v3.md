# H3 before/after school foreground v3

## 1. Feature-set comparison

| feature set | old group | old ROC | new group | new ROC | ΔROC | old PR | new PR | ΔPR | old top20 | new top20 | Δtop20 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `quality_only` | `n/a` | n/a | `cyrillic_handwriting|word|unknown` | 0.5021 | n/a | n/a | 0.2180 | n/a | n/a | 0.1773 | n/a |
| `geometry_control` | `n/a` | n/a | `dataset:hkr_words` | 0.5726 | n/a | n/a | 0.2378 | n/a | n/a | 0.2475 | n/a |
| `structural_core` | `hkr_words|word|unknown` | 0.6733 | `hkr_words|word|unknown` | 0.6723 | -0.0010 | 0.3466 | 0.3532 | 0.0066 | 0.3761 | 0.3853 | 0.0092 |
| `all_non_geometry` | `n/a` | n/a | `hkr_words|word|unknown` | 0.6723 | n/a | n/a | 0.3532 | n/a | n/a | 0.3853 | n/a |
| `all_features_no_text_len` | `n/a` | n/a | `hkr_words|word|unknown` | 0.6544 | n/a | n/a | 0.3278 | n/a | n/a | 0.3532 | n/a |

## 2. Interpretation rule

If H3 improves mainly for school-notebooks groups, foreground v3 repaired the diagnostic graph signal for that subset. If global H3 changes little, that is acceptable: this was a preprocessing repair, not a model retraining step.