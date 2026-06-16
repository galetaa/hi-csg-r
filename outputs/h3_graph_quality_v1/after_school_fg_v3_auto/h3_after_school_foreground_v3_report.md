# H3 after school foreground v3

## 1. Input

- manifest: `data/experiments/htr_graph_v1/graph_ready/tri10k_mixed_school_fg_v3_auto/test.jsonl`
- predictions: `outputs/robustness_v1/image_only/clean/predictions.jsonl`
- joined n: 5563
- feature n: 39

## 2. Dataset summary

| dataset | n | mean CER | nonzero CER rate |
|---|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 0.1203 | 0.5342 |
| `hkr_words` | 2000 | 0.0582 | 0.3455 |
| `school_notebooks_clean` | 2000 | 0.0765 | 0.2815 |

## 3. Best correlation

`graph_endpoint_count`: Spearman r=0.0981, abs=0.0981, n=5563

## 4. Best high-error detection by feature set

| feature set | group | n | ROC-AUC | PR-AUC | lift | top20 precision |
|---|---|---:|---:|---:|---:|---:|
| `all_features_no_text_len` | `hkr_words|word|unknown` | 1090 | 0.6544 | 0.3278 | 1.5466 | 0.3532 |
| `all_non_geometry` | `hkr_words|word|unknown` | 1090 | 0.6723 | 0.3532 | 1.6665 | 0.3853 |
| `geometry_control` | `dataset:hkr_words` | 2000 | 0.5726 | 0.2378 | 1.1541 | 0.2475 |
| `quality_only` | `cyrillic_handwriting|word|unknown` | 1100 | 0.5021 | 0.2180 | 1.0034 | 0.1773 |
| `structural_core` | `hkr_words|word|unknown` | 1090 | 0.6723 | 0.3532 | 1.6666 | 0.3853 |

## 5. Top high-error results

| rank | feature set | group | n | ROC-AUC | PR-AUC | lift | top20 precision |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `structural_core` | `hkr_words|word|unknown` | 1090 | 0.6723 | 0.3532 | 1.6666 | 0.3853 |
| 2 | `all_non_geometry` | `hkr_words|word|unknown` | 1090 | 0.6723 | 0.3532 | 1.6665 | 0.3853 |
| 3 | `all_features_no_text_len` | `hkr_words|word|unknown` | 1090 | 0.6544 | 0.3278 | 1.5466 | 0.3532 |
| 4 | `all_non_geometry` | `dataset:hkr_words` | 2000 | 0.6471 | 0.3224 | 1.5652 | 0.3675 |
| 5 | `structural_core` | `dataset:hkr_words` | 2000 | 0.6471 | 0.3224 | 1.5652 | 0.3675 |
| 6 | `all_features_no_text_len` | `dataset:hkr_words` | 2000 | 0.6455 | 0.3208 | 1.5570 | 0.3550 |
| 7 | `structural_core` | `dataset:cyrillic_handwriting` | 1563 | 0.6208 | 0.3394 | 1.4222 | 0.3546 |
| 8 | `all_non_geometry` | `dataset:cyrillic_handwriting` | 1563 | 0.6208 | 0.3394 | 1.4222 | 0.3546 |
| 9 | `all_features_no_text_len` | `global` | 5563 | 0.6160 | 0.2944 | 1.4394 | 0.3037 |
| 10 | `all_features_no_text_len` | `dataset:cyrillic_handwriting` | 1563 | 0.6140 | 0.3384 | 1.4181 | 0.3419 |