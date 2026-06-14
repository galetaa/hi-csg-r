# Mixed Cyrillic image-only baselines report — Stage 3.3

## 1. Purpose

This report compares single-dataset Cyrillic HTR baselines with mixed-dataset image-only baselines. The goal is to determine whether a universal Cyrillic CRNN-CTC model improves cross-domain recognition before graph-aware experiments.

## 2. Mixed runs

| run | training composition | selected penalty |
|---|---|---:|
| Mixed Cyrillic balanced50k v1 | Balanced training: 50k samples from each Cyrillic dataset. | -0.2 |
| Mixed Cyrillic natural-full v1 | Natural full training: all available train samples from each Cyrillic dataset. | -0.4 |

## 3. Test CER comparison

| dataset | single full CER | mixed balanced50k CER | mixed natural-full CER | natural-full vs single |
|---|---:|---:|---:|---:|
| Cyrillic Handwriting | 0.1405 | 0.1281 | 0.1208 | 14.0% |
| HKR Words | 0.1525 | 0.0691 | 0.0623 | 59.2% |
| School Notebooks Clean | 0.0838 | 0.1002 | 0.0744 | 11.1% |

## 4. Full per-dataset metrics for mixed natural-full

| dataset | split | n | CER | WER | exact | pred_len | blank | penalty | epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cyrillic Handwriting | val | 7232 | 0.0723 | 0.2939 | 0.6962 | 7.42 | 0.835 | -0.4 | 46 |
| Cyrillic Handwriting | test | 1563 | 0.1208 | 0.4936 | 0.4562 | 9.40 | 0.842 | -0.4 | 46 |
| HKR Words | val | 6494 | 0.0565 | 0.2555 | 0.6729 | 10.25 | 0.852 | -0.4 | 46 |
| HKR Words | test | 6495 | 0.0623 | 0.2867 | 0.6411 | 10.43 | 0.860 | -0.4 | 46 |
| School Notebooks Clean | val | 24214 | 0.0432 | 0.1793 | 0.8252 | 5.93 | 0.840 | -0.4 | 46 |
| School Notebooks Clean | test | 24623 | 0.0744 | 0.2742 | 0.7327 | 5.85 | 0.824 | -0.4 | 46 |

## 5. Interpretation

Mixed Cyrillic natural-full improves all three Cyrillic test sets relative to their single-dataset full baselines. The largest gain is on HKR Words, suggesting that additional Cyrillic domains provide useful regularization and character-shape coverage.

The balanced50k run already improves Cyrillic Handwriting and HKR Words but hurts School Notebooks. The natural-full run fixes this by restoring the full School Notebooks training mass while preserving the gains on the other datasets.

Therefore, `mixed_cyrillic_natural_full_v1` should be treated as the primary Cyrillic image-only baseline before graph-aware experiments.

## 6. Stage 3.3 conclusion

```text
[x] mixed balanced50k baseline
[x] mixed natural-full baseline
[x] per-dataset validation/test evaluation
[x] universal Cyrillic image-only baseline selected
primary baseline: mixed_cyrillic_natural_full_v1
```

## 7. Next stage

Next: Stage 4 graph-aware experiments. Start with a lightweight graph-feature fusion baseline before moving to full graph neural models.
