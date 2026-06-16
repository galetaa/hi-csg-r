# Graph feature cross-evaluation

## Overall

| checkpoint | feature manifest | CER | WER | exact |
|---|---|---:|---:|---:|
| `graph-v2` | `old` | 0.13970 | 0.49042 | 0.43897 |
| `graph-v2` | `school-fg-v3` | 0.13943 | 0.48975 | 0.43969 |
| `graph-v3` | `old` | 0.15396 | 0.52364 | 0.40626 |
| `graph-v3` | `school-fg-v3` | 0.15338 | 0.52190 | 0.40823 |

## By dataset

| dataset | old model + old fg | old model + v3 fg | new model + old fg | new model + v3 fg |
|---|---:|---:|---:|---:|
| `hkr_words` | 0.08803 | 0.08803 | 0.10185 | 0.10185 |
| `cyrillic_handwriting` | 0.19968 | 0.19968 | 0.21758 | 0.21758 |
| `school_notebooks_clean` | 0.15635 | 0.15525 | 0.16681 | 0.16445 |

## Controlled contrasts

| contrast | ΔCER | interpretation |
|---|---:|---|
| v3 features on old checkpoint | -0.00027 | Inference-time feature-distribution effect. |
| v3 features on new checkpoint | -0.00058 | Whether the new model benefits from its matching v3 features. |
| new vs old checkpoint on old features | +0.01426 | Training-run/model difference with a shared old manifest. |
| new vs old checkpoint on v3 features | +0.01395 | Training-run/model difference with a shared v3 manifest. |

## Decision rules

- If the new checkpoint is worse with both manifests, the main issue is the new training run or optimization, not foreground v3 alone.
- If each checkpoint works best only with its matching manifest, the feature distribution changed and the graph branch is sensitive to that shift.
- If v3 features hurt both checkpoints, visual graph repair does not translate into useful graph-fusion features.
- If v3 features improve school-notebooks but hurt the other datasets, global normalization or shared fusion is causing cross-domain interference.
