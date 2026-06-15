# H2 gold audit candidate pool — v1

## 1. Purpose

This candidate pool is designed for manual structural audit of visible-stroke graphs. It samples across CER and graph-structural-risk quadrants.

## 2. Inputs

```text
manifest: data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/test.jsonl
predictions: outputs/robustness_v1/image_only/clean/predictions.jsonl
joined samples: 5563
structural features: 30
grouping mode: dataset_level
per cell target: 25
```

## 3. Selected candidates by cell

| cell | n | mean CER | mean risk |
|---|---:|---:|---:|
| `A_highCER_highRisk` | 25 | 0.3938 | 0.8988 |
| `B_highCER_lowRisk` | 25 | 0.3738 | 0.1537 |
| `C_lowCER_highRisk` | 25 | 0.0000 | 0.8597 |
| `D_lowCER_lowRisk` | 25 | 0.0000 | 0.0974 |

## 4. Selected candidates by dataset

| dataset | n |
|---|---:|
| `cyrillic_handwriting` | 37 |
| `hkr_words` | 40 |
| `school_notebooks_clean` | 23 |

## 5. Manual annotation fields

Use `annotation_template.csv`. Fill these fields:

- `ink_visible_ok`: 0/1
- `skeleton_follows_ink`: 0/1
- `missed_visible_stroke`: 0/1
- `spurious_stroke`: 0/1
- `endpoint_error`: 0/1
- `junction_error`: 0/1
- `loop_error`: 0/1
- `critical_topology_error`: 0/1
- `graph_quality_0_3`: 0=bad, 1=weak, 2=usable, 3=good
- `notes`: short free-text note

## 6. Strict use

Do not use this pool to estimate population-level graph quality. It is deliberately biased toward informative cases. Use it to build the H2 rubric, failure taxonomy, and later a balanced gold subset.