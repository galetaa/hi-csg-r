# CRNN-CTC HI-CSG-R Adapter Protocol v1

Status: frozen before feature construction and model training.

Any scientific configuration change after the first full seed-42 run requires a new
protocol version. Bug fixes must be recorded in checkpoint metadata and the experiment
registry.

## Scope

The experiment adds one x-aligned HI-CSG-R residual adapter to the existing CRNN-CTC.
The CNN, visual projection, BiLSTM, classifier, CTC vocabulary, preprocessing, splits,
and canonical `+10k` training corpus remain the basis of the recognizer.

Excluded work includes GNN/Graph Transformer recognizers, cross-attention, a second
inference decoder, autoregressive decoding, learned graph extraction, new datasets,
domain identifiers, and test-driven architecture selection.

## Frozen Inputs

Canonical manifests:

```text
train: data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/train.jsonl
val:   data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/val.jsonl
test:  data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/test.jsonl
vocab: data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/vocab.json
```

Expected sample counts are `39998` train, `6000` validation, and `5563` mixed
test. Any older file under `data/experiments/htr_adapter_v1/manifests` without
`xaligned_graph_npz` or with a different count is a placeholder and is not an
experimental input. The feature builder must regenerate the enhanced manifests
from the canonical paths above.

Enhanced manifests created by the v1 builder:

```text
train:          data/experiments/htr_adapter_v1/manifests/train.jsonl
val:            data/experiments/htr_adapter_v1/manifests/val.jsonl
mixed test:     data/experiments/htr_adapter_v1/manifests/test.jsonl
Cyrillic test:  data/experiments/htr_adapter_v1/manifests/test_cyrillic.jsonl
HKR test:       data/experiments/htr_adapter_v1/manifests/test_hkr.jsonl
School test:    data/experiments/htr_adapter_v1/manifests/test_school.jsonl
```

Frozen additional source manifests:

```text
page-disjoint HKR+School:
  data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1/test.jsonl
School clean_core:
  data/experiments/iter2_quality_manifests/school_notebooks_lineaware_v3/test.clean_core.jsonl
School hard_real:
  data/experiments/iter2_quality_manifests/school_notebooks_lineaware_v3/test.hard_real.jsonl
robustness directory (15 distorted manifests):
  data/experiments/htr_graph_v1/robustness_v2_recomputed/manifests
```

Every additional source manifest is passed through the same x-aligned builder.
In particular, each robustness graph is reconstructed from its distorted image.

Canonical image-only `+10k` checkpoints:

```text
seed 42: outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1/best.pt
seed 43: outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1_seed43/best.pt
seed 44: outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1_seed44/best.pt
```

Checkpoint SHA256 values at protocol freeze:

```text
seed 42: dab5192fee3a0b575401fa15d3f66826beb19438dba870670b3f3369bedc66bf
seed 43: d1c90f5f3fad307d9a54339eb239fc7f9ed0fb4b0484da6021602cf00c57188a
seed 44: 3be0676d169d99b1cb817b99aa8f0a8ee43b35881a03770e39a4244df3fb8410
```

Seeds are matched: every adapter or matched-fine-tuning run starts from the canonical
checkpoint with the same seed.

The executable experiment registry and analysis workflow is:

```text
notebooks/htr_adapter_v1_full_experiment.ipynb
```

It is generated reproducibly by
`tools/build_hi_csg_r_adapter_experiment_notebook_v1.py`. The notebook executes one
explicit `RUN_STAGE` at a time: `prepare`, `smoke`, `seed42`, `final_seeds`,
`final_test`, or `report`; `check` is the fast validation-only mode. The default
stage is `prepare`. Final test cells additionally require a successful validation
gate, a frozen checkpoint registry, and explicit one-time authorization.

## Fixed Evaluation

- Primary metric: corpus micro-CER.
- Secondary metrics: sample macro-CER, WER, exact match, and domain CER.
- Checkpoint selection: validation micro-CER only.
- Fixed train/evaluation blank logit penalty: `-0.4`.
- Final seeds: `42`, `43`, `44`.
- Test data may be evaluated only after the seed-42 validation gate and checkpoint freeze.

## X-Aligned Features

Feature version: `hi_csg_r_xaligned_v1`.

The current image is binarized using the existing dataset-specific policy, skeletonized,
and converted with the existing `hi_csg_r_v1` pixel-graph extractor. Robustness images
must be processed independently; a clean graph may not be reused.

The image width is divided into `T = max(width // 4, 1)` bins. Nodes are assigned by
normalized x coordinate. Every polyline segment is assigned by its midpoint and weighted
by segment length. The only smoothing is the fixed `[0.25, 0.50, 0.25]` filter with
boundary renormalization.

Frozen feature order:

1. `ink_fraction`
2. `skeleton_density`
3. `edge_length_density`
4. `stroke_width_mean`
5. `stroke_width_std`
6. `curvature_mean`
7. `orientation_horizontal`
8. `orientation_vertical`
9. `orientation_diag_pos`
10. `orientation_diag_neg`
11. `node_density`
12. `endpoint_density`
13. `junction_density`
14. `loop_edge_fraction`
15. `component_count_norm`
16. `short_branch_fraction`
17. `boundary_crossings_norm`
18. `ambiguous_edge_fraction`
19. `graph_occupancy`
20. `warning_density`

The gate quality vector contains normalized features 18-20 only. No dataset ID, target
length, target transcription, split, error-derived category, or test calibration enters
the adapter or gate.

Normalizer statistics are fitted on the x-aligned train manifest only:

```text
z = clip((x - train_mean) / max(train_std, 1e-6), -5, 5)
```

For `M2`, normalized features 11-20 and the quality vector are zeroed. The architecture
and parameter count remain identical to `M3`.

## Model

Fusion point: after the existing 256-dimensional visual projection and before the
existing BiLSTM.

Temporal adapter:

```text
LayerNorm(20)
Conv1d(20, 64, 3, padding=1) + GELU + Dropout(0.10)
Conv1d(64, 128, 3, padding=1) + GELU + Dropout(0.10)
Linear(128, 256) + LayerNorm(256)
```

The final graph projection has zero weight and bias. Fusion follows:

```text
F = LayerNorm(V + aG)
```

For the exact zero-initialized state only, the implementation bypasses the fusion
LayerNorm and returns `V`; this is the narrow initial-equivalence condition. After graph
warm-up makes `G` non-zero, the fixed formula above is used.

Gate:

```text
concat(V, G, Q) -> Linear(515,64) -> GELU -> Linear(64,1) -> sigmoid
```

Final gate bias: `-1.5`.

Auxiliary training-only head:

```text
G -> Linear(256, vocabulary size) -> CTC
```

Joint loss:

```text
L = L_fused_ctc + 0.15 * L_graph_aux_ctc
```

Adapter, gate, fusion normalization, and auxiliary head must add no more than 400,000
parameters.

## Training

Warm-up, 5 epochs:

- train graph adapter and auxiliary head only;
- loss is auxiliary graph CTC;
- learning rates are `3e-4`.

Joint fine-tuning, 25 epochs:

```text
graph adapter: 3e-4
gate:          3e-4
aux head:      3e-4
BiLSTM:        5e-5
classifier:    5e-5
last CNN block:1e-5
```

Other settings:

```text
optimizer: AdamW
weight decay: 1e-4
batch size: 16
gradient clip: 5.0
blank penalty: -0.4
```

`M0-FT` uses the same 25 joint epochs, batches, shared-layer learning rates, unfreeze
policy, penalty, and validation selection rule. It has no graph modules.

## Models

- `M0`: existing canonical image-only `+10k`.
- `M0-FT`: matched partial fine-tuning, seeds 42/43/44.
- `M1`: historical global-vector fusion, no retraining.
- `M2`: topology-off x-aligned adapter, seed 42.
- `M3`: full x-aligned adapter, seeds 42/43/44 if validation gate passes.
- `M3-shuffle`: inference-only domain/width/ink-matched graph replacement.

## Seed-42 Validation Gate

Continue to seeds 43/44 only when all conditions hold:

1. `M3` validation micro-CER improves over `M0-FT` by at least 2% relative.
2. Two or more core validation domains do not degrade.
3. No core domain degrades by more than 0.005 absolute CER.
4. Correct graph beats matched shuffled graph.
5. Gate values have non-zero variation.
6. Graph adapter gradients do not collapse.
7. `M3` is better than or meaningfully different from `M2`.

Failure stops the architecture branch. Only clear mapping, mask, length, scaling, or
serialization bugs may be corrected under a new recorded run.

## Success Criteria

Minimum positive result:

- mean `M3` CER below `M0-FT`;
- improvement in at least two of three seeds;
- paired overall micro-CER confidence interval below zero;
- no core domain worsens by more than 0.005 CER;
- correct graph beats shuffled graph;
- full topology is not reducible to `M2`.

The primary comparison is `M3` versus `M0-FT`. Paired bootstrap uses per-sample edit
counts on identical samples. Declared primary comparisons use Holm correction.

## Configuration Registry

Frozen configurations are stored in:

```text
configs/htr_adapter_v1/m0_ft_seed42.yaml
configs/htr_adapter_v1/m0_ft_seed43.yaml
configs/htr_adapter_v1/m0_ft_seed44.yaml
configs/htr_adapter_v1/m2_geometry_seed42.yaml
configs/htr_adapter_v1/m3_full_seed42.yaml
configs/htr_adapter_v1/m3_full_seed43.yaml
configs/htr_adapter_v1/m3_full_seed44.yaml
```
