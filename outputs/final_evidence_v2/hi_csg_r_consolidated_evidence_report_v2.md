# HI-CSG-R consolidated evidence report — v2

## 1. Executive verdict

```text
Overall result: mixed but scientifically informative
Strong H1: rejected
Partial relative-robustness H1: supported
H2 visible-structure preservation: partially supported
School Notebooks preprocessing repair: independently supported
H3 graph diagnostics: localized partial support
Absolute HTR improvement from graph fusion: not supported
```

Canonical visible-stroke graph descriptors form a useful intermediate representation for robustness analysis, preprocessing validation, and high-error sample triage. They do not currently produce a recognizer that outperforms the image-only baseline in absolute character error rate.

## 2. H1 — Robustness under visual distortions

### 2.1 Primary paired result

| metric | result |
|---|---:|
| image-only relative degradation | 33.77% |
| graph relative degradation | 21.72% |
| relative robustness advantage | 12.05% |
| cluster-bootstrap 95% CI | 9.37%–14.81% |
| paired permutation p | 0.000050 |
| absolute degradation advantage | -0.00333 |
| absolute advantage 95% CI | -0.00528–-0.00137 |
| graph − image distorted CER | 0.06297 |

The graph model has a statistically supported advantage in relative CER degradation. However, its absolute degradation is not better, and its distorted-image CER remains substantially higher.

### 2.2 Descriptive absolute performance

| model | clean CER | mean distorted CER | mean absolute delta | relative degradation |
|---|---:|---:|---:|---:|
| `image_only` | 0.08224 | 0.11365 | 0.03141 | 38.20% |
| `graph_vector_v2_recomputed` | 0.13943 | 0.16971 | 0.03028 | 21.72% |

### 2.3 Distortion-family evidence

| family | image relative | graph relative | advantage | 95% CI | p | verdict |
|---|---:|---:|---:|---:|---:|---|
| `blur` | 23.23% | 20.66% | 2.58% | -0.44%–5.52% | 0.028399 | inconclusive |
| `low_contrast` | 31.19% | 15.72% | 15.48% | 11.87%–19.17% | 0.000050 | supported |
| `noise` | 32.81% | 11.80% | 21.01% | 17.95%–24.11% | 0.000050 | supported |
| `thick_strokes` | 14.68% | 14.77% | -0.09% | -2.90%–2.77% | 0.529624 | not supported |
| `thin_strokes` | 66.94% | 45.64% | 21.29% | 17.05%–25.97% | 0.000050 | supported |

Relative robustness is supported for low contrast, additive noise, and stroke thinning. Blur is inconclusive under the combined bootstrap-and-permutation criterion, while stroke thickening shows no relative advantage.

## 3. H2 — Visible-structure preservation

### 3.1 Original diagnostic audit

| subset | n | critical topology error | skeleton follows ink | mean graph quality |
|---|---:|---:|---:|---:|
| `HKR + Cyrillic` | 77 | 2.60% | 96.10% | 2.870 |
| `School Notebooks, old preprocessing` | 23 | 95.65% | 0.00% | 0.826 |

The original School Notebooks failure was localized to foreground extraction rather than to canonical graph construction itself.

### 3.2 Independent random validation of foreground v3

| metric | result |
|---|---:|
| random test samples | 100 |
| raw good-fix rate | 92.00% |
| partial-fix rate | 8.00% |
| complete bad-fix rate | 0.00% |
| strict usable rate | 89.00% |
| real-ink loss rate | 4.00% |
| residual background-artifact rate | 7.00% |
| skeleton-follows-ink rate | 96.00% |

`school_dark_auto` is therefore accepted as the School Notebooks foreground extraction method for subsequent graph processing. The result generalizes beyond the original diagnostic subset to the sampled test distribution.

### 3.3 Controlled recognition cross-evaluation

| checkpoint | old features CER | v3 features CER | ΔCER |
|---|---:|---:|---:|
| graph-v2 | 0.13970 | 0.13943 | -0.00027 |
| graph-v3 retrain | 0.15396 | 0.15338 | -0.00058 |

Foreground v3 does not cause the graph-v3 checkpoint degradation. Both checkpoints improve slightly when evaluated with the repaired features, while the newly trained checkpoint remains worse under both manifests. The degradation is therefore attributed to the training run rather than to foreground repair.

## 4. H3 — Graph-derived error diagnostics

| metric | result |
|---|---:|
| best global correlation feature | `graph_endpoint_count` |
| best global Spearman r | 0.0981 |
| best structural feature set | `structural_core` |
| best subgroup | `hkr_words|word|unknown` |
| subgroup n | 1090 |
| ROC-AUC | 0.6723 |
| PR-AUC | 0.3532 |
| PR-AUC lift | 1.6666 |
| top-20% precision | 0.3853 |

Global individual structural features have weak associations with recognition error. Multifeature descriptors provide useful but localized high-error detection, especially in the HKR word subset. They should be interpreted as sample-difficulty signals rather than direct graph-quality scores.

## 5. Final hypothesis matrix

| hypothesis | final status | supported interpretation |
|---|---|---|
| H1 strong: graph-aware HTR is more robust overall | rejected | Absolute clean and distorted CER remain worse. |
| H1 partial: graph model has lower relative sensitivity | supported | Paired corpus relative advantage is statistically supported. |
| H2: visible structure is preserved | partially supported | Supported diagnostically for HKR/Cyrillic and after preprocessing repair for sampled School Notebooks data. |
| H3: graph descriptors diagnose recognition difficulty | partially supported | Useful multifeature signal exists in localized strata. |
| Graph fusion improves recognition accuracy | not supported | Current fusion models remain inferior to image-only HTR. |

## 6. Safe claims

- The graph-vector model has a statistically supported relative robustness advantage under the tested distortions.
- The graph-vector model remains worse than the image-only baseline in clean and distorted absolute CER.
- `school_dark_auto` substantially repairs School Notebooks foreground extraction on an independently sampled test subset.
- Foreground repair improves visible structural extraction but does not materially improve graph-fusion recognition.
- Multifeature graph descriptors provide localized value for high-error sample triage.
- The generated graph describes visible static stroke structure, not the true writing trajectory.

## 7. Claims to avoid

- Do not claim graph-aware recognition is superior to image-only recognition.
- Do not claim that strong H1 is confirmed.
- Do not claim uniform topology preservation across all datasets and acquisition settings.
- Do not treat the random-100 School Notebooks result as evidence for all handwriting domains.
- Do not claim that foreground v3 improves recognition accuracy in a practically meaningful way.
- Do not describe structural risk as direct graph-quality ground truth.
- Do not describe the graph as reconstructed pen trajectory.

## 8. Final project framing

The contribution is not a new top-performing recognizer. It is a controlled study of canonical visible-stroke graph descriptors as an intermediate structural representation for offline handwriting. The representation provides measurable value for relative robustness analysis, preprocessing validation, and localized failure triage, while the experiments also identify the limits of direct global graph-vector fusion.
