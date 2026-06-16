# Results

## 1. Recognition performance

| model/checkpoint | feature manifest | CER | WER | exact |
|---|---|---:|---:|---:|
| graph-v2 | old | 0.13970 | 0.49042 | 0.43897 |
| graph-v2 | foreground v3 | 0.13943 | 0.48975 | 0.43969 |
| graph-v3 retrain | old | 0.15396 | 0.52364 | 0.40626 |
| graph-v3 retrain | foreground v3 | 0.15338 | 0.52190 | 0.40823 |

Replacing the old graph features with foreground-v3 features changed the
retained graph-v2 CER by
-0.00027.

The new checkpoint remained worse with both feature manifests. Its
degradation was therefore caused primarily by the training run rather
than by foreground repair.

## 2. Descriptive robustness

| model | clean CER | mean distorted CER | absolute delta | relative degradation |
|---|---:|---:|---:|---:|
| image-only | 0.08224 | 0.11365 | 0.03141 | 38.20% |
| graph-vector, recomputed features | 0.13943 | 0.16971 | 0.03028 | 21.72% |

The graph-vector recognizer has lower proportional degradation but worse
clean and distorted absolute CER.

## 3. Paired corpus robustness

| metric | result |
|---|---:|
| image-only relative degradation | 33.77% |
| graph relative degradation | 21.72% |
| relative advantage | 12.05% |
| relative advantage 95% CI | 9.37%–14.81% |
| one-sided permutation p | 0.000050 |
| absolute degradation advantage | -0.00333 |
| absolute advantage 95% CI | -0.00528–-0.00137 |
| graph − image distorted CER | 0.06297 |

The graph model has a statistically supported relative robustness
advantage. It does not have a positive absolute degradation advantage
and remains worse in absolute distorted CER.

## 4. Robustness by distortion family

| family | image relative | graph relative | advantage | 95% CI | p | verdict |
|---|---:|---:|---:|---:|---:|---|
| `blur` | 23.23% | 20.66% | 2.58% | -0.44%–5.52% | 0.028399 | inconclusive |
| `low_contrast` | 31.19% | 15.72% | 15.48% | 11.87%–19.17% | 0.000050 | supported |
| `noise` | 32.81% | 11.80% | 21.01% | 17.95%–24.11% | 0.000050 | supported |
| `thick_strokes` | 14.68% | 14.77% | -0.09% | -2.90%–2.77% | 0.529624 | not supported |
| `thin_strokes` | 66.94% | 45.64% | 21.29% | 17.05%–25.97% | 0.000050 | supported |

Relative robustness is supported for low contrast, additive noise, and
stroke thinning. Blur is inconclusive under the combined confidence-
interval and permutation criterion. Stroke thickening provides no
relative advantage.

## 5. Original H2 diagnostic audit

| subset | n | critical topology error | skeleton follows ink | mean graph quality |
|---|---:|---:|---:|---:|
| HKR + Cyrillic | 77 | 2.60% | 96.10% | 2.870 |
| School Notebooks, old preprocessing | 23 | 95.65% | 0.00% | 0.826 |

The School Notebooks failure was localized to foreground extraction.

## 6. Independent random-100 foreground validation

| metric | count | rate |
|---|---:|---:|
| raw good fix | 92/100 | 92.00% |
| partial fix | 8/100 | 8.00% |
| bad fix | 0/100 | 0.00% |
| strict usable | 89/100 | 89.00% |
| real ink erased | 4/100 | 4.00% |
| residual artifact | 7/100 | 7.00% |
| skeleton follows ink | 96/100 | 96.00% |

The random validation supports `school_dark_auto` for the sampled School
Notebooks test distribution.

## 7. H3 graph diagnostics

| metric | result |
|---|---:|
| best global feature | `graph_endpoint_count` |
| global Spearman r | 0.0981 |
| structural feature set | `structural_core` |
| subgroup | `hkr_words|word|unknown` |
| n | 1090 |
| ROC-AUC | 0.6723 |
| PR-AUC | 0.3532 |
| PR-AUC lift | 1.6666 |
| top-20% precision | 0.3853 |

Individual global descriptors have weak correlations with CER.
Multifeature graph descriptors provide useful but localized high-error
detection.
