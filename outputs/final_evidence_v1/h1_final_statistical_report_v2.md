# H1 final statistical robustness report v2

## 1. Strict verdict

```text
Strong H1 supported: no
Partial H1 supported: yes
Relative robustness supported: yes
Absolute HTR advantage supported: no
```

## 2. Primary paired result

| metric | result |
|---|---:|
| image-only relative degradation | 33.77% |
| graph relative degradation | 21.72% |
| relative robustness advantage | 12.05% |
| relative advantage 95% CI | 9.37%–14.81% |
| one-sided permutation p | 0.000050 |
| absolute degradation advantage | -0.00333 |
| absolute advantage 95% CI | -0.00528–-0.00137 |
| graph − image distorted CER | 0.06297 |
| distorted CER gap 95% CI | 0.06019–0.06585 |

## 3. Results by dataset

| dataset | image rel. | graph rel. | advantage | 95% CI | p | relative verdict |
|---|---:|---:|---:|---:|---:|---|
| `cyrillic_handwriting` | 16.70% | 12.97% | 3.73% | 0.66%–6.88% | 0.003850 | `supported` |
| `hkr_words` | 28.26% | 19.06% | 9.20% | 4.64%–14.08% | 0.000050 | `supported` |
| `school_notebooks_clean` | 75.73% | 38.61% | 37.12% | 28.85%–46.13% | 0.000050 | `supported` |

## 4. Results by distortion family

| family | image rel. | graph rel. | advantage | 95% CI | p | relative verdict | absolute verdict |
|---|---:|---:|---:|---:|---:|---|---|
| `blur` | 23.23% | 20.66% | 2.58% | -0.44%–5.52% | 0.028399 | `inconclusive` | `graph_degrades_more` |
| `low_contrast` | 31.19% | 15.72% | 15.48% | 11.87%–19.17% | 0.000050 | `supported` | `graph_degrades_less` |
| `noise` | 32.81% | 11.80% | 21.01% | 17.95%–24.11% | 0.000050 | `supported` | `graph_degrades_less` |
| `thick_strokes` | 14.68% | 14.77% | -0.09% | -2.90%–2.77% | 0.529624 | `not_supported` | `graph_degrades_more` |
| `thin_strokes` | 66.94% | 45.64% | 21.29% | 17.05%–25.97% | 0.000050 | `supported` | `graph_degrades_more` |

## 5. Estimand note

The primary inferential result uses corpus-level CER with paired cluster resampling over the 5,563 clean source samples. Each source sample is retained together with all 15 distortion conditions.

The earlier 38.20% image-only degradation is the arithmetic mean of condition-level relative degradation. The paired corpus estimate is 33.77%. Both are valid descriptive quantities, but the paired corpus analysis is used for statistical inference.

## 6. Final scientific conclusion

The graph-vector model exhibits a statistically supported reduction in relative CER degradation under the tested visual distortions. The overall relative advantage is 12.05%, with a 95% cluster-bootstrap interval of 9.37%–14.81% and a one-sided paired permutation p-value of 0.000050.

This advantage is supported for low contrast, additive noise, and thinning of strokes; blur is inconclusive under the combined bootstrap-and-permutation criterion, and no advantage is found for stroke thickening.

However, the graph model does not have a positive absolute degradation advantage, and its absolute CER on distorted images remains substantially worse than the image-only baseline. Therefore strong H1 is rejected. The evidence supports only a partial claim of lower relative sensitivity to distortion.
