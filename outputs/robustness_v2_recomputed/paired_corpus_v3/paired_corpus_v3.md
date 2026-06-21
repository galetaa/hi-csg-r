# Paired corpus robustness analysis v3

## 1. Diagnostics

- joined records: 83445
- unique clean samples: 5563
- conditions: 15
- missing joins: `{}`

## 2. Corpus-level paired results

| scope | n | image relative degradation | graph relative degradation | relative advantage | relative 95% CI | relative one-sided p | absolute advantage | absolute 95% CI | graph−image distorted CER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dataset:cyrillic_handwriting` | 1563 | 16.70% | 12.97% | 3.73% | 0.66%–6.88% | 0.003850 | -0.00557 | -0.00937–-0.00184 | 0.08346 |
| `dataset:hkr_words` | 2000 | 28.26% | 19.06% | 9.20% | 4.64%–14.08% | 0.000050 | -0.00185 | -0.00421–0.00052 | 0.03704 |
| `dataset:school_notebooks_clean` | 2000 | 75.73% | 38.61% | 37.12% | 28.85%–46.13% | 0.000050 | -0.00317 | -0.00789–0.00165 | 0.08345 |
| `family:blur` | 5563 | 23.23% | 20.66% | 2.58% | -0.44%–5.52% | 0.028399 | -0.01027 | -0.01292–-0.00767 | 0.06990 |
| `family:low_contrast` | 5563 | 31.19% | 15.72% | 15.48% | 11.87%–19.17% | 0.000050 | 0.00298 | 0.00002–0.00591 | 0.05666 |
| `family:noise` | 5563 | 32.81% | 11.80% | 21.01% | 17.95%–24.11% | 0.000050 | 0.00973 | 0.00739–0.01207 | 0.04990 |
| `family:thick_strokes` | 5563 | 14.68% | 14.77% | -0.09% | -2.90%–2.77% | 0.529624 | -0.00888 | -0.01153–-0.00621 | 0.06851 |
| `family:thin_strokes` | 5563 | 66.94% | 45.64% | 21.29% | 17.05%–25.97% | 0.000050 | -0.01023 | -0.01321–-0.00725 | 0.06987 |
| `overall` | 5563 | 33.77% | 21.72% | 12.05% | 9.37%–14.81% | 0.000050 | -0.00333 | -0.00528–-0.00137 | 0.06297 |

## 3. Strict interpretation

The graph model has a statistically supported corpus-level relative robustness advantage.
A consistent positive absolute degradation advantage is not established.
Absolute distorted CER remains worse for the graph model.

This supports only a partial robustness claim. It is not evidence of superior absolute HTR accuracy.
