# Paired robustness analysis v2

## 1. Diagnostics

- joined records: 83445
- unique clean samples: 5563
- distortion conditions: 15
- missing joins: `{}`

## 2. Paired results

| scope | n | image rel. degradation | graph rel. degradation | advantage | advantage 95% CI | paired ΔCER advantage | paired 95% CI | one-sided p | graph−image distorted CER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dataset:cyrillic_handwriting` | 1563 | 16.68% | 12.54% | 4.14% | 0.78%–7.71% | -0.00482 | -0.00887–-0.00067 | 0.989401 | 0.08298 |
| `dataset:hkr_words` | 2000 | 28.18% | 17.46% | 10.72% | 5.99%–15.87% | -0.00034 | -0.00303–0.00235 | 0.592120 | 0.03798 |
| `dataset:school_notebooks_clean` | 2000 | 72.25% | 35.85% | 36.40% | 27.55%–46.29% | -0.00114 | -0.00645–0.00428 | 0.655717 | 0.08202 |
| `family:blur` | 5563 | 25.86% | 19.86% | 6.00% | 2.46%–9.54% | -0.00789 | -0.01111–-0.00474 | 1.000000 | 0.07246 |
| `family:low_contrast` | 5563 | 41.09% | 19.84% | 21.26% | 16.57%–26.10% | 0.00467 | 0.00080–0.00861 | 0.007800 | 0.05990 |
| `family:noise` | 5563 | 41.77% | 13.99% | 27.78% | 23.73%–31.89% | 0.01381 | 0.01062–0.01691 | 0.000050 | 0.05076 |
| `family:thick_strokes` | 5563 | 14.98% | 15.32% | -0.34% | -3.66%–3.00% | -0.01017 | -0.01352–-0.00693 | 1.000000 | 0.07474 |
| `family:thin_strokes` | 5563 | 67.27% | 44.39% | 22.89% | 18.00%–28.11% | -0.00984 | -0.01341–-0.00631 | 1.000000 | 0.07441 |
| `overall` | 5563 | 38.20% | 22.68% | 15.52% | 12.19%–18.96% | -0.00188 | -0.00441–0.00064 | 0.930903 | 0.06646 |

## 3. Interpretation

The paired analysis does not provide unambiguous statistical support for a graph robustness advantage.
The graph model nevertheless has worse absolute CER on distorted samples.

The paired test evaluates robustness change for the same source samples. It does not convert a relative robustness result into an absolute HTR improvement.
