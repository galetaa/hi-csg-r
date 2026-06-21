# Robustness with recomputed graph features

## 1. Overall

| mode | clean CER | mean distorted CER | absolute delta | relative degradation |
|---|---:|---:|---:|---:|
| `image_only` | 0.08224 | 0.11365 | 0.03141 | 38.20% |
| `graph_frozen_clean` | 0.13970 | 0.16971 | 0.03001 | 21.48% |
| `graph_recomputed_v3` | 0.13943 | 0.16971 | 0.03028 | 21.72% |

## 2. By distortion family

| distortion | image-only rel. degradation | frozen graph | recomputed graph |
|---|---:|---:|---:|
| `blur` | 25.86% | 20.43% | 20.66% |
| `low_contrast` | 41.09% | 15.49% | 15.72% |
| `noise` | 41.77% | 11.58% | 11.80% |
| `thick_strokes` | 14.98% | 14.55% | 14.77% |
| `thin_strokes` | 67.27% | 45.36% | 45.64% |

## 3. Methodological interpretation

The graph model retains a relative robustness advantage when graph features are recomputed from distorted images.
Frozen clean graph features made the previous robustness estimate optimistic.

Absolute CER must still be compared directly: lower relative degradation alone does not establish better recognition.
