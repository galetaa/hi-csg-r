# H1 robustness report — v1

## 1. Strict verdict

```text
Strong H1 supported: no
Partial robustness signal: yes
```

Graph-aware models show lower relative CER degradation under the tested distortion families, but they substantially underperform the image-only baseline on clean data and also have worse absolute CER on distorted data. Therefore the current evidence is insufficient for strong H1.

## 2. Main metrics

| model | clean CER | mean distorted CER | absolute CER delta | relative degradation | clean gap vs image-only |
|---|---:|---:|---:|---:|---:|
| `image_only` | 0.08224 | 0.11365 | 0.03141 | 38.20% | 0.00000 |
| `graph_vector_v2` | 0.13970 | 0.16971 | 0.03001 | 21.48% | 0.05746 |
| `gated_v2_dist` | 0.14376 | 0.17243 | 0.02867 | 19.94% | 0.06152 |

## 3. Distortion-family interpretation

| distortion | image-only rel. degradation | graph-vector rel. degradation | gated rel. degradation | strict interpretation |
|---|---:|---:|---:|---|
| `blur` | 25.86% | 20.43% | 18.21% | relative robustness signal |
| `low_contrast` | 41.09% | 15.49% | 14.13% | relative robustness signal |
| `noise` | 41.77% | 11.58% | 13.66% | relative robustness signal |
| `thick_strokes` | 14.98% | 14.55% | 13.05% | relative robustness signal |
| `thin_strokes` | 67.27% | 45.36% | 40.68% | relative robustness signal |

## 4. Methodological conclusion

The result should not be presented as a clean HTR improvement. The correct conclusion is narrower: graph-aware variants are less sensitive in relative terms to the tested visual distortions, but their absolute recognition quality is worse. This supports continuing with graph-quality and failure-analysis experiments, not further architecture search.

## 5. Next required work

1. Add paired bootstrap or permutation tests using per-sample predictions.
2. Run H3: graph quality/confidence versus per-sample CER.
3. Start the gold subset for H2 structural graph quality.
4. Stop adding new HTR architectures unless H2/H3 exposes a specific failure mode.