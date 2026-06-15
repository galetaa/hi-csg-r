# HI-CSG-R consolidated evidence report — v1

## 1. Executive verdict

```text
Overall result: mixed / partial support
H1 robustness: partial support only
H2 structural preservation: partial support with preprocessing exception
H3 graph diagnostics: partial support
```

The current evidence supports a narrower claim than originally hoped. Canonical visible-stroke graph descriptors are useful as diagnostic and robustness-related signals, but the current graph-aware HTR models do not outperform the image-only baseline in absolute recognition quality.

## 2. H1 — Robustness

### Verdict

```text
H1 strong form: not confirmed
H1 weak/relative form: partially supported
```

Graph-aware models showed lower relative CER degradation under distortions, but they had much worse clean CER and worse absolute distorted CER than the image-only baseline. Therefore this is not evidence that graph-aware recognition is better overall.

| model | clean CER | mean distorted CER | absolute CER delta | relative degradation |
|---|---:|---:|---:|---:|
| `image_only` | 0.0822 | 0.1136 | 0.0314 | 38.20% |
| `graph_vector_v2` | 0.1397 | 0.1697 | 0.0300 | 21.48% |
| `gated_v2_dist` | 0.1438 | 0.1724 | 0.0287 | 19.94% |

### H1 conclusion

The correct claim is: graph-aware variants are less sensitive in relative terms, but they are not competitive with the image-only baseline in absolute CER. This supports robustness analysis, not a better HTR system claim.

## 3. H2 — Structural graph preservation

### Verdict

```text
H2-v1: partial support with school-notebooks preprocessing exception
```

Manual audit indicates that HKR and Cyrillic samples generally preserve visible stroke structure. School-notebooks samples are dominated by upstream crop/border/binarization artifacts and should be reported separately.

| subset | n | critical topology error rate | skeleton follows ink rate | mean graph quality |
|---|---:|---:|---:|---:|
| `HKR + Cyrillic` | 77 | 2.60% | 96.10% | 2.870 |
| `school_notebooks_clean` | 23 | 95.65% | 0.00% | 0.826 |

### School-notebooks exception

The school-notebooks subset is not valid evidence against the graph abstraction itself. The observed failures occur earlier: crop/background borders are binarized as foreground, which then corrupts skeletons and graphs. A simple border-connected-component suppression check was rejected because it either failed to fix the artifact or removed handwriting.

## 4. H3 — Graph diagnostics

### Verdict

```text
H3: partial support
```

Global single-feature graph metrics do not strongly explain CER. However, multifeature structural descriptors provide useful high-error detection in stratified subsets.

| best multifeature signal | value |
|---|---:|
| feature set | `structural_core` |
| group | `hkr_words|word|unknown` |
| n | 1090 |
| ROC-AUC | 0.6733 |
| PR-AUC | 0.3466 |
| PR-AUC lift | 1.6356 |
| top20 precision | 0.3761 |

### H3 conclusion

The structural descriptor set is useful for finding hard samples, but it should not be described as graph quality. Manual H2 audit showed that high structural risk often marks sample difficulty rather than visible skeleton failure.

## 5. Safe claims

- Canonical visible-stroke graph descriptors are diagnostically useful in some settings.
- Graph-aware HTR variants are relatively less sensitive to distortions, but not better recognizers in absolute CER.
- HKR/Cyrillic graph extraction preserves visible stroke structure reasonably well in the audited subset.
- School-notebooks failures are dominated by upstream crop/binarization border artifacts.
- The current structural risk score is a hard-sample indicator, not a direct graph-quality score.

## 6. Unsafe claims to avoid

- Do not claim that graph-aware recognition beats the image-only baseline.
- Do not claim H1 is fully confirmed.
- Do not claim H2 holds uniformly across all datasets.
- Do not claim structural risk is equivalent to graph quality.
- Do not present school-notebooks failures as pure graph-topology failures.

## 7. Recommended thesis/research framing

The project should be framed as evidence that offline handwriting images can be converted into reproducible visible-stroke structural descriptors that support interpretability, robustness analysis, and failure triage. It should not be framed as a new state-of-the-art HTR model.

A precise final claim:

> Canonical visible-stroke graph descriptors provide a useful intermediate diagnostic representation for offline handwritten text recognition. They show partial robustness and high-error detection value, while exposing preprocessing limitations in noisy cropped notebook data. Current graph-aware recognition models do not outperform a strong image-only recognizer in absolute CER.

## 8. Next work

1. Freeze architecture experiments.
2. Treat school-notebooks as a preprocessing/crop-cleaning problem, not as graph topology evidence.
3. If time remains, run a small preprocessing experiment only on school-notebooks, but do not retrain HTR.
4. Prepare final figures/tables for H1/H2/H3.
5. Write the limitations section explicitly.