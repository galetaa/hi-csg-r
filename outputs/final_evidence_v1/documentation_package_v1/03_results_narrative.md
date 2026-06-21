# 03 — Results narrative

## H1 — Robustness

The strong form of H1 is not confirmed. The image-only baseline remains the best absolute recognizer. Graph-aware variants show lower relative CER degradation under distortions, but they start from much worse clean CER and also have worse mean distorted CER.

| model | clean CER | mean distorted CER | relative degradation |
|---|---:|---:|---:|
| image-only | 0.0822 | 0.1136 | 38.20% |
| graph-vector | 0.1397 | 0.1697 | 21.48% |
| gated dist | 0.1438 | 0.1724 | 19.94% |

Interpretation: H1 receives weak/relative support only. It supports robustness analysis, not a claim of improved HTR.

## H2 — Structural preservation

H2 is partially supported with an important preprocessing exception. In the diagnostic audit subset, HKR and Cyrillic samples generally preserve visible stroke structure. School-notebooks samples are dominated by crop/border/binarization artifacts.

| subset | n | critical topology error | skeleton follows ink | mean quality |
|---|---:|---:|---:|---:|
| HKR + Cyrillic | 77 | 2.60% | 96.10% | 2.870 |
| school-notebooks | 23 | 95.65% | 0.00% | 0.826 |

Interpretation: school-notebooks should not be aggregated into a pure graph-topology failure claim. They expose an upstream preprocessing limitation.

## H3 — Graph diagnostics

H3 is partially supported. Global single-feature correlations are weak, but multifeature structural descriptors provide useful high-error detection in some stratified subsets.

| best signal | value |
|---|---:|
| feature set | `structural_core` |
| group | `hkr_words|word|unknown` |
| n | 1090 |
| ROC-AUC | 0.6733 |
| PR-AUC | 0.3466 |
| PR-AUC lift | 1.6356 |

Interpretation: graph-derived structural descriptors can help identify hard samples, but structural risk is not graph quality.
