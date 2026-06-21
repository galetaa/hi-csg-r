# Final results tables — v1

## Table 1. Hypothesis-level verdicts

| hypothesis | verdict | supported claim | main caveat |
|---|---|---|---|
| H1 robustness | partial support | Graph-aware variants show lower relative CER degradation under distortions. | They remain worse than image-only in clean and distorted absolute CER. |
| H2 structural preservation | partial support with preprocessing exception | HKR/Cyrillic audit samples usually preserve visible stroke structure. | School-notebooks failures are dominated by crop/border/binarization artifacts. |
| H3 graph diagnostics | partial support | Structural descriptors can detect high-error samples in stratified subsets. | Global single-feature correlations are weak; risk is not graph quality. |

## Table 2. H1 robustness summary

| model | clean CER | mean distorted CER | absolute CER delta | relative CER degradation | interpretation |
|---|---:|---:|---:|---:|---|
| `image_only` | 0.0822 | 0.1136 | 0.0314 | 38.20% | Best absolute recognizer; primary baseline. |
| `graph_vector_v2` | 0.1397 | 0.1697 | 0.0300 | 21.48% | Lower relative degradation but worse absolute CER. |
| `gated_v2_dist` | 0.1438 | 0.1724 | 0.0287 | 19.94% | Lowest relative degradation but worse absolute CER and low graph gate. |

## Table 3. H2 manual audit summary

| subset | n | critical topology error | skeleton follows ink | border artifact | mean graph quality | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `HKR + Cyrillic` | 77 | 2.60% | 96.10% | n/a | 2.870 | Usable diagnostic evidence for visible-stroke preservation. |
| `school_notebooks_clean` | 23 | 95.65% | 0.00% | 100.00% | 0.826 | Preprocessing/binarization failure mode; report separately. |

## Table 4. H2 by dataset

| dataset | n | usable | critical topology error | skeleton follows ink | border artifact | mean graph quality | failure stage |
|---|---:|---:|---:|---:|---:|---:|---|
| `cyrillic_handwriting` | 37 | 94.59% | 5.41% | 91.89% | 0.00% | 2.757 | ok:37 |
| `hkr_words` | 40 | 97.50% | 0.00% | 100.00% | 0.00% | 2.975 | ok:40 |
| `school_notebooks_clean` | 23 | 78.26% | 95.65% | 0.00% | 100.00% | 0.826 | binarization:23 |

## Table 5. H3 diagnostic signal

| analysis | best feature/set | group | n | metric | value | interpretation |
|---|---|---|---:|---|---:|---|
| global single-feature correlation | `dir_v_frac` | global | 5563 | Spearman r | -0.1049 | Weak. |
| single-feature high-error detection | `dir_v_frac` | global | 5563 | ROC-AUC | 0.5658 | Weak. |
| quality proxy only | `warning_count` | `cyrillic_handwriting` | 1563 | ROC-AUC | 0.5000 | Not useful. |
| multifeature structural descriptors | `structural_core` | `hkr_words|word|unknown` | 1090 | ROC-AUC | 0.6733 | Useful but localized. |

## Table 6. Final safe claim matrix

| claim | status |
|---|---|
| Graph descriptors are useful diagnostic signals. | supported, with stratification caveat |
| Graph-aware models outperform image-only HTR. | not supported |
| Graph-aware models degrade less relatively under distortions. | partially supported |
| Current graph pipeline preserves visible structure on all datasets. | not supported |
| Current graph pipeline preserves visible structure on audited HKR/Cyrillic samples. | partially supported |
| School-notebooks failures are graph-topology failures. | not supported; they are preprocessing artifacts |
| Structural risk is graph quality. | not supported |
| Structural risk can help find hard samples. | partially supported |
