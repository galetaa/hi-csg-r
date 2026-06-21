# School foreground v3 — controlled conclusion

## 1. Final verdict

| component | decision |
|---|---|
| `school_dark_auto` | accepted for future School Notebooks graph extraction |
| graph-fusion v3 checkpoint | rejected |
| graph-fusion v2 checkpoint | retained as final graph-vector model |
| historical H1 results | unchanged |
| domain-specific normalization | not justified by current evidence |

## 2. Controlled cross-evaluation

| checkpoint | old features CER | v3 features CER | feature ΔCER |
|---|---:|---:|---:|
| graph-v2 | 0.13970 | 0.13943 | -0.00027 |
| graph-v3 | 0.15396 | 0.15338 | -0.00058 |

## 3. Training-run effect

| shared manifest | graph-v3 − graph-v2 CER |
|---|---:|
| old features | +0.01426 |
| v3 features | +0.01395 |

The new checkpoint is worse under both manifests. Therefore its degradation cannot be attributed to foreground v3.

## 4. Dataset-local feature effect

| checkpoint | School Notebooks ΔCER from v3 features |
|---|---:|
| graph-v2 | -0.00110 |
| graph-v3 | -0.00236 |

HKR and Cyrillic CER remain unchanged under inference-time feature replacement. The small benefit is localized to School Notebooks, the only dataset whose foreground extraction was changed.

## 5. H2 result

- audited samples: 23
- good fix rate: 0.826
- partial fix rate: 0.087
- erased-ink rate: 0.000
- skeleton-follows-ink rate after repair: 0.826

## 6. H3 result

The best structural-core result remains localized to `hkr_words|word|unknown` with ROC-AUC 0.6723, PR-AUC 0.3532, and top-20% precision 0.3853.

Foreground repair therefore does not materially change the main H3 conclusion.

## 7. Scientific conclusion

Foreground v3 substantially improves the visible structural representation of audited School Notebooks samples. Cross-evaluation shows that the repaired features are compatible with the existing graph-fusion model and provide a small dataset-local CER improvement. A newly trained graph-fusion checkpoint nevertheless performs worse under both old and repaired feature manifests. Thus, improved visible graph quality does not automatically translate into improved HTR accuracy.

## 8. Next validation requirement

Because the original 23 samples came from a diagnostic CER/risk-quadrant subset, foreground v3 must next be validated on an independently sampled random School Notebooks subset before population-level repair rates are reported.
