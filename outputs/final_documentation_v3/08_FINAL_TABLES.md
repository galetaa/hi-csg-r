# Final tables

## Table 1. Hypothesis verdicts

| hypothesis                                 | verdict                         |
| ------------------------------------------ | ------------------------------- |
| Strong H1: better robust HTR system        | rejected                        |
| Partial H1: lower relative sensitivity     | supported                       |
| H2: visible structure preservation         | partially supported             |
| School foreground repair                   | supported on random test sample |
| H3: structural error diagnostics           | localized partial support       |
| Graph fusion improves absolute recognition | not supported                   |

## Table 2. Absolute recognition and robustness

| model        |                                          clean CER |                                          mean distorted CER |                                            relative degradation |
| ------------ | -------------------------------------------------: | ----------------------------------------------------------: | --------------------------------------------------------------: |
| image-only   |          0.08224 |          0.11365 |          38.20% |
| graph-vector | 0.13943 | 0.16971 | 21.72% |

## Table 3. Primary paired robustness result

| statistic          |                                                     value |
| ------------------ | --------------------------------------------------------: |
| relative advantage |                           12.05% |
| 95% CI             |      9.37%–14.81% |
| permutation p      | 0.000050 |
| absolute advantage |                        -0.00333 |
| absolute 95% CI    |      -0.00528–-0.00137 |
| distorted CER gap  |                         0.06297 |

## Table 4. Robustness families

| family                         | image relative | graph relative | advantage | 95% CI | verdict |
| ------------------------------ | -------------: | -------------: | --------: | -----: | ------- |
| | blur | 23.23% | 20.66% | 2.58% | -0.44%–5.52% | inconclusive |
| low_contrast | 31.19% | 15.72% | 15.48% | 11.87%–19.17% | supported |
| noise | 32.81% | 11.80% | 21.01% | 17.95%–24.11% | supported |
| thick_strokes | 14.68% | 14.77% | -0.09% | -2.90%–2.77% | not supported |
| thin_strokes | 66.94% | 45.64% | 21.29% | 17.05%–25.97% | supported | |                |                |           |        |         |

## Table 5. Random School Notebooks validation

| metric               |                                                     count |                                                  rate |
| -------------------- | --------------------------------------------------------: | ----------------------------------------------------: |
| good fix             |                   92 |                   92.00% |
| partial fix          |                8 |                8.00% |
| bad fix              |                    0 |                    0.00% |
| strict usable        |              89 |              89.00% |
| ink loss             |            4 |            4.00% |
| residual artifact    |  7 |  7.00% |
| skeleton follows ink | 96 | 96.00% |

## Table 6. Graph-feature cross-evaluation

| model    |                                old features CER |                      foreground-v3 features CER |                                                                                                 delta |
| -------- | ----------------------------------------------: | ----------------------------------------------: | ----------------------------------------------------------------------------------------------------: |
| graph-v2 | 0.13970 | 0.13943 | -0.00027 |
| graph-v3 | 0.15396 | 0.15338 | -0.00058 |

## Table 7. H3 diagnostic signal

| feature set                      | subgroup                   |                    n |                  ROC-AUC |                  PR-AUC |                 top-20 precision |
| -------------------------------- | -------------------------- | -------------------: | -----------------------: | ----------------------: | -------------------------------: |
| `structural_core` | `hkr_words|word|unknown` | 1090 | 0.6723 | 0.3532 | 0.3853 |
|                              |                            |                      |                          |                         |                                  |
