# Stage 4 graph-vector fusion report

## 1. Purpose

This report compares the tri10k image-only control model against global graph-vector fusion variants. All graph-vector variants exclude `text_len` to avoid label-length leakage.

## 2. Overall mixed validation

| run | best penalty | CER | WER | exact | pred_len | blank |
|---|---:|---:|---:|---:|---:|---:|
| tri10k image-only v1 | -0.8 | 0.1046 | 0.3888 | 0.5733 | 7.79 | 0.840 |
| tri10k graph-fusion v1 all-features | -0.5 | 0.1030 | 0.3925 | 0.5590 | 7.79 | 0.828 |
| tri10k graph-fusion v2 lowcap-all | -0.4 | 0.0991 | 0.3853 | 0.5632 | 7.79 | 0.832 |
| tri10k graph-fusion v3 normtopo | -0.2 | 0.1004 | 0.3877 | 0.5623 | 7.77 | 0.838 |

## 3. Test CER by dataset

| run | Cyrillic test | HKR test | School test |
|---|---:|---:|---:|
| tri10k image-only v1 | 0.1932 | 0.0956 | 0.1575 |
| tri10k graph-fusion v1 all-features | 0.2046 | 0.0942 | 0.1612 |
| tri10k graph-fusion v2 lowcap-all | 0.1984 | 0.0879 | 0.1580 |
| tri10k graph-fusion v3 normtopo | 0.1968 | 0.0889 | 0.1597 |

## 4. Relative change vs image-only

| run | Cyrillic | HKR | School |
|---|---:|---:|---:|
| tri10k graph-fusion v1 all-features | -5.9% | 1.5% | -2.3% |
| tri10k graph-fusion v2 lowcap-all | -2.7% | 8.0% | -0.3% |
| tri10k graph-fusion v3 normtopo | -1.9% | 7.0% | -1.4% |

## 5. Interpretation

Global graph-vector fusion improves mixed validation CER, with the best result from the low-capacity all-feature variant. However, the improvement is not robust across datasets: HKR Words improves, while Cyrillic Handwriting and School Notebooks do not consistently improve on test.

The normalized-topology-only variant does not outperform low-capacity all-feature fusion, suggesting that raw geometry and domain/style cues contribute to the observed gain.

Conclusion: global graph-vector fusion is useful as a diagnostic baseline but should not be treated as the final graph-aware model. The next stage should inject graph-derived structure locally, aligned with image coordinates.

## 6. Next step

Proceed to local graph-aware CRNN using additional foreground/skeleton/distance channels.
