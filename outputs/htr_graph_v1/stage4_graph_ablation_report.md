# Stage 4 graph-aware ablation report

## 1. Purpose

This report evaluates whether graph-derived information improves HTR on the tri10k mixed Cyrillic subset. Two graph-aware families were tested: global graph-vector fusion and local graph-derived image channels.

## 2. Main results

| model | mixed val CER | Cyrillic test CER | HKR test CER | School test CER |
|---|---:|---:|---:|---:|
| image-only old trainer | 0.1046 | 0.1932 | 0.0956 | 0.1575 |
| graph-vector v2 lowcap-all | 0.0991 | 0.1984 | 0.0879 | 0.1580 |
| local gray | 0.1025 | 0.2015 | 0.0913 | 0.1666 |
| local gray+fg | 0.1068 | 0.2056 | 0.0871 | 0.1742 |
| local gray+skel | 0.1054 | 0.2076 | 0.0872 | 0.1678 |
| local gray+dist | 0.1027 | 0.1995 | 0.0874 | 0.1681 |
| local gray+fg+skel | 0.1064 | 0.2043 | 0.0934 | 0.1744 |
| local gray+fg+skel+dist | 0.1025 | 0.2003 | 0.0853 | 0.1683 |

## 3. Local-channel comparison against local gray control

| model | mixed val Δ | Cyrillic Δ | HKR Δ | School Δ |
|---|---:|---:|---:|---:|
| local gray | 0.0% | 0.0% | 0.0% | 0.0% |
| local gray+fg | -4.2% | -2.0% | 4.6% | -4.6% |
| local gray+skel | -2.8% | -3.0% | 4.5% | -0.7% |
| local gray+dist | -0.2% | 1.0% | 4.3% | -0.9% |
| local gray+fg+skel | -3.8% | -1.4% | -2.3% | -4.7% |
| local gray+fg+skel+dist | 0.0% | 0.6% | 6.6% | -1.0% |

## 4. Interpretation

The global graph-vector low-capacity fusion gives the best mixed validation CER, but the gain is not stable across all test domains. It improves HKR Words but does not improve Cyrillic Handwriting or School Notebooks consistently.

The local-channel ablation shows that graph-derived channels are useful for HKR Words. The full local channel variant improves HKR from 0.0913 to 0.0853 CER relative to the local gray control. However, the same channels hurt School Notebooks and do not reliably improve Cyrillic Handwriting.

Therefore, naive early fusion of graph-derived channels is not robust enough. The next architecture should use gated or residual graph injection so that the model can suppress graph channels when they are harmful.

## 5. Stage 4 conclusion

```text
[x] graph feature extraction completed
[x] global graph-vector fusion tested
[x] local graph-channel fusion tested
[x] channel ablation completed
[!] graph signal is useful but domain-dependent
[next] gated/residual graph-aware fusion
```