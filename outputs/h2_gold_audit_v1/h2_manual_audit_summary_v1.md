# H2 manual audit summary — v1

## 1. Overall

| metric | value |
|---|---:|
| n | 100 |
| usable rate | 0.920 |
| critical topology error rate | 0.240 |
| skeleton follows ink rate | 0.740 |
| mean graph quality 0–3 | 2.400 |

## 2. By audit cell

| cell | n | usable | critical | follows ink | mean quality | mean CER | mean risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_highCER_highRisk` | 25 | 0.800 | 0.240 | 0.720 | 2.320 | 0.394 | 0.899 |
| `B_highCER_lowRisk` | 25 | 0.960 | 0.240 | 0.760 | 2.480 | 0.374 | 0.154 |
| `C_lowCER_highRisk` | 25 | 0.920 | 0.320 | 0.680 | 2.160 | 0.000 | 0.860 |
| `D_lowCER_lowRisk` | 25 | 1.000 | 0.160 | 0.800 | 2.640 | 0.000 | 0.097 |

## 3. By dataset

| dataset | n | usable | critical | follows ink | mean quality | mean CER |
|---|---:|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 37 | 0.946 | 0.054 | 0.919 | 2.757 | 0.264 |
| `hkr_words` | 40 | 0.975 | 0.000 | 1.000 | 2.975 | 0.135 |
| `school_notebooks_clean` | 23 | 0.783 | 0.957 | 0.000 | 0.826 | 0.176 |

## 4. Strict interpretation

High structural risk does not strongly correspond to visible critical graph failures. It appears to capture sample difficulty or structural complexity more than extraction failure.

## 5. Recommended conclusion

Use this audit to separate two claims: graph structural descriptors can help identify hard samples, but the current scalar risk score should not be presented as a direct graph-quality measure.