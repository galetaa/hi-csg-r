# H2 manual audit summary — v2

## 1. Overall

| metric | value |
|---|---:|
| n | 100 |
| usable rate | 0.920 |
| critical topology error rate | 0.240 |
| skeleton follows ink rate | 0.740 |
| border artifact rate | 0.230 |
| mean graph quality 0–3 | 2.400 |

## 2. Failure stages

| stage | n |
|---|---:|
| `binarization` | 23 |
| `ok` | 77 |

## 3. By dataset

| dataset | n | usable | critical | follows ink | border artifact | mean quality | failure stages |
|---|---:|---:|---:|---:|---:|---:|---|
| `cyrillic_handwriting` | 37 | 0.946 | 0.054 | 0.919 | 0.000 | 2.757 | ok:37 |
| `hkr_words` | 40 | 0.975 | 0.000 | 1.000 | 0.000 | 2.975 | ok:40 |
| `school_notebooks_clean` | 23 | 0.783 | 0.957 | 0.000 | 1.000 | 0.826 | binarization:23 |

## 4. By audit cell

| cell | n | usable | critical | follows ink | border artifact | mean quality | mean CER | mean risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_highCER_highRisk` | 25 | 0.800 | 0.240 | 0.720 | 0.200 | 2.320 | 0.394 | 0.899 |
| `B_highCER_lowRisk` | 25 | 0.960 | 0.240 | 0.760 | 0.200 | 2.480 | 0.374 | 0.154 |
| `C_lowCER_highRisk` | 25 | 0.920 | 0.320 | 0.680 | 0.320 | 2.160 | 0.000 | 0.860 |
| `D_lowCER_lowRisk` | 25 | 1.000 | 0.160 | 0.800 | 0.200 | 2.640 | 0.000 | 0.097 |

## 5. Strict interpretation

The school-notebooks subset is dominated by upstream crop/binarization artifacts. These samples should not be interpreted as pure graph-topology failures.

The current structural risk score should be interpreted as a hard-sample indicator, not as a direct graph-quality score. Manual audit separates extraction failures from crop and binarization failures.