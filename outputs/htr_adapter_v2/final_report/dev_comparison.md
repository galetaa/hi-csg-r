# HI-CSG-R Late Correction v2: development comparison

| Run | Gate | Baseline CER | Correct CER | Shuffle CER | Zero CER | Relative improvement |
|---|---:|---:|---:|---:|---:|---:|
| v2_1_dev_p05_seed42 | STOP | 0.139370 | 0.139106 | 0.139106 | 0.139370 | 0.189% |
| v2_2_dev_p05_seed42 | STOP | 0.139370 | 0.139053 | 0.138711 | 0.139370 | 0.227% |

Кандидат выбирается только среди вариантов со статусом `PASS`, по минимальному development micro-CER. Holdout и test в выборе не участвуют.
