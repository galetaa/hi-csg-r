# Confidence-Aware Selective Prediction - Iteration 2

Risk models are fit on val exact-error labels and evaluated on test.

## Risk Quality

| model | risk method | AUC all | AUC School | ECE all | ECE School |
|---|---|---:|---:|---:|---:|
| `baseline` | `feature_only` | 0.6021 | 0.6216 | 0.0647 | 0.0827 |
| `baseline` | `model_confidence` | 0.8020 | 0.8144 | 0.0445 | 0.0734 |
| `baseline` | `confidence_graph` | 0.8148 | 0.8220 | 0.0175 | 0.0420 |
| `plus_5k_context` | `feature_only` | 0.6120 | 0.6280 | 0.0472 | 0.0735 |
| `plus_5k_context` | `model_confidence` | 0.7990 | 0.8185 | 0.0439 | 0.0866 |
| `plus_5k_context` | `confidence_graph` | 0.8158 | 0.8286 | 0.0226 | 0.0276 |
| `plus_10k_context` | `feature_only` | 0.6036 | 0.5987 | 0.0569 | 0.0808 |
| `plus_10k_context` | `model_confidence` | 0.7967 | 0.8199 | 0.0383 | 0.0772 |
| `plus_10k_context` | `confidence_graph` | 0.8122 | 0.8284 | 0.0251 | 0.0198 |

## Selective Coverage

| model | risk method | scope | coverage | CER | WER | exact |
|---|---|---|---:|---:|---:|---:|
| `baseline` | `feature_only` | `all` | 0.5000 | 0.1469 | 0.4955 | 0.5040 |
| `baseline` | `feature_only` | `all` | 0.7000 | 0.1442 | 0.5064 | 0.4825 |
| `baseline` | `feature_only` | `all` | 0.8000 | 0.1429 | 0.5046 | 0.4742 |
| `baseline` | `feature_only` | `all` | 0.9000 | 0.1421 | 0.5045 | 0.4647 |
| `baseline` | `feature_only` | `all` | 1.0000 | 0.1452 | 0.5130 | 0.4420 |
| `baseline` | `feature_only` | `school` | 0.5000 | 0.1463 | 0.4105 | 0.5920 |
| `baseline` | `feature_only` | `school` | 0.7000 | 0.1441 | 0.4457 | 0.5607 |
| `baseline` | `feature_only` | `school` | 0.8000 | 0.1472 | 0.4637 | 0.5431 |
| `baseline` | `feature_only` | `school` | 0.9000 | 0.1503 | 0.4828 | 0.5250 |
| `baseline` | `feature_only` | `school` | 1.0000 | 0.1581 | 0.5005 | 0.5070 |
| `baseline` | `model_confidence` | `all` | 0.5000 | 0.0600 | 0.2867 | 0.6682 |
| `baseline` | `model_confidence` | `all` | 0.7000 | 0.0804 | 0.3707 | 0.5745 |
| `baseline` | `model_confidence` | `all` | 0.8000 | 0.0955 | 0.4202 | 0.5267 |
| `baseline` | `model_confidence` | `all` | 0.9000 | 0.1145 | 0.4671 | 0.4829 |
| `baseline` | `model_confidence` | `all` | 1.0000 | 0.1452 | 0.5130 | 0.4420 |
| `baseline` | `model_confidence` | `school` | 0.5000 | 0.0598 | 0.2590 | 0.7460 |
| `baseline` | `model_confidence` | `school` | 0.7000 | 0.0818 | 0.3479 | 0.6564 |
| `baseline` | `model_confidence` | `school` | 0.8000 | 0.1002 | 0.4012 | 0.6075 |
| `baseline` | `model_confidence` | `school` | 0.9000 | 0.1230 | 0.4508 | 0.5572 |
| `baseline` | `model_confidence` | `school` | 1.0000 | 0.1581 | 0.5005 | 0.5070 |
| `baseline` | `confidence_graph` | `all` | 0.5000 | 0.0665 | 0.2972 | 0.6779 |
| `baseline` | `confidence_graph` | `all` | 0.7000 | 0.0832 | 0.3695 | 0.5886 |
| `baseline` | `confidence_graph` | `all` | 0.8000 | 0.0960 | 0.4167 | 0.5348 |
| `baseline` | `confidence_graph` | `all` | 0.9000 | 0.1141 | 0.4654 | 0.4851 |
| `baseline` | `confidence_graph` | `all` | 1.0000 | 0.1452 | 0.5130 | 0.4420 |
| `baseline` | `confidence_graph` | `school` | 0.5000 | 0.0623 | 0.2460 | 0.7570 |
| `baseline` | `confidence_graph` | `school` | 0.7000 | 0.0819 | 0.3400 | 0.6643 |
| `baseline` | `confidence_graph` | `school` | 0.8000 | 0.1013 | 0.3969 | 0.6106 |
| `baseline` | `confidence_graph` | `school` | 0.9000 | 0.1228 | 0.4503 | 0.5572 |
| `baseline` | `confidence_graph` | `school` | 1.0000 | 0.1581 | 0.5005 | 0.5070 |
| `plus_5k_context` | `feature_only` | `all` | 0.5000 | 0.1373 | 0.4704 | 0.5313 |
| `plus_5k_context` | `feature_only` | `all` | 0.7000 | 0.1358 | 0.4799 | 0.5072 |
| `plus_5k_context` | `feature_only` | `all` | 0.8000 | 0.1341 | 0.4808 | 0.4960 |
| `plus_5k_context` | `feature_only` | `all` | 0.9000 | 0.1334 | 0.4818 | 0.4845 |
| `plus_5k_context` | `feature_only` | `all` | 1.0000 | 0.1360 | 0.4907 | 0.4605 |
| `plus_5k_context` | `feature_only` | `school` | 0.5000 | 0.1252 | 0.3720 | 0.6330 |
| `plus_5k_context` | `feature_only` | `school` | 0.7000 | 0.1276 | 0.4043 | 0.6036 |
| `plus_5k_context` | `feature_only` | `school` | 0.8000 | 0.1301 | 0.4250 | 0.5837 |
| `plus_5k_context` | `feature_only` | `school` | 0.9000 | 0.1332 | 0.4422 | 0.5661 |
| `plus_5k_context` | `feature_only` | `school` | 1.0000 | 0.1403 | 0.4650 | 0.5425 |
| `plus_5k_context` | `model_confidence` | `all` | 0.5000 | 0.0574 | 0.2654 | 0.6855 |
| `plus_5k_context` | `model_confidence` | `all` | 0.7000 | 0.0761 | 0.3501 | 0.5937 |
| `plus_5k_context` | `model_confidence` | `all` | 0.8000 | 0.0903 | 0.3972 | 0.5474 |
| `plus_5k_context` | `model_confidence` | `all` | 0.9000 | 0.1070 | 0.4432 | 0.5027 |
| `plus_5k_context` | `model_confidence` | `all` | 1.0000 | 0.1360 | 0.4907 | 0.4605 |
| `plus_5k_context` | `model_confidence` | `school` | 0.5000 | 0.0523 | 0.2160 | 0.7870 |
| `plus_5k_context` | `model_confidence` | `school` | 0.7000 | 0.0733 | 0.3075 | 0.6979 |
| `plus_5k_context` | `model_confidence` | `school` | 0.8000 | 0.0866 | 0.3578 | 0.6488 |
| `plus_5k_context` | `model_confidence` | `school` | 0.9000 | 0.1057 | 0.4097 | 0.5978 |
| `plus_5k_context` | `model_confidence` | `school` | 1.0000 | 0.1403 | 0.4650 | 0.5425 |
| `plus_5k_context` | `confidence_graph` | `all` | 0.5000 | 0.0648 | 0.2779 | 0.6952 |
| `plus_5k_context` | `confidence_graph` | `all` | 0.7000 | 0.0800 | 0.3511 | 0.6091 |
| `plus_5k_context` | `confidence_graph` | `all` | 0.8000 | 0.0922 | 0.3960 | 0.5557 |
| `plus_5k_context` | `confidence_graph` | `all` | 0.9000 | 0.1068 | 0.4411 | 0.5063 |
| `plus_5k_context` | `confidence_graph` | `all` | 1.0000 | 0.1360 | 0.4907 | 0.4605 |
| `plus_5k_context` | `confidence_graph` | `school` | 0.5000 | 0.0531 | 0.2080 | 0.7950 |
| `plus_5k_context` | `confidence_graph` | `school` | 0.7000 | 0.0748 | 0.3018 | 0.7029 |
| `plus_5k_context` | `confidence_graph` | `school` | 0.8000 | 0.0887 | 0.3559 | 0.6506 |
| `plus_5k_context` | `confidence_graph` | `school` | 0.9000 | 0.1058 | 0.4078 | 0.5989 |
| `plus_5k_context` | `confidence_graph` | `school` | 1.0000 | 0.1403 | 0.4650 | 0.5425 |
| `plus_10k_context` | `feature_only` | `all` | 0.5000 | 0.1390 | 0.4821 | 0.5216 |
| `plus_10k_context` | `feature_only` | `all` | 0.7000 | 0.1350 | 0.4796 | 0.5118 |
| `plus_10k_context` | `feature_only` | `all` | 0.8000 | 0.1338 | 0.4811 | 0.5007 |
| `plus_10k_context` | `feature_only` | `all` | 0.9000 | 0.1324 | 0.4825 | 0.4873 |
| `plus_10k_context` | `feature_only` | `all` | 1.0000 | 0.1351 | 0.4924 | 0.4636 |
| `plus_10k_context` | `feature_only` | `school` | 0.5000 | 0.1302 | 0.3940 | 0.6130 |
| `plus_10k_context` | `feature_only` | `school` | 0.7000 | 0.1324 | 0.4236 | 0.5836 |
| `plus_10k_context` | `feature_only` | `school` | 0.8000 | 0.1351 | 0.4456 | 0.5637 |
| `plus_10k_context` | `feature_only` | `school` | 0.9000 | 0.1362 | 0.4614 | 0.5489 |
| `plus_10k_context` | `feature_only` | `school` | 1.0000 | 0.1389 | 0.4718 | 0.5390 |
| `plus_10k_context` | `model_confidence` | `all` | 0.5000 | 0.0569 | 0.2661 | 0.6927 |
| `plus_10k_context` | `model_confidence` | `all` | 0.7000 | 0.0771 | 0.3507 | 0.5994 |
| `plus_10k_context` | `model_confidence` | `all` | 0.8000 | 0.0911 | 0.3957 | 0.5535 |
| `plus_10k_context` | `model_confidence` | `all` | 0.9000 | 0.1077 | 0.4422 | 0.5083 |
| `plus_10k_context` | `model_confidence` | `all` | 1.0000 | 0.1351 | 0.4924 | 0.4636 |
| `plus_10k_context` | `model_confidence` | `school` | 0.5000 | 0.0529 | 0.2270 | 0.7780 |
| `plus_10k_context` | `model_confidence` | `school` | 0.7000 | 0.0708 | 0.3025 | 0.7057 |
| `plus_10k_context` | `model_confidence` | `school` | 0.8000 | 0.0870 | 0.3563 | 0.6525 |
| `plus_10k_context` | `model_confidence` | `school` | 0.9000 | 0.1086 | 0.4178 | 0.5922 |
| `plus_10k_context` | `model_confidence` | `school` | 1.0000 | 0.1389 | 0.4718 | 0.5390 |
| `plus_10k_context` | `confidence_graph` | `all` | 0.5000 | 0.0639 | 0.2794 | 0.7031 |
| `plus_10k_context` | `confidence_graph` | `all` | 0.7000 | 0.0809 | 0.3518 | 0.6097 |
| `plus_10k_context` | `confidence_graph` | `all` | 0.8000 | 0.0922 | 0.3963 | 0.5587 |
| `plus_10k_context` | `confidence_graph` | `all` | 0.9000 | 0.1076 | 0.4420 | 0.5097 |
| `plus_10k_context` | `confidence_graph` | `all` | 1.0000 | 0.1351 | 0.4924 | 0.4636 |
| `plus_10k_context` | `confidence_graph` | `school` | 0.5000 | 0.0551 | 0.2230 | 0.7840 |
| `plus_10k_context` | `confidence_graph` | `school` | 0.7000 | 0.0748 | 0.3054 | 0.7029 |
| `plus_10k_context` | `confidence_graph` | `school` | 0.8000 | 0.0884 | 0.3563 | 0.6519 |
| `plus_10k_context` | `confidence_graph` | `school` | 0.9000 | 0.1087 | 0.4164 | 0.5928 |
| `plus_10k_context` | `confidence_graph` | `school` | 1.0000 | 0.1389 | 0.4718 | 0.5390 |

## School clean_core vs hard_real

| model | risk method | bucket | n | CER | WER | exact |
|---|---|---|---:|---:|---:|---:|
| `baseline` | `feature_only` | `clean_core` | 1764 | 0.1478 | 0.4858 | 0.5227 |
| `baseline` | `feature_only` | `hard_real` | 236 | 0.2351 | 0.6102 | 0.3898 |
| `baseline` | `model_confidence` | `clean_core` | 1764 | 0.1478 | 0.4858 | 0.5227 |
| `baseline` | `model_confidence` | `hard_real` | 236 | 0.2351 | 0.6102 | 0.3898 |
| `baseline` | `confidence_graph` | `clean_core` | 1764 | 0.1478 | 0.4858 | 0.5227 |
| `baseline` | `confidence_graph` | `hard_real` | 236 | 0.2351 | 0.6102 | 0.3898 |
| `plus_5k_context` | `feature_only` | `clean_core` | 1764 | 0.1312 | 0.4450 | 0.5629 |
| `plus_5k_context` | `feature_only` | `hard_real` | 236 | 0.2088 | 0.6144 | 0.3898 |
| `plus_5k_context` | `model_confidence` | `clean_core` | 1764 | 0.1312 | 0.4450 | 0.5629 |
| `plus_5k_context` | `model_confidence` | `hard_real` | 236 | 0.2088 | 0.6144 | 0.3898 |
| `plus_5k_context` | `confidence_graph` | `clean_core` | 1764 | 0.1312 | 0.4450 | 0.5629 |
| `plus_5k_context` | `confidence_graph` | `hard_real` | 236 | 0.2088 | 0.6144 | 0.3898 |
| `plus_10k_context` | `feature_only` | `clean_core` | 1764 | 0.1291 | 0.4538 | 0.5573 |
| `plus_10k_context` | `feature_only` | `hard_real` | 236 | 0.2126 | 0.6059 | 0.4025 |
| `plus_10k_context` | `model_confidence` | `clean_core` | 1764 | 0.1291 | 0.4538 | 0.5573 |
| `plus_10k_context` | `model_confidence` | `hard_real` | 236 | 0.2126 | 0.6059 | 0.4025 |
| `plus_10k_context` | `confidence_graph` | `clean_core` | 1764 | 0.1291 | 0.4538 | 0.5573 |
| `plus_10k_context` | `confidence_graph` | `hard_real` | 236 | 0.2126 | 0.6059 | 0.4025 |

## Files

- `selective_summary.json`
- `coverage_curves.csv`
- `risk_table_by_bucket.csv`
- `calibration_bins.csv`
- `confidence_predictions/*.jsonl`
