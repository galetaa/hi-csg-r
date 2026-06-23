# Selective Prediction - Iteration 2 Line Augmentation

Risk score is feature-only: graph/foreground extremeness, School quality rules, graph warnings, and short-text risk. Predictions are not used to compute the risk score.

## Overall

| model | n | CER | WER | exact | risk AUC all | risk AUC School |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 5563 | 0.1453 | 0.5134 | 0.4411 | 0.5037 | 0.4581 |
| `plus_5k_context` | 5563 | 0.1360 | 0.4907 | 0.4605 | 0.4964 | 0.4714 |
| `plus_10k_context` | 5563 | 0.1351 | 0.4924 | 0.4636 | 0.5071 | 0.4800 |

## School Clean vs Hard

| model | bucket | n | CER | WER | exact |
|---|---|---:|---:|---:|---:|
| `baseline` | `clean_core` | 1764 | 0.1477 | 0.4895 | 0.5193 |
| `baseline` | `hard_real` | 236 | 0.2307 | 0.5932 | 0.4068 |
| `baseline` | `invalid_or_review` | 0 | 0.0000 | 0.0000 | 0.0000 |
| `plus_5k_context` | `clean_core` | 1764 | 0.1312 | 0.4450 | 0.5629 |
| `plus_5k_context` | `hard_real` | 236 | 0.2088 | 0.6144 | 0.3898 |
| `plus_5k_context` | `invalid_or_review` | 0 | 0.0000 | 0.0000 | 0.0000 |
| `plus_10k_context` | `clean_core` | 1764 | 0.1291 | 0.4538 | 0.5573 |
| `plus_10k_context` | `hard_real` | 236 | 0.2126 | 0.6059 | 0.4025 |
| `plus_10k_context` | `invalid_or_review` | 0 | 0.0000 | 0.0000 | 0.0000 |

## Selective Coverage

| model | scope | coverage | CER | WER | exact |
|---|---|---:|---:|---:|---:|
| `baseline` | `all` | 0.5000 | 0.1227 | 0.5000 | 0.4551 |
| `baseline` | `all` | 0.7000 | 0.1320 | 0.5162 | 0.4343 |
| `baseline` | `all` | 0.8000 | 0.1357 | 0.5212 | 0.4285 |
| `baseline` | `all` | 0.9000 | 0.1410 | 0.5253 | 0.4244 |
| `baseline` | `all` | 1.0000 | 0.1453 | 0.5134 | 0.4411 |
| `baseline` | `school` | 0.5000 | 0.1351 | 0.5330 | 0.4780 |
| `baseline` | `school` | 0.7000 | 0.1491 | 0.5454 | 0.4657 |
| `baseline` | `school` | 0.8000 | 0.1476 | 0.5209 | 0.4888 |
| `baseline` | `school` | 0.9000 | 0.1490 | 0.5053 | 0.5033 |
| `baseline` | `school` | 1.0000 | 0.1575 | 0.5018 | 0.5060 |
| `plus_5k_context` | `all` | 0.5000 | 0.1193 | 0.4850 | 0.4637 |
| `plus_5k_context` | `all` | 0.7000 | 0.1235 | 0.4916 | 0.4548 |
| `plus_5k_context` | `all` | 0.8000 | 0.1267 | 0.4980 | 0.4476 |
| `plus_5k_context` | `all` | 0.9000 | 0.1320 | 0.5014 | 0.4444 |
| `plus_5k_context` | `all` | 1.0000 | 0.1360 | 0.4907 | 0.4605 |
| `plus_5k_context` | `school` | 0.5000 | 0.1221 | 0.4855 | 0.5270 |
| `plus_5k_context` | `school` | 0.7000 | 0.1299 | 0.4964 | 0.5136 |
| `plus_5k_context` | `school` | 0.8000 | 0.1306 | 0.4794 | 0.5294 |
| `plus_5k_context` | `school` | 0.9000 | 0.1323 | 0.4644 | 0.5439 |
| `plus_5k_context` | `school` | 1.0000 | 0.1403 | 0.4650 | 0.5425 |
| `plus_10k_context` | `all` | 0.5000 | 0.1151 | 0.4802 | 0.4730 |
| `plus_10k_context` | `all` | 0.7000 | 0.1205 | 0.4835 | 0.4651 |
| `plus_10k_context` | `all` | 0.8000 | 0.1241 | 0.4936 | 0.4573 |
| `plus_10k_context` | `all` | 0.9000 | 0.1301 | 0.5006 | 0.4504 |
| `plus_10k_context` | `all` | 1.0000 | 0.1351 | 0.4924 | 0.4636 |
| `plus_10k_context` | `school` | 0.5000 | 0.1154 | 0.4820 | 0.5300 |
| `plus_10k_context` | `school` | 0.7000 | 0.1244 | 0.4989 | 0.5150 |
| `plus_10k_context` | `school` | 0.8000 | 0.1284 | 0.4866 | 0.5262 |
| `plus_10k_context` | `school` | 0.9000 | 0.1309 | 0.4725 | 0.5394 |
| `plus_10k_context` | `school` | 1.0000 | 0.1389 | 0.4718 | 0.5390 |

## School Feature Signal

Spearman correlation with per-sample CER on School test samples.

| model | feature | rho |
|---|---|---:|
| `baseline` | `risk_score` | 0.0447 |
| `baseline` | `fg_fraction` | 0.1386 |
| `baseline` | `skel_fraction` | 0.1446 |
| `baseline` | `cc_count` | 0.1905 |
| `baseline` | `dir_h_frac` | -0.0771 |
| `baseline` | `stroke_width_mean` | -0.0575 |
| `baseline` | `ruling_response_mean` | -0.0538 |
| `plus_5k_context` | `risk_score` | 0.0312 |
| `plus_5k_context` | `fg_fraction` | 0.1500 |
| `plus_5k_context` | `skel_fraction` | 0.1337 |
| `plus_5k_context` | `cc_count` | 0.1773 |
| `plus_5k_context` | `dir_h_frac` | -0.0277 |
| `plus_5k_context` | `stroke_width_mean` | -0.0242 |
| `plus_5k_context` | `ruling_response_mean` | -0.0618 |
| `plus_10k_context` | `risk_score` | 0.0511 |
| `plus_10k_context` | `fg_fraction` | 0.1892 |
| `plus_10k_context` | `skel_fraction` | 0.1489 |
| `plus_10k_context` | `cc_count` | 0.1732 |
| `plus_10k_context` | `dir_h_frac` | -0.0662 |
| `plus_10k_context` | `stroke_width_mean` | -0.0174 |
| `plus_10k_context` | `ruling_response_mean` | -0.0277 |

## Files

- `risk_table_by_bucket.csv`
- `coverage_curves.csv`
- `selective_summary.json`
