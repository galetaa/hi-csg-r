# H3 graph quality vs CER report — v1

## 1. Purpose

This report tests whether graph-derived structural features are diagnostically related to recognition errors. The primary analysis uses image-only predictions so the graph features are not part of the model input.

## 2. Dataset

```text
joined samples: 5563
manifest samples: 5563
prediction samples: 5563
high-error quantile: 0.8
high-error CER threshold: 0.16667
```

## 3. Dataset CER breakdown

| dataset | n | mean CER | median CER | p90 CER | exact-rate |
|---|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | 1563 | 0.12034 | 0.07143 | 0.33333 | 0.46577 |
| `hkr_words` | 2000 | 0.05816 | 0.00000 | 0.18182 | 0.65450 |
| `school_notebooks_clean` | 2000 | 0.07653 | 0.00000 | 0.27273 | 0.71850 |

## 4. Top Spearman correlations with CER

| feature | n | Spearman r | p-value | Pearson r |
|---|---:|---:|---:|---:|
| `dir_v_frac` | 5563 | -0.1049 | 4.387e-15 | -0.0724 |
| `dir_h_frac` | 5563 | 0.0818 | 1.014e-09 | 0.0789 |
| `aspect_ratio` | 5563 | 0.0816 | 1.074e-09 | -0.0177 |
| `width` | 5563 | 0.0816 | 1.074e-09 | -0.0177 |
| `skel_fraction` | 5563 | -0.0702 | 1.608e-07 | -0.0072 |
| `graph_endpoint_count` | 5563 | 0.0547 | 4.452e-05 | 0.0092 |
| `stroke_width_p90` | 5563 | 0.0476 | 3.874e-04 | 0.1210 |
| `dir_diag_down_frac` | 5563 | -0.0467 | 4.981e-04 | -0.0009 |
| `stroke_width_mean` | 5563 | 0.0435 | 1.188e-03 | 0.1067 |
| `stroke_width_p50` | 5563 | 0.0426 | 1.501e-03 | 0.0864 |
| `degree_hist_4` | 5563 | -0.0417 | 1.871e-03 | 0.0187 |
| `bbox_w_frac` | 5563 | 0.0413 | 2.048e-03 | -0.0469 |
| `branchpoint_per_100_skel` | 5563 | -0.0409 | 2.301e-03 | 0.0100 |
| `graph_avg_degree` | 5563 | -0.0398 | 2.968e-03 | 0.0096 |
| `degree_hist_3` | 5563 | -0.0393 | 3.334e-03 | 0.0029 |
| `stroke_width_std` | 5563 | 0.0389 | 3.671e-03 | 0.1353 |
| `cc_area_max_frac` | 5563 | -0.0389 | 3.735e-03 | 0.0436 |
| `graph_nodes` | 5563 | 0.0372 | 5.505e-03 | -0.0090 |
| `skel_pixels` | 5563 | 0.0372 | 5.505e-03 | -0.0090 |
| `cc_count` | 5563 | 0.0369 | 5.943e-03 | 0.0049 |

## 5. Top single-feature high-error detectors

| feature | ROC-AUC | PR-AUC | direction | top20 precision | top20 recall |
|---|---:|---:|---|---:|---:|
| `dir_v_frac` | 0.5658 | 0.2461 | lower_feature_higher_error | 0.2624 | 0.2566 |
| `dir_h_frac` | 0.5586 | 0.2434 | higher_feature_higher_error | 0.2399 | 0.2346 |
| `stroke_width_std` | 0.5564 | 0.2697 | higher_feature_higher_error | 0.2767 | 0.2707 |
| `stroke_width_p90` | 0.5558 | 0.2560 | higher_feature_higher_error | 0.2523 | 0.3902 |
| `graph_nodes` | 0.5533 | 0.2420 | lower_feature_higher_error | 0.2563 | 0.2513 |
| `skel_pixels` | 0.5533 | 0.2420 | lower_feature_higher_error | 0.2563 | 0.2513 |
| `graph_edges_8n` | 0.5531 | 0.2421 | lower_feature_higher_error | 0.2513 | 0.2469 |
| `stroke_width_mean` | 0.5456 | 0.2556 | higher_feature_higher_error | 0.2812 | 0.2750 |
| `graph_branchpoint_count` | 0.5408 | 0.2416 | lower_feature_higher_error | 0.2559 | 0.2583 |
| `bbox_y0_frac` | 0.5404 | 0.2313 | higher_feature_higher_error | 0.2451 | 0.3172 |
| `stroke_width_p50` | 0.5396 | 0.2377 | higher_feature_higher_error | 0.2196 | 0.5782 |
| `bbox_area_frac` | 0.5391 | 0.2305 | lower_feature_higher_error | 0.2417 | 0.2364 |
| `bbox_w_frac` | 0.5388 | 0.2256 | lower_feature_higher_error | 0.2462 | 0.2408 |
| `aspect_ratio` | 0.5343 | 0.2231 | lower_feature_higher_error | 0.2368 | 0.2320 |
| `width` | 0.5343 | 0.2231 | lower_feature_higher_error | 0.2368 | 0.2320 |
| `graph_endpoint_count` | 0.5312 | 0.2275 | lower_feature_higher_error | 0.2504 | 0.2715 |
| `fg_fraction` | 0.5292 | 0.2367 | higher_feature_higher_error | 0.2525 | 0.2469 |
| `bbox_x0_frac` | 0.5277 | 0.2246 | higher_feature_higher_error | 0.2480 | 0.2425 |
| `cc_area_max_frac` | 0.5268 | 0.2278 | higher_feature_higher_error | 0.2327 | 0.2276 |
| `dir_diag_down_frac` | 0.5265 | 0.2423 | lower_feature_higher_error | 0.2516 | 0.2460 |

## 6. Strict interpretation

Correlations are weak; graph features do not strongly explain CER by themselves.

Single-feature high-error detection is weak; multi-feature diagnostics or gold graph quality may be needed.

This analysis is diagnostic only. It does not prove that graph features improve recognition. It tests whether graph-derived quality and structural measures are informative about failure cases.