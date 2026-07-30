# X-Aligned HI-CSG-R Feature Audit v1

Overall status: **PASS**

| split | expected | audited | failures | zero bins | T range |
|---|---:|---:|---:|---:|---|
| `train` | 39998 | 39998 | 0 | 0.0367 | 9..400 |
| `val` | 6000 | 6000 | 0 | 0.0449 | 16..404 |
| `test` | 5563 | 5563 | 0 | 0.0442 | 16..297 |

## Train Feature Distributions

| feature | min | max | mean | std | zero fraction |
|---|---:|---:|---:|---:|---:|
| `ink_fraction` | 0 | 1 | 0.14038 | 0.0857722 | 0.0369 |
| `skeleton_density` | 0 | 0.37207 | 0.0496292 | 0.0317752 | 0.0427 |
| `edge_length_density` | 0 | 0.276504 | 0.0486478 | 0.0287571 | 0.0440 |
| `stroke_width_mean` | 0 | 115.811 | 2.73326 | 1.46413 | 0.0441 |
| `stroke_width_std` | 0 | 16.9363 | 0.636803 | 0.439602 | 0.0829 |
| `curvature_mean` | 0 | 1.79745 | 0.362441 | 0.167267 | 0.0604 |
| `orientation_horizontal` | 0 | 1 | 0.332279 | 0.214349 | 0.0748 |
| `orientation_vertical` | 0 | 1 | 0.156742 | 0.133444 | 0.1601 |
| `orientation_diag_pos` | 0 | 1 | 0.10229 | 0.104226 | 0.2068 |
| `orientation_diag_neg` | 0 | 1 | 0.325357 | 0.192203 | 0.0796 |
| `node_density` | 0 | 425 | 33.9996 | 32.5424 | 0.1269 |
| `endpoint_density` | 0 | 391.667 | 21.149 | 23.98 | 0.2387 |
| `junction_density` | 0 | 141.667 | 12.8506 | 14.2017 | 0.3219 |
| `loop_edge_fraction` | 0 | 1 | 0.0031383 | 0.0450397 | 0.9920 |
| `component_count_norm` | 0 | 237.5 | 38.2456 | 24.8109 | 0.0440 |
| `short_branch_fraction` | 0 | 1 | 0.0449345 | 0.086929 | 0.5846 |
| `boundary_crossings_norm` | 0 | 9.125 | 1.63993 | 1.06262 | 0.0522 |
| `ambiguous_edge_fraction` | 0 | 0.181483 | 1.86339e-07 | 0.00016317 | 1.0000 |
| `graph_occupancy` | 0 | 1 | 0.920937 | 0.238809 | 0.0424 |
| `warning_density` | 0 | 1 | 0.0480729 | 0.0971654 | 0.5802 |

## Count Consistency

| split | quantity | max abs delta | mean abs delta |
|---|---|---:|---:|
| `train` | `node` | 0 | 0 |
| `train` | `endpoint` | 0 | 0 |
| `train` | `junction` | 0 | 0 |
| `train` | `edge_length` | 2.63753e-11 | 7.95398e-13 |
| `train` | `edge_length_record` | 2.22453e-05 | 2.70436e-06 |
| `val` | `node` | 0 | 0 |
| `val` | `endpoint` | 0 | 0 |
| `val` | `junction` | 0 | 0 |
| `val` | `edge_length` | 2.95586e-12 | 2.27085e-13 |
| `val` | `edge_length_record` | 1.51836e-05 | 2.39316e-06 |
| `test` | `node` | 0 | 0 |
| `test` | `endpoint` | 0 | 0 |
| `test` | `junction` | 0 | 0 |
| `test` | `edge_length` | 5.00222e-12 | 2.24672e-13 |
| `test` | `edge_length_record` | 1.53571e-05 | 2.49434e-06 |

Normalizer provenance is accepted only when its stored train-manifest SHA256 matches the first audited manifest.
