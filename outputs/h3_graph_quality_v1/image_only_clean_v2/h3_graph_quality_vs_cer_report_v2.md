# H3 graph diagnostics report — v2

## 1. Strict interpretation

```text
useful_multifeature_h3_signal
```

This v2 analysis adds stratification and multifeature cross-validated high-error detection.

## 2. Best multifeature high-error detectors

| group mode | group | feature set | n | ROC-AUC | PR-AUC | PR-AUC lift | top20 precision |
|---|---|---|---:|---:|---:|---:|---:|
| `dataset_category` | `hkr_words|word|unknown` | `structural_core` | 1090 | 0.6733 | 0.3466 | 1.636 | 0.3761 |
| `dataset_category` | `hkr_words|word|unknown` | `all_non_geometry` | 1090 | 0.6733 | 0.3466 | 1.636 | 0.3761 |
| `dataset_level` | `hkr_words|word` | `structural_core` | 1090 | 0.6733 | 0.3466 | 1.636 | 0.3761 |
| `dataset_level` | `hkr_words|word` | `all_non_geometry` | 1090 | 0.6733 | 0.3466 | 1.636 | 0.3761 |
| `dataset_category` | `hkr_words|word|unknown` | `all_features` | 1090 | 0.6677 | 0.3440 | 1.623 | 0.3807 |
| `dataset_level` | `hkr_words|word` | `all_features` | 1090 | 0.6677 | 0.3440 | 1.623 | 0.3807 |
| `dataset` | `hkr_words` | `structural_core` | 2000 | 0.6549 | 0.3286 | 1.595 | 0.3725 |
| `dataset` | `hkr_words` | `all_non_geometry` | 2000 | 0.6549 | 0.3286 | 1.595 | 0.3725 |
| `dataset` | `hkr_words` | `all_features` | 2000 | 0.6501 | 0.3254 | 1.580 | 0.3600 |
| `dataset_category` | `cyrillic_handwriting|phrase|unknown` | `structural_core` | 463 | 0.6369 | 0.3394 | 1.672 | 0.3333 |
| `dataset_category` | `cyrillic_handwriting|phrase|unknown` | `all_non_geometry` | 463 | 0.6369 | 0.3394 | 1.672 | 0.3333 |
| `dataset_level` | `cyrillic_handwriting|phrase` | `structural_core` | 463 | 0.6369 | 0.3394 | 1.672 | 0.3333 |
| `dataset_level` | `cyrillic_handwriting|phrase` | `all_non_geometry` | 463 | 0.6369 | 0.3394 | 1.672 | 0.3333 |
| `dataset_category` | `cyrillic_handwriting|phrase|unknown` | `all_features` | 463 | 0.6217 | 0.3214 | 1.583 | 0.3763 |
| `dataset_level` | `cyrillic_handwriting|phrase` | `all_features` | 463 | 0.6217 | 0.3214 | 1.583 | 0.3763 |
| `global` | `global` | `all_features` | 5563 | 0.6185 | 0.2992 | 1.463 | 0.3217 |
| `dataset_category` | `cyrillic_handwriting|word|unknown` | `structural_core` | 1100 | 0.6172 | 0.3044 | 1.401 | 0.3318 |
| `dataset_category` | `cyrillic_handwriting|word|unknown` | `all_non_geometry` | 1100 | 0.6172 | 0.3044 | 1.401 | 0.3318 |
| `dataset_level` | `cyrillic_handwriting|word` | `structural_core` | 1100 | 0.6172 | 0.3044 | 1.401 | 0.3318 |
| `dataset_level` | `cyrillic_handwriting|word` | `all_non_geometry` | 1100 | 0.6172 | 0.3044 | 1.401 | 0.3318 |
| `global` | `global` | `structural_core` | 5563 | 0.6136 | 0.2944 | 1.439 | 0.3136 |
| `global` | `global` | `all_non_geometry` | 5563 | 0.6136 | 0.2944 | 1.439 | 0.3136 |
| `dataset` | `cyrillic_handwriting` | `structural_core` | 1563 | 0.6043 | 0.3171 | 1.329 | 0.3387 |
| `dataset` | `cyrillic_handwriting` | `all_non_geometry` | 1563 | 0.6043 | 0.3171 | 1.329 | 0.3387 |
| `dataset_category` | `hkr_words|phrase|unknown` | `structural_core` | 910 | 0.6025 | 0.2850 | 1.409 | 0.3242 |
| `dataset_category` | `hkr_words|phrase|unknown` | `all_non_geometry` | 910 | 0.6025 | 0.2850 | 1.409 | 0.3242 |
| `dataset_level` | `hkr_words|phrase` | `structural_core` | 910 | 0.6025 | 0.2850 | 1.409 | 0.3242 |
| `dataset_level` | `hkr_words|phrase` | `all_non_geometry` | 910 | 0.6025 | 0.2850 | 1.409 | 0.3242 |
| `dataset_category` | `cyrillic_handwriting|word|unknown` | `all_features` | 1100 | 0.6001 | 0.2890 | 1.330 | 0.3045 |
| `dataset_level` | `cyrillic_handwriting|word` | `all_features` | 1100 | 0.6001 | 0.2890 | 1.330 | 0.3045 |

## 3. Best stratified correlations

| group mode | group | feature | n | Spearman r | abs r |
|---|---|---|---:|---:|---:|
| `dataset` | `cyrillic_handwriting` | `aspect_ratio` | 1563 | 0.1827 | 0.1827 |
| `dataset` | `cyrillic_handwriting` | `width` | 1563 | 0.1827 | 0.1827 |
| `dataset_category` | `hkr_words|word|unknown` | `dir_v_frac` | 1090 | -0.1814 | 0.1814 |
| `dataset_level` | `hkr_words|word` | `dir_v_frac` | 1090 | -0.1814 | 0.1814 |
| `dataset` | `cyrillic_handwriting` | `graph_endpoint_count` | 1563 | 0.1795 | 0.1795 |
| `dataset` | `cyrillic_handwriting` | `cc_count` | 1563 | 0.1744 | 0.1744 |
| `dataset` | `cyrillic_handwriting` | `skel_components` | 1563 | 0.1744 | 0.1744 |
| `dataset_category` | `cyrillic_handwriting|phrase|unknown` | `cc_count` | 463 | 0.1626 | 0.1626 |
| `dataset_category` | `cyrillic_handwriting|phrase|unknown` | `skel_components` | 463 | 0.1626 | 0.1626 |
| `dataset_level` | `cyrillic_handwriting|phrase` | `cc_count` | 463 | 0.1626 | 0.1626 |
| `dataset_level` | `cyrillic_handwriting|phrase` | `skel_components` | 463 | 0.1626 | 0.1626 |
| `dataset` | `cyrillic_handwriting` | `bbox_x0_frac` | 1563 | -0.1590 | 0.1590 |
| `dataset` | `cyrillic_handwriting` | `bbox_w_frac` | 1563 | 0.1578 | 0.1578 |
| `dataset` | `cyrillic_handwriting` | `graph_nodes` | 1563 | 0.1554 | 0.1554 |
| `dataset` | `cyrillic_handwriting` | `skel_pixels` | 1563 | 0.1554 | 0.1554 |
| `dataset` | `cyrillic_handwriting` | `graph_edges_8n` | 1563 | 0.1527 | 0.1527 |
| `dataset_category` | `cyrillic_handwriting|word|unknown` | `graph_isolated_count` | 1100 | 0.1521 | 0.1521 |
| `dataset_level` | `cyrillic_handwriting|word` | `graph_isolated_count` | 1100 | 0.1521 | 0.1521 |
| `dataset_category` | `cyrillic_handwriting|word|unknown` | `degree_hist_0` | 1100 | 0.1510 | 0.1510 |
| `dataset_level` | `cyrillic_handwriting|word` | `degree_hist_0` | 1100 | 0.1510 | 0.1510 |
| `dataset` | `cyrillic_handwriting` | `graph_isolated_count` | 1563 | 0.1502 | 0.1502 |
| `dataset` | `hkr_words` | `dir_v_frac` | 2000 | -0.1486 | 0.1486 |
| `dataset_category` | `cyrillic_handwriting|word|unknown` | `bbox_w_frac` | 1100 | 0.1466 | 0.1466 |
| `dataset_level` | `cyrillic_handwriting|word` | `bbox_w_frac` | 1100 | 0.1466 | 0.1466 |
| `dataset` | `cyrillic_handwriting` | `degree_hist_0` | 1563 | 0.1443 | 0.1443 |
| `dataset_category` | `cyrillic_handwriting|word|unknown` | `bbox_x0_frac` | 1100 | -0.1409 | 0.1409 |
| `dataset_level` | `cyrillic_handwriting|word` | `bbox_x0_frac` | 1100 | -0.1409 | 0.1409 |
| `dataset` | `cyrillic_handwriting` | `bbox_area_frac` | 1563 | 0.1387 | 0.1387 |
| `dataset` | `cyrillic_handwriting` | `dir_v_frac` | 1563 | -0.1377 | 0.1377 |
| `dataset_category` | `hkr_words|word|unknown` | `dir_h_frac` | 1090 | 0.1333 | 0.1333 |

## 4. Methodological note

Geometry-control features are reported as controls. If geometry features dominate, that is not strong evidence for graph quality. The primary evidence should come from `quality_only`, `structural_core`, or `all_non_geometry` feature sets.