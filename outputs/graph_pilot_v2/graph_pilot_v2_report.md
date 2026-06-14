# Graph pilot v2 report

## Status

```text
graph_pilot_v2 is the main expanded graph pilot.
It includes IAM, Cyrillic Handwriting, HKR Words, School Notebooks, HWR200, and HKR Forms.
```

## Pilot composition

```json
{
  "num_records": 670,
  "by_dataset": {
    "iam": 100,
    "cyrillic_handwriting": 150,
    "hkr_words": 150,
    "school_notebooks": 150,
    "hwr200": 90,
    "hkr_forms": 30
  },
  "by_level": {
    "line": 100,
    "word": 348,
    "phrase": 102,
    "page_from_document_condition": 90,
    "form_page": 30
  },
  "by_split": {
    "train": 593,
    "val": 77
  },
  "school_categories": {
    "pupil_text": 100,
    "pupil_comment": 30,
    "teacher_comment": 20
  },
  "hwr200_conditions": {
    "scan": 30,
    "photo_light": 30,
    "photo_dark": 30
  }
}
```

## Graph-builder summary

```json
{
  "num_binary_summaries": 670,
  "num_selected_summaries": 60,
  "num_graphs_built": 88,
  "num_skipped": 2,
  "built_by_dataset": {
    "iam": 10,
    "cyrillic_handwriting": 10,
    "hkr_words": 10,
    "school_notebooks": 20,
    "hwr200": 18,
    "hkr_forms": 20
  },
  "built_by_method": {
    "otsu": 59,
    "sauvola": 10,
    "otsu_gridless": 19
  }
}
```

## Primary graph variants

```json
{
  "iam": "otsu",
  "cyrillic_handwriting": "otsu",
  "hkr_words": "otsu",
  "school_notebooks": "sauvola",
  "hwr200": "otsu_gridless",
  "hkr_forms": "otsu_gridless"
}
```

## Methodological decisions

- IAM: primary graph variant is `otsu`.

- Cyrillic Handwriting: primary graph variant is `otsu`.

- HKR Words: primary graph variant is `otsu`.

- School Notebooks: primary graph variant is `sauvola`; Otsu often over-selects polygon/background foreground.

- HWR200: `otsu_gridless` is a diagnostic page-level stress-test variant, not a clean canonical graph setting.

- HKR Forms: `otsu_gridless` is a diagnostic page/form stress-test variant, not a clean canonical graph setting.

- `adaptive_gaussian` is not used for graph-builder because it tends to increase skeleton noise.

- Component filtering is not used by default on word/line/polygon crops.

- `min_skel=8` is only a diagnostic page-noise filter for page/form datasets.

- Raw node/edge/component counts must not be compared across word/line/page levels without normalization.


## Binary/skeleton statistics

### cyrillic_handwriting / adaptive_gaussian
```text
n: 150
foreground_ratio_mean: 0.163963
foreground_pixels_mean: 2342.713
skeleton_pixels_mean: 804.547
grid_removed_ratio_mean: None
```

### cyrillic_handwriting / otsu
```text
n: 150
foreground_ratio_mean: 0.14256
foreground_pixels_mean: 2057.58
skeleton_pixels_mean: 624.153
grid_removed_ratio_mean: None
```

### cyrillic_handwriting / sauvola
```text
n: 150
foreground_ratio_mean: 0.131624
foreground_pixels_mean: 1813.26
skeleton_pixels_mean: 591.66
grid_removed_ratio_mean: None
```

### hkr_forms / adaptive_gaussian
```text
n: 30
foreground_ratio_mean: 0.082108
foreground_pixels_mean: 334411.067
skeleton_pixels_mean: 101711.4
grid_removed_ratio_mean: None
```

### hkr_forms / otsu
```text
n: 30
foreground_ratio_mean: 0.070306
foreground_pixels_mean: 286341.167
skeleton_pixels_mean: 96054.7
grid_removed_ratio_mean: None
```

### hkr_forms / otsu_gridless
```text
n: 30
foreground_ratio_mean: 0.028517
foreground_pixels_mean: 116145.733
skeleton_pixels_mean: 52741.467
grid_removed_ratio_mean: 0.597046
```

### hkr_forms / sauvola
```text
n: 30
foreground_ratio_mean: 0.075798
foreground_pixels_mean: 308710.633
skeleton_pixels_mean: 98911.467
grid_removed_ratio_mean: None
```

### hkr_forms / sauvola_gridless
```text
n: 30
foreground_ratio_mean: 0.032152
foreground_pixels_mean: 130949.7
skeleton_pixels_mean: 55338.633
grid_removed_ratio_mean: 0.578451
```

### hkr_words / adaptive_gaussian
```text
n: 150
foreground_ratio_mean: 0.139335
foreground_pixels_mean: 4117.953
skeleton_pixels_mean: 995.153
grid_removed_ratio_mean: None
```

### hkr_words / otsu
```text
n: 150
foreground_ratio_mean: 0.130777
foreground_pixels_mean: 3858.827
skeleton_pixels_mean: 953.973
grid_removed_ratio_mean: None
```

### hkr_words / sauvola
```text
n: 150
foreground_ratio_mean: 0.121467
foreground_pixels_mean: 3596.327
skeleton_pixels_mean: 944.147
grid_removed_ratio_mean: None
```

### hwr200 / adaptive_gaussian
```text
n: 90
foreground_ratio_mean: 0.123475
foreground_pixels_mean: 352854.378
skeleton_pixels_mean: 128300.556
grid_removed_ratio_mean: None
```

### hwr200 / otsu
```text
n: 90
foreground_ratio_mean: 0.226719
foreground_pixels_mean: 666073.5
skeleton_pixels_mean: 107651.478
grid_removed_ratio_mean: None
```

### hwr200 / otsu_gridless
```text
n: 90
foreground_ratio_mean: 0.074057
foreground_pixels_mean: 213298.444
skeleton_pixels_mean: 79531.933
grid_removed_ratio_mean: 0.472764
```

### hwr200 / sauvola
```text
n: 90
foreground_ratio_mean: 0.102495
foreground_pixels_mean: 297052.922
skeleton_pixels_mean: 106639.233
grid_removed_ratio_mean: None
```

### hwr200 / sauvola_gridless
```text
n: 90
foreground_ratio_mean: 0.085845
foreground_pixels_mean: 249393.933
skeleton_pixels_mean: 98792.144
grid_removed_ratio_mean: 0.158833
```

### iam / adaptive_gaussian
```text
n: 100
foreground_ratio_mean: 0.092583
foreground_pixels_mean: 17485.88
skeleton_pixels_mean: 3262.31
grid_removed_ratio_mean: None
```

### iam / otsu
```text
n: 100
foreground_ratio_mean: 0.088178
foreground_pixels_mean: 16710.48
skeleton_pixels_mean: 3012.79
grid_removed_ratio_mean: None
```

### iam / sauvola
```text
n: 100
foreground_ratio_mean: 0.090668
foreground_pixels_mean: 17157.6
skeleton_pixels_mean: 3043.08
grid_removed_ratio_mean: None
```

### school_notebooks / adaptive_gaussian
```text
n: 150
foreground_ratio_mean: 0.202989
foreground_pixels_mean: 5378.847
skeleton_pixels_mean: 1216.72
grid_removed_ratio_mean: None
```

### school_notebooks / otsu
```text
n: 150
foreground_ratio_mean: 0.36322
foreground_pixels_mean: 10316.34
skeleton_pixels_mean: 734.093
grid_removed_ratio_mean: None
```

### school_notebooks / sauvola
```text
n: 150
foreground_ratio_mean: 0.151596
foreground_pixels_mean: 4063.02
skeleton_pixels_mean: 949.98
grid_removed_ratio_mean: None
```

## Graph statistics

### cyrillic_handwriting / otsu
```text
n: 10
node_count_mean: 48.1
edge_count_mean: 45.9
component_count_mean: 13.0
junction_count_mean: 17.2
endpoint_count_mean: 30.9
skeleton_pixels_mean: 549.0
```

### hkr_forms / otsu
```text
n: 10
node_count_mean: 5217.7
edge_count_mean: 4645.2
component_count_mean: 1606.0
junction_count_mean: 1610.6
endpoint_count_mean: 3607.1
skeleton_pixels_mean: 92501.9
```

### hkr_forms / otsu_gridless
```text
n: 10
node_count_mean: 8004.8
edge_count_mean: 6683.0
component_count_mean: 4064.3
junction_count_mean: 1373.5
endpoint_count_mean: 6631.3
skeleton_pixels_mean: 50183.7
```

### hkr_words / otsu
```text
n: 10
node_count_mean: 45.9
edge_count_mean: 43.8
component_count_mean: 8.4
junction_count_mean: 19.8
endpoint_count_mean: 26.1
skeleton_pixels_mean: 785.2
```

### hwr200 / otsu
```text
n: 9
node_count_mean: 9787.333
edge_count_mean: 10372.889
component_count_mean: 3680.889
junction_count_mean: 3577.778
endpoint_count_mean: 6209.556
skeleton_pixels_mean: 96863.222
```

### hwr200 / otsu_gridless
```text
n: 9
node_count_mean: 10656.667
edge_count_mean: 9684.111
component_count_mean: 5275.0
junction_count_mean: 2314.222
endpoint_count_mean: 8342.444
skeleton_pixels_mean: 75214.444
```

### iam / otsu
```text
n: 10
node_count_mean: 128.0
edge_count_mean: 121.1
component_count_mean: 27.0
junction_count_mean: 51.6
endpoint_count_mean: 76.4
skeleton_pixels_mean: 2663.6
```

### school_notebooks / otsu
```text
n: 10
node_count_mean: 19.8
edge_count_mean: 24.0
component_count_mean: 7.7
junction_count_mean: 8.7
endpoint_count_mean: 11.1
skeleton_pixels_mean: 610.8
```

### school_notebooks / sauvola
```text
n: 10
node_count_mean: 88.4
edge_count_mean: 87.4
component_count_mean: 26.9
junction_count_mean: 31.8
endpoint_count_mean: 56.6
skeleton_pixels_mean: 1194.4
```

## Binary warnings

```text
hwr200                   otsu               hwr200_page                        90
hwr200                   otsu_gridless      hwr200_page                        90
hwr200                   sauvola            hwr200_page                        90
hwr200                   sauvola_gridless   hwr200_page                        90
hwr200                   adaptive_gaussian  hwr200_page                        90
school_notebooks         otsu               too_high_foreground_ratio          49
hwr200                   otsu               large_page_scaled                  39
hwr200                   otsu_gridless      large_page_scaled                  39
hwr200                   sauvola            large_page_scaled                  39
hwr200                   sauvola_gridless   large_page_scaled                  39
hwr200                   adaptive_gaussian  large_page_scaled                  39
hkr_forms                otsu               large_page_scaled                  30
hkr_forms                otsu               hkr_forms_page                     30
hkr_forms                otsu               hkr_possible_form_grid             30
hkr_forms                otsu_gridless      large_page_scaled                  30
hkr_forms                otsu_gridless      hkr_forms_page                     30
hkr_forms                otsu_gridless      hkr_possible_form_grid             30
hkr_forms                sauvola            large_page_scaled                  30
hkr_forms                sauvola            hkr_forms_page                     30
hkr_forms                sauvola            hkr_possible_form_grid             30
hkr_forms                sauvola_gridless   large_page_scaled                  30
hkr_forms                sauvola_gridless   hkr_forms_page                     30
hkr_forms                sauvola_gridless   hkr_possible_form_grid             30
hkr_forms                adaptive_gaussian  large_page_scaled                  30
hkr_forms                adaptive_gaussian  hkr_forms_page                     30
hkr_forms                adaptive_gaussian  hkr_possible_form_grid             30
hwr200                   otsu               too_high_foreground_ratio          10
```

## Graph warnings

```text
hkr_forms                otsu               hkr_forms_page                     10
hkr_forms                otsu               hkr_possible_form_grid             10
hkr_forms                otsu               large_page_scaled                  10
hkr_forms                otsu               too_many_components                10
hkr_forms                otsu               too_many_junctions                 10
hkr_forms                otsu_gridless      hkr_forms_page                     10
hkr_forms                otsu_gridless      hkr_possible_form_grid             10
hkr_forms                otsu_gridless      large_page_scaled                  10
hkr_forms                otsu_gridless      too_many_components                10
hkr_forms                otsu_gridless      too_many_junctions                 10
hwr200                   otsu               hwr200_page                        9
hwr200                   otsu               too_many_components                9
hwr200                   otsu               too_many_junctions                 9
hwr200                   otsu_gridless      hwr200_page                        9
hwr200                   otsu_gridless      too_many_components                9
hwr200                   otsu_gridless      too_many_junctions                 9
school_notebooks         otsu               too_high_foreground_ratio          4
hwr200                   otsu               large_page_scaled                  3
hwr200                   otsu_gridless      large_page_scaled                  3
school_notebooks         otsu               no_special_nodes_detected          2
hwr200                   otsu               too_high_foreground_ratio          1
hwr200                   otsu               too_many_short_branches            1
hwr200                   otsu_gridless      too_many_short_branches            1
```

## Skipped graph builds

```json
[
  {
    "pilot_id": "pilot2_hwr200_0006",
    "dataset": "hwr200",
    "method": "otsu",
    "reason": "too_many_skeleton_pixels",
    "skeleton_pixels": 304671
  },
  {
    "pilot_id": "pilot2_hwr200_0006",
    "dataset": "hwr200",
    "method": "otsu_gridless",
    "reason": "too_many_skeleton_pixels",
    "skeleton_pixels": 189625
  }
]
```

## Interpretation

- Clean crop/line datasets already produce usable graph overlays.

- Page-level HWR200/HKR Forms remain stress-test datasets because grid/form/background structures dominate graph complexity.

- School Notebooks is a valid polygon crop graph dataset, but its masked crop geometry makes local thresholding preferable.

- The next step is to export normalized graph quality metrics, not only raw graph counts.
