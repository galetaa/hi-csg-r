# Graph pilot report

## Summary

```text
binary/skeleton records: 370
graphs built: 56
graphs skipped: 4
```

## Built by dataset

```json
{
  "iam": 10,
  "cyrillic_handwriting": 10,
  "hwr200": 16,
  "hkr_forms": 20
}
```

## Built by method

```json
{
  "otsu": 36,
  "otsu_gridless": 20
}
```

## Binary/skeleton statistics

### cyrillic_handwriting / adaptive_gaussian
```text
n: 150
foreground_ratio_mean: 0.15646036671329638
foreground_pixels_mean: 2531.8933333333334
skeleton_pixels_mean: 841.7266666666667
grid_removed_ratio_mean: None
```

### cyrillic_handwriting / otsu
```text
n: 150
foreground_ratio_mean: 0.1389288243299517
foreground_pixels_mean: 2228.233333333333
skeleton_pixels_mean: 637.64
grid_removed_ratio_mean: None
```

### cyrillic_handwriting / sauvola
```text
n: 150
foreground_ratio_mean: 0.1230553263260974
foreground_pixels_mean: 1883.42
skeleton_pixels_mean: 602.6933333333334
grid_removed_ratio_mean: None
```

### hkr_forms / adaptive_gaussian
```text
n: 30
foreground_ratio_mean: 0.08406364172068355
foreground_pixels_mean: 342374.4
skeleton_pixels_mean: 103871.5
grid_removed_ratio_mean: None
```

### hkr_forms / otsu
```text
n: 30
foreground_ratio_mean: 0.07265601879133111
foreground_pixels_mean: 295913.43333333335
skeleton_pixels_mean: 98313.53333333334
grid_removed_ratio_mean: None
```

### hkr_forms / otsu_gridless
```text
n: 30
foreground_ratio_mean: 0.02947985824657893
foreground_pixels_mean: 120065.56666666667
skeleton_pixels_mean: 54478.0
grid_removed_ratio_mean: 0.5955689781871432
```

### hkr_forms / sauvola
```text
n: 30
foreground_ratio_mean: 0.0780847083087802
foreground_pixels_mean: 318023.4
skeleton_pixels_mean: 100881.5
grid_removed_ratio_mean: None
```

### hkr_forms / sauvola_gridless
```text
n: 30
foreground_ratio_mean: 0.03328574117724088
foreground_pixels_mean: 135566.16666666666
skeleton_pixels_mean: 57133.0
grid_removed_ratio_mean: 0.5744057010324853
```

### hwr200 / adaptive_gaussian
```text
n: 90
foreground_ratio_mean: 0.1350840890047925
foreground_pixels_mean: 341688.4111111111
skeleton_pixels_mean: 120804.62222222223
grid_removed_ratio_mean: None
```

### hwr200 / otsu
```text
n: 90
foreground_ratio_mean: 0.2103705640264581
foreground_pixels_mean: 555894.5777777778
skeleton_pixels_mean: 103874.86666666667
grid_removed_ratio_mean: None
```

### hwr200 / otsu_gridless
```text
n: 90
foreground_ratio_mean: 0.09066583751433677
foreground_pixels_mean: 227163.73333333334
skeleton_pixels_mean: 86231.74444444444
grid_removed_ratio_mean: 0.4138323150916966
```

### hwr200 / sauvola
```text
n: 90
foreground_ratio_mean: 0.11511184250389728
foreground_pixels_mean: 292298.0
skeleton_pixels_mean: 103180.78888888888
grid_removed_ratio_mean: None
```

### hwr200 / sauvola_gridless
```text
n: 90
foreground_ratio_mean: 0.09884061973396539
foreground_pixels_mean: 245254.76666666666
skeleton_pixels_mean: 94004.74444444444
grid_removed_ratio_mean: 0.12796686325601692
```

### iam / adaptive_gaussian
```text
n: 100
foreground_ratio_mean: 0.09291913976564714
foreground_pixels_mean: 18717.39
skeleton_pixels_mean: 3487.62
grid_removed_ratio_mean: None
```

### iam / otsu
```text
n: 100
foreground_ratio_mean: 0.0884964037065596
foreground_pixels_mean: 17897.16
skeleton_pixels_mean: 3170.23
grid_removed_ratio_mean: None
```

### iam / sauvola
```text
n: 100
foreground_ratio_mean: 0.09044512746791884
foreground_pixels_mean: 18227.63
skeleton_pixels_mean: 3194.53
grid_removed_ratio_mean: None
```

## Graph statistics

### cyrillic_handwriting / otsu
```text
n: 10
node_count_mean: 69.1
edge_count_mean: 70.8
component_count_mean: 19.7
junction_count_mean: 25.6
endpoint_count_mean: 43.5
skeleton_pixels_mean: 617.0
```

### hkr_forms / otsu
```text
n: 10
node_count_mean: 5444.7
edge_count_mean: 4826.3
component_count_mean: 1538.3
junction_count_mean: 1724.7
endpoint_count_mean: 3720.0
skeleton_pixels_mean: 101015.6
```

### hkr_forms / otsu_gridless
```text
n: 10
node_count_mean: 8608.8
edge_count_mean: 7215.2
component_count_mean: 4323.6
junction_count_mean: 1515.3
endpoint_count_mean: 7093.5
skeleton_pixels_mean: 56348.6
```

### hwr200 / otsu
```text
n: 6
node_count_mean: 7846.5
edge_count_mean: 8939.5
component_count_mean: 2464.8333333333335
junction_count_mean: 3022.3333333333335
endpoint_count_mean: 4824.166666666667
skeleton_pixels_mean: 81994.0
```

### hwr200 / otsu_gridless
```text
n: 10
node_count_mean: 16935.4
edge_count_mean: 16342.3
component_count_mean: 10614.5
junction_count_mean: 2634.1
endpoint_count_mean: 14301.3
skeleton_pixels_mean: 99567.2
```

### iam / otsu
```text
n: 10
node_count_mean: 161.8
edge_count_mean: 159.8
component_count_mean: 24.0
junction_count_mean: 74.1
endpoint_count_mean: 87.7
skeleton_pixels_mean: 3393.4
```

## Warnings

```text
hwr200                   otsu_gridless      hwr200_page                      10
hwr200                   otsu_gridless      too_many_components              10
hwr200                   otsu_gridless      too_many_junctions               10
hkr_forms                otsu               hkr_forms_page                   10
hkr_forms                otsu               hkr_possible_form_grid           10
hkr_forms                otsu               large_page_scaled                10
hkr_forms                otsu               too_many_components              10
hkr_forms                otsu               too_many_junctions               10
hkr_forms                otsu_gridless      hkr_forms_page                   10
hkr_forms                otsu_gridless      hkr_possible_form_grid           10
hkr_forms                otsu_gridless      large_page_scaled                10
hkr_forms                otsu_gridless      too_many_components              10
hkr_forms                otsu_gridless      too_many_junctions               10
hwr200                   otsu               hwr200_page                      6
hwr200                   otsu               too_many_components              6
hwr200                   otsu               too_many_junctions               6
hwr200                   otsu_gridless      large_page_scaled                4
hwr200                   otsu_gridless      too_many_short_branches          2
hwr200                   otsu               large_page_scaled                1
hwr200                   otsu               too_many_short_branches          1
```

## Skipped graphs

```json
[
  {
    "pilot_id": "pilot_hwr200_0001",
    "dataset": "hwr200",
    "method": "otsu",
    "reason": "too_many_skeleton_pixels",
    "skeleton_pixels": 215376
  },
  {
    "pilot_id": "pilot_hwr200_0002",
    "dataset": "hwr200",
    "method": "otsu",
    "reason": "too_many_skeleton_pixels",
    "skeleton_pixels": 218936
  },
  {
    "pilot_id": "pilot_hwr200_0003",
    "dataset": "hwr200",
    "method": "otsu",
    "reason": "too_many_skeleton_pixels",
    "skeleton_pixels": 193879
  },
  {
    "pilot_id": "pilot_hwr200_0008",
    "dataset": "hwr200",
    "method": "otsu",
    "reason": "too_many_skeleton_pixels",
    "skeleton_pixels": 182630
  }
]
```

## Current decisions

- IAM: `otsu` is the primary graph-builder variant.

- Cyrillic Handwriting: `otsu` is the primary graph-builder variant.

- HWR200: `otsu` and `otsu_gridless` are diagnostic variants; full-page graph is a stress-test.

- HKR Forms: `otsu` and `otsu_gridless` are diagnostic variants; form/grid background remains a major factor.

- `adaptive_gaussian` is not used for graph-builder pilot because it tends to produce more skeleton noise.

- Raw graph counts must not be compared across word/line/page levels without normalization.
