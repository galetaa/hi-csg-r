# Stage 2 checkpoint report — HI-CSG-R graph pilot v2

## 1. Purpose

Stage 2 established the first reproducible HI-CSG-R graph extraction pipeline for offline handwritten images. The stage moved from preprocessed handwriting images to binary masks, skeletons, pixel graphs, canonical graph JSON files, visual overlays, diagnostics, normalized graph metrics, and clean/stress graph subsets.

## 2. Current graph pipeline

```text
feature image
→ binarization
→ skeletonization
→ pixel graph
→ node detection
→ junction clustering
→ edge tracing
→ graph.json
→ overlay
→ normalized graph metrics
```

## 3. Datasets used in graph pilot v2

```text
IAM                  → English line-level clean graph
Cyrillic Handwriting → Russian word/phrase clean graph
HKR Words            → Russian/Kazakh Cyrillic word/phrase clean graph
School Notebooks     → Russian polygon crop clean graph
HWR200               → scan/photo/dark page-level stress-test
HKR Forms            → form/page stress-test
```

## 4. Primary graph variants

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

## 5. Methodological decisions

- IAM uses `otsu` as the primary graph variant.

- Cyrillic Handwriting uses `otsu` as the primary graph variant.

- HKR Words uses `otsu` as the primary graph variant.

- School Notebooks uses `sauvola` as the primary graph variant because Otsu often over-selects polygon/background foreground.

- HWR200 uses `otsu_gridless` only as a page-level stress-test variant.

- HKR Forms uses `otsu_gridless` only as a page/form stress-test variant.

- `adaptive_gaussian` is not used for graph-builder because it tends to increase skeleton noise.

- Component filtering is not used by default on word/line/polygon crops.

- `min_skel=8` is only a diagnostic page-noise filter for page/form datasets.

- Clean graph metrics and page stress metrics must not be mixed in one evaluation table.

- `mean_width_proxy` is a stroke-width proxy, not true pen pressure.


## 6. Clean graph subset v1

Clean graph subset contains only primary graph variants from crop/line datasets with zero warning risk.

```json
{
  "path": "outputs/graph_pilot_v2/clean_graph_subset_v1.jsonl",
  "count": 40,
  "by_dataset": {
    "iam": 10,
    "cyrillic_handwriting": 10,
    "hkr_words": 10,
    "school_notebooks": 10
  },
  "by_method": {
    "otsu": 30,
    "sauvola": 10
  },
  "metrics_by_dataset": {
    "cyrillic_handwriting": {
      "n": 10,
      "skeleton_pixels_mean": 549.0,
      "nodes_per_1k_skeleton_mean": 86.09895438276996,
      "components_per_1k_skeleton_mean": 24.103980378125527,
      "junctions_per_1k_skeleton_mean": 30.19118999592855,
      "warning_risk_score_mean": 0.0
    },
    "hkr_words": {
      "n": 10,
      "skeleton_pixels_mean": 785.2,
      "nodes_per_1k_skeleton_mean": 60.45030030393144,
      "components_per_1k_skeleton_mean": 11.980519574186504,
      "junctions_per_1k_skeleton_mean": 25.000400827767585,
      "warning_risk_score_mean": 0.0
    },
    "iam": {
      "n": 10,
      "skeleton_pixels_mean": 2663.6,
      "nodes_per_1k_skeleton_mean": 47.852255385268634,
      "components_per_1k_skeleton_mean": 10.488746118127498,
      "junctions_per_1k_skeleton_mean": 18.597176782287036,
      "warning_risk_score_mean": 0.0
    },
    "school_notebooks": {
      "n": 10,
      "skeleton_pixels_mean": 1194.4,
      "nodes_per_1k_skeleton_mean": 79.3434840977559,
      "components_per_1k_skeleton_mean": 26.19138656049168,
      "junctions_per_1k_skeleton_mean": 27.086392515990887,
      "warning_risk_score_mean": 0.0
    }
  }
}
```

## 7. Page stress graph subset v1

Page stress subset contains HWR200 and HKR Forms primary stress variants. These graphs are useful for robustness and failure analysis, but they are not treated as clean canonical stroke graphs.

```json
{
  "path": "outputs/graph_pilot_v2/page_stress_graph_subset_v1.jsonl",
  "count": 19,
  "by_dataset": {
    "hwr200": 9,
    "hkr_forms": 10
  },
  "by_method": {
    "otsu_gridless": 19
  },
  "metrics_by_dataset": {
    "hkr_forms": {
      "n": 10,
      "skeleton_pixels_mean": 50183.7,
      "nodes_per_1k_skeleton_mean": 164.18330702146264,
      "components_per_1k_skeleton_mean": 85.53848591379963,
      "junctions_per_1k_skeleton_mean": 28.070284378283795,
      "warning_risk_score_mean": 65.0
    },
    "hwr200": {
      "n": 9,
      "skeleton_pixels_mean": 75214.44444444444,
      "nodes_per_1k_skeleton_mean": 146.0503953503772,
      "components_per_1k_skeleton_mean": 74.52590978366061,
      "junctions_per_1k_skeleton_mean": 30.795423469144062,
      "warning_risk_score_mean": 58.333333333333336
    }
  }
}
```

## 8. Clean graph features v1

The first clean graph feature table has been exported.

```json
{
  "path": "outputs/graph_pilot_v2/graph_features_clean_v1.csv",
  "count": 40,
  "by_dataset": {
    "iam": 10,
    "cyrillic_handwriting": 10,
    "hkr_words": 10,
    "school_notebooks": 10
  },
  "by_method": {
    "otsu": 30,
    "sauvola": 10
  },
  "metrics_by_dataset": {
    "cyrillic_handwriting": {
      "n": 10,
      "foreground_ratio_mean": 0.13719205712137506,
      "skeleton_density_mean": 0.041455508155821066,
      "skeleton_pixels_mean": 549.0,
      "nodes_per_1k_skeleton_mean": 86.09895438276996,
      "edges_per_1k_skeleton_mean": 81.92679069840237,
      "components_per_1k_skeleton_mean": 24.103980378125527,
      "junctions_per_1k_skeleton_mean": 30.19118999592855,
      "endpoints_per_1k_skeleton_mean": 55.907764386841414,
      "short_branches_per_1k_skeleton_mean": 17.60750915023726,
      "junction_endpoint_ratio_mean": 0.601514195886433,
      "edge_node_ratio_mean": 0.9517364111114113,
      "component_node_ratio_mean": 0.2862989712989713,
      "edges_per_component_mean": 3.891777389277389,
      "nodes_per_component_mean": 3.978071095571096,
      "mean_width_proxy_mean": 3.593701905786466,
      "warning_count_mean": 0.0,
      "warning_risk_score_mean": 0.0
    },
    "hkr_words": {
      "n": 10,
      "foreground_ratio_mean": 0.146022409601299,
      "skeleton_density_mean": 0.037243926288952484,
      "skeleton_pixels_mean": 785.2,
      "nodes_per_1k_skeleton_mean": 60.45030030393144,
      "edges_per_1k_skeleton_mean": 56.396432018587994,
      "components_per_1k_skeleton_mean": 11.980519574186504,
      "junctions_per_1k_skeleton_mean": 25.000400827767585,
      "endpoints_per_1k_skeleton_mean": 35.44989947616385,
      "short_branches_per_1k_skeleton_mean": 7.857293548319634,
      "junction_endpoint_ratio_mean": 0.8275450544650929,
      "edge_node_ratio_mean": 0.9444802154580854,
      "component_node_ratio_mean": 0.19415734204953997,
      "edges_per_component_mean": 6.049801587301588,
      "nodes_per_component_mean": 6.0883730158730165,
      "mean_width_proxy_mean": 3.8742745975390163,
      "warning_count_mean": 0.0,
      "warning_risk_score_mean": 0.0
    },
    "iam": {
      "n": 10,
      "foreground_ratio_mean": 0.09488292045919744,
      "skeleton_density_mean": 0.018626770645874202,
      "skeleton_pixels_mean": 2663.6,
      "nodes_per_1k_skeleton_mean": 47.852255385268634,
      "edges_per_1k_skeleton_mean": 44.545112198876595,
      "components_per_1k_skeleton_mean": 10.488746118127498,
      "junctions_per_1k_skeleton_mean": 18.597176782287036,
      "endpoints_per_1k_skeleton_mean": 29.255078602981598,
      "short_branches_per_1k_skeleton_mean": 6.630694896482967,
      "junction_endpoint_ratio_mean": 0.6741542838225676,
      "edge_node_ratio_mean": 0.9298563748607693,
      "component_node_ratio_mean": 0.21874386391947193,
      "edges_per_component_mean": 5.139884041884042,
      "nodes_per_component_mean": 5.3024551004551,
      "mean_width_proxy_mean": 4.922012923808266,
      "warning_count_mean": 0.0,
      "warning_risk_score_mean": 0.0
    },
    "school_notebooks": {
      "n": 10,
      "foreground_ratio_mean": 0.16598678305323894,
      "skeleton_density_mean": 0.03646064500248604,
      "skeleton_pixels_mean": 1194.4,
      "nodes_per_1k_skeleton_mean": 79.3434840977559,
      "edges_per_1k_skeleton_mean": 78.79100483792702,
      "components_per_1k_skeleton_mean": 26.19138656049168,
      "junctions_per_1k_skeleton_mean": 27.086392515990887,
      "endpoints_per_1k_skeleton_mean": 52.25709158176501,
      "short_branches_per_1k_skeleton_mean": 18.343357060735066,
      "junction_endpoint_ratio_mean": 0.7309186589201646,
      "edge_node_ratio_mean": 1.0180413650300533,
      "component_node_ratio_mean": 0.2880914226094388,
      "edges_per_component_mean": 5.763174225481528,
      "nodes_per_component_mean": 5.411900712539658,
      "mean_width_proxy_mean": 3.901413633120819,
      "warning_count_mean": 0.0,
      "warning_risk_score_mean": 0.0
    }
  }
}
```

## 9. Page stress graph features v1

The page stress graph feature table has also been exported, but it must remain separate from clean graph features.

```json
{
  "path": "outputs/graph_pilot_v2/graph_features_page_stress_v1.csv",
  "count": 19,
  "by_dataset": {
    "hwr200": 9,
    "hkr_forms": 10
  },
  "by_method": {
    "otsu_gridless": 19
  },
  "metrics_by_dataset": {
    "hkr_forms": {
      "n": 10,
      "foreground_ratio_mean": 0.026157238263602438,
      "skeleton_density_mean": 0.012321670595167944,
      "skeleton_pixels_mean": 50183.7,
      "nodes_per_1k_skeleton_mean": 164.18330702146264,
      "edges_per_1k_skeleton_mean": 138.7764471244912,
      "components_per_1k_skeleton_mean": 85.53848591379963,
      "junctions_per_1k_skeleton_mean": 28.070284378283795,
      "endpoints_per_1k_skeleton_mean": 136.11302264317885,
      "short_branches_per_1k_skeleton_mean": 35.050168206139475,
      "junction_endpoint_ratio_mean": 0.22833748095665585,
      "edge_node_ratio_mean": 0.8433243561290062,
      "component_node_ratio_mean": 0.49876912332691414,
      "edges_per_component_mean": 1.765451408740858,
      "nodes_per_component_mean": 2.0910484241859715,
      "mean_width_proxy_mean": 2.329402059728027,
      "warning_count_mean": 5.0,
      "warning_risk_score_mean": 65.0
    },
    "hwr200": {
      "n": 9,
      "foreground_ratio_mean": 0.08081224431666112,
      "skeleton_density_mean": 0.031097690880114975,
      "skeleton_pixels_mean": 75214.44444444444,
      "nodes_per_1k_skeleton_mean": 146.0503953503772,
      "edges_per_1k_skeleton_mean": 133.80724599961576,
      "components_per_1k_skeleton_mean": 74.52590978366061,
      "junctions_per_1k_skeleton_mean": 30.795423469144062,
      "endpoints_per_1k_skeleton_mean": 115.2549718812331,
      "short_branches_per_1k_skeleton_mean": 36.71973025005627,
      "junction_endpoint_ratio_mean": 0.35452660498758454,
      "edge_node_ratio_mean": 0.9210300382777521,
      "component_node_ratio_mean": 0.45445294023496186,
      "edges_per_component_mean": 2.7081870786030984,
      "nodes_per_component_mean": 2.894871243047007,
      "mean_width_proxy_mean": 2.8000864450032847,
      "warning_count_mean": 3.4444444444444446,
      "warning_risk_score_mean": 58.333333333333336
    }
  }
}
```

## 10. Current interpretation

### 10.1 Clean crop/line graphs

IAM, Cyrillic Handwriting, HKR Words, and School Notebooks all produced clean primary graph subsets with zero warning risk. The normalized features show meaningful dataset-level differences. For example, IAM has longer line-level samples and therefore larger absolute skeleton length, while Cyrillic and School Notebooks have higher normalized node/component density in this pilot.

### 10.2 Page/form stress graphs

HWR200 and HKR Forms remain high-complexity stress-test datasets. Their warning risk is expected, because full-page forms, grid backgrounds, and document layout structures create many components, endpoints, and junctions. These datasets should be used for robustness and failure analysis, not for clean graph feature training without crop/region selection.

### 10.3 School Notebooks thresholding

School Notebooks uses polygon-masked crops. Otsu often over-selects the polygon/background area, while Sauvola better follows the handwritten stroke structure visually. Therefore, Sauvola is the primary graph variant for School Notebooks.

## 11. Artifacts produced

```text
data/pilot/graph_pilot_v2.jsonl
data/pilot/graph_pilot_v2_summary.json
outputs/graph_pilot_v2/binary_skeleton_pilot_summary.json
outputs/graph_pilot_v2/graph_builder_pilot_report.json
outputs/graph_pilot_v2/graph_pilot_v2_report.md
outputs/graph_pilot_v2/graph_failure_cases_v2.json
outputs/graph_pilot_v2/graph_quality_metrics_v1.csv
outputs/graph_pilot_v2/graph_quality_metrics_v1_summary.json
outputs/graph_pilot_v2/clean_graph_subset_v1.jsonl
outputs/graph_pilot_v2/page_stress_graph_subset_v1.jsonl
outputs/graph_pilot_v2/graph_features_clean_v1.csv
outputs/graph_pilot_v2/graph_features_page_stress_v1.csv
outputs/graph_pilot_v2/graph_features_v1_summary.json
```

## 12. Acceptance criteria status

```text
[x] HI-CSG-R graph schema documented
[x] Pilot subset created
[x] Binary masks produced
[x] Skeletons produced
[x] Pixel graph produced
[x] Canonical graph JSON saved
[x] Graph overlays saved
[x] Diagnostics saved
[x] Dataset-specific binarization decisions made
[x] Failure cases selected
[x] Normalized graph metrics exported
[x] Clean graph subset separated from page stress subset
```

## 13. Next stage

Recommended next stage:

```text
Stage 3 — HTR baselines and graph-aware experimental protocol
```

Stage 3 should include:

- image-only HTR baselines for IAM, Cyrillic Handwriting, HKR Words, and School Notebooks;

- graph feature sanity checks on clean_graph_subset_v1;

- larger clean graph extraction run for crop/line datasets;

- graph-only baseline prototype;

- image+graph fusion design;

- robustness/failure analysis using page_stress_graph_subset_v1.
