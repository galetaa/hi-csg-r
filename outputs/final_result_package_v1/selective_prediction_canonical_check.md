# Selective prediction canonical check v1

Verdict: **WEAK_PASS_LEAKAGE_REVIEW**

Selective prediction artifacts reference canonical +10k and required variants, but leakage-risk keys were detected. Manual review required.

## Variant coverage

- confidence: `True`
- graph_or_quality: `True`
- confidence_graph: `True`

## Canonical files

- `outputs/htr_graph_v1/selective_iter2_confidence_v1/calibration_bins.csv`
- `outputs/htr_graph_v1/selective_iter2_confidence_v1/coverage_curves.csv`
- `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_point_strata.md`
- `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_points.json`
- `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_points.md`
- `outputs/htr_graph_v1/selective_iter2_confidence_v1/risk_table_by_bucket.csv`
- `outputs/htr_graph_v1/selective_iter2_confidence_v1/selective_summary.json`
- `outputs/htr_graph_v1/selective_iter2_confidence_v1/selective_summary.md`
- `outputs/htr_graph_v1/selective_iter2_lineaug_v1/coverage_curves.csv`
- `outputs/htr_graph_v1/selective_iter2_lineaug_v1/risk_table_by_bucket.csv`
- `outputs/htr_graph_v1/selective_iter2_lineaug_v1/selective_summary.json`
- `outputs/htr_graph_v1/selective_iter2_lineaug_v1/selective_summary.md`

## Bad-model / exploratory references

None detected.

## Leakage review files

- `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_point_strata.md` hits=['text_len']

## Inspected files

| path | canonical | bad hits | leakage hits |
|---|---:|---|---|
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/calibration_bins.csv` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/coverage_curves.csv` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_point_strata.csv` | False | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_point_strata.md` | True | `[]` | `['text_len']` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_points.json` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_points.md` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/result_card.md` | False | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/risk_table_by_bucket.csv` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/selective_summary.json` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_confidence_v1/selective_summary.md` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_lineaug_v1/coverage_curves.csv` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_lineaug_v1/result_card.md` | False | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_lineaug_v1/risk_table_by_bucket.csv` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_lineaug_v1/selective_summary.json` | True | `[]` | `[]` |
| `outputs/htr_graph_v1/selective_iter2_lineaug_v1/selective_summary.md` | True | `[]` | `[]` |