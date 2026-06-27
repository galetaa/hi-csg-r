# Selective prediction leakage clearance v1

Clearance status: **PASS_WITH_NOTE**

## Reasons

- Canonical +10k artifacts and all required variants were detected; the only leakage-risk hit is `text_len` in a reporting/stratification file.

## Variant coverage

- confidence: `True`
- graph_or_quality: `True`
- confidence_graph: `True`

## Reviewed leakage notes

- `outputs/htr_graph_v1/selective_iter2_confidence_v1/operating_point_strata.md`
  - hits: `['text_len']`
  - cleared: `True`
  - review: `text_len` appears in a post-hoc operating-point stratification/reporting table, not as an evident risk-model feature.

## Final interpretation

Selective prediction is acceptable as a secondary applied result if described as canonical +10k confidence/graph-quality risk analysis. The detected `text_len` occurrence must be described as post-hoc stratification/reporting, not as a model feature.