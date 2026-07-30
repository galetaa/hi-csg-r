# HI-CSG-R adapter v2 feature audit

**Status:** `PASS`

Normalizer fitted only on `adapter_v2_train`.

| Split | Records | Valid bins | Non-finite | Name mismatches |
|---|---:|---:|---:|---:|
| train | 35498 | 2575073 | 0 | 0 |
| dev | 3000 | 229404 | 0 | 0 |
| holdout | 1500 | 120132 | 0 | 0 |

- inactive features: `[]`
- ambiguous_edge_fraction active: `True`
- risk quantiles: `{'component_count_norm': {'q05': 6.25, 'q50': 25.0, 'q95': 87.5}, 'short_branch_fraction': {'q05': 0.0, 'q50': 0.0, 'q95': 0.21475814282894135}, 'warning_density': {'q05': 0.0, 'q50': 0.0, 'q95': 0.2290380597114563}}`
