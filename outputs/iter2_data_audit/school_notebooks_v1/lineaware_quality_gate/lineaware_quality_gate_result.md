# School lineaware v3 quality gate

- annotations: `outputs/iter2_data_audit/school_notebooks_v1/lineaware_quality_gate/lineaware_quality_gate_annotations.csv`
- accepted: `True`

| group | n | complete | usable | ink loss | line residual | neighbor removed | skeleton follows | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `overall` | 120 | 1.000 | 0.958 | 0.050 | 0.075 | 1.000 | 1.000 | True |
| `high_feature_change` | 30 | 1.000 | 1.000 | 0.000 | 0.067 | 1.000 | 1.000 | True |
| `high_ruling_response` | 30 | 1.000 | 0.833 | 0.200 | 0.000 | 1.000 | 1.000 | False |
| `old_binary_issue` | 30 | 1.000 | 1.000 | 0.000 | 0.167 | 1.000 | 1.000 | False |
| `random_stable_control` | 30 | 1.000 | 1.000 | 0.000 | 0.067 | 1.000 | 1.000 | True |
