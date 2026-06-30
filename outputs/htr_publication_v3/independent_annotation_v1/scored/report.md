# Independent Annotation v1 Score

- reference: `outputs/h2_gold_audit_v1/annotations/annotation_100_filled.csv`
- second: `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_second_filled.csv`
- reference rows: 100
- second rows: 0
- matched sample ids: 0
- minimally complete rows: 0
- formal IAA ready: False

## Agreement

| field | n | agreement | Wilson 95% CI | Cohen kappa | weighted kappa | missing second |
|---|---:|---:|---:|---:|---:|---:|
| `audit_usable` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `ink_visible_ok` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `skeleton_follows_ink` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `missed_visible_stroke` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `spurious_stroke` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `endpoint_error` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `junction_error` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `loop_error` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `critical_topology_error` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |
| `graph_quality_0_3` | 0 | n/a | [n/a, n/a] | n/a | n/a | 0 |

## Interpretation

- If the second CSV was filled by a genuinely independent annotator, these metrics support formal inter-annotator agreement reporting for fields with adequate n.
- If the second CSV was filled by the same annotator or by an AI assistant, report this as repeated/AI consistency rather than formal independent IAA.
