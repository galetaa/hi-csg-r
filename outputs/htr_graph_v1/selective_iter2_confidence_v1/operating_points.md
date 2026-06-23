# Confidence Operating Points

Model: `plus_10k_context`
Risk method: `confidence_graph`

Thresholds are selected on validation School samples to match target School coverage, then applied unchanged to test.

## Operating Points

| point | threshold | split | scope | target School coverage | actual coverage | n accepted | CER | WER | exact |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `strict` | 0.2917 | `val` | `school` | 0.5000 | 0.5000 | 1000 | 0.0286 | 0.1130 | 0.8870 |
| `strict` | 0.2917 | `val` | `all` | 0.5000 | 0.3140 | 1884 | 0.0288 | 0.1186 | 0.8795 |
| `strict` | 0.2917 | `test` | `school` | 0.5000 | 0.3905 | 781 | 0.0491 | 0.1908 | 0.8143 |
| `strict` | 0.2917 | `test` | `all` | 0.5000 | 0.2294 | 1276 | 0.0519 | 0.2036 | 0.7900 |
| `balanced` | 0.5787 | `val` | `school` | 0.8000 | 0.8000 | 1600 | 0.0454 | 0.1894 | 0.8119 |
| `balanced` | 0.5787 | `val` | `all` | 0.8000 | 0.6700 | 4020 | 0.0479 | 0.2100 | 0.7729 |
| `balanced` | 0.5787 | `test` | `school` | 0.8000 | 0.6615 | 1323 | 0.0685 | 0.2834 | 0.7249 |
| `balanced` | 0.5787 | `test` | `all` | 0.8000 | 0.5578 | 3103 | 0.0677 | 0.2958 | 0.6800 |
| `broad` | 0.7788 | `val` | `school` | 0.9000 | 0.9000 | 1800 | 0.0572 | 0.2422 | 0.7600 |
| `broad` | 0.7788 | `val` | `all` | 0.9000 | 0.8373 | 5024 | 0.0614 | 0.2754 | 0.6955 |
| `broad` | 0.7788 | `test` | `school` | 0.9000 | 0.7795 | 1559 | 0.0845 | 0.3435 | 0.6652 |
| `broad` | 0.7788 | `test` | `all` | 0.9000 | 0.7575 | 4214 | 0.0864 | 0.3747 | 0.5812 |

## Error Audit

- strict accepted errors: 268
- strict rejected correct: 1571

Files:
- `accepted_errors_high_confidence.jsonl`
- `rejected_correct_low_confidence.jsonl`
