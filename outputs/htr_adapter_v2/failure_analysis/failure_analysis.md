# HI-CSG-R Late Correction v2: failure/intervention analysis

Samples: `3000`

| Group | N | CER | Mean uncertainty | Mean gate | Mean correction norm |
|---|---:|---:|---:|---:|---:|
| A_baseline_correct_v2_wrong | 28 | 0.093023 | 0.023642 | 0.017236 | 0.129373 |
| B_baseline_wrong_v2_correct | 28 | 0.000000 | 0.024008 | 0.016328 | 0.125113 |
| C_both_wrong | 1703 | 0.199665 | 0.032925 | 0.018706 | 0.140611 |
| D_both_correct | 1241 | 0.000000 | 0.013440 | 0.009842 | 0.072441 |

- intervention precision: `0.365535`
- changed prediction rate: `0.127667`

Списки `graph_helps`, `graph_hurts`, `high_intervention_unchanged` и `low_intervention_errors` сохранены по 20 примеров, если в evaluation существует достаточно случаев соответствующего типа.
