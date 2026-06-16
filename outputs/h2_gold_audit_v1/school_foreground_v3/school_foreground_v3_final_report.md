# School foreground v3 final report

## 1. Verdict

```text
Selected method: school_dark_auto
Status: candidate preprocessing fix found
```

Manual browser audit showed that replacing Sauvola with simple dark-threshold foreground extraction substantially improves school-notebooks foreground masks on the audited samples.

## 2. Manual audit summary

| metric | value |
|---|---:|
| n | 23 |
| good fix rate | 82.61% |
| partial fix rate | 8.70% |
| bad fix rate | 8.70% |
| real ink erased rate | 0.00% |
| background/blob artifact after rate | 17.39% |
| skeleton follows ink after rate | 82.61% |

## 3. Manual best variant counts

| variant | n |
|---|---:|
| `baseline_sauvola` | 1 |
| `global_dark_120` | 9 |
| `global_dark_145` | 13 |

## 4. Feature-level comparison

### `global_dark_145`

| split | old fg | new fg | old skel | new skel | old high-fg | new high-fg |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 0.1955 | 0.1679 | 0.0574 | 0.0430 | 0.01% | 4.77% |
| `val` | 0.2073 | 0.1602 | 0.0628 | 0.0437 | 0.00% | 3.35% |
| `test` | 0.2026 | 0.1559 | 0.0618 | 0.0440 | 0.00% | 1.75% |

### `school_dark_auto`

| split | old fg | new fg | old skel | new skel | old high-fg | new high-fg |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 0.1955 | 0.1488 | 0.0574 | 0.0426 | 0.01% | 0.93% |
| `val` | 0.2073 | 0.1459 | 0.0628 | 0.0430 | 0.00% | 0.50% |
| `test` | 0.2026 | 0.1449 | 0.0618 | 0.0436 | 0.00% | 0.05% |

## 5. Decision

`school_dark_auto` is selected over fixed `global_dark_145` because it reduces excessive foreground more consistently while preserving nonzero skeleton structure. It is a deterministic preprocessing fix for school-notebooks graph extraction, not an HTR architecture change.

## 6. Limits

This fix is validated on the H2 school-notebooks audit subset and feature-level split summaries. It should be used for graph extraction repair and H2 follow-up, not yet as evidence of improved HTR accuracy.