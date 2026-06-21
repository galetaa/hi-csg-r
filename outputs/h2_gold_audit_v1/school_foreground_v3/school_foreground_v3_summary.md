# School preprocessing v2 summary

## 1. Aggregate

| metric | value |
|---|---:|
| n | 23 |
| good fix rate | 0.826 |
| partial fix rate | 0.087 |
| bad fix rate | 0.087 |
| real ink erased rate | 0.000 |
| border artifact after rate | 0.174 |
| skeleton follows ink after rate | 0.826 |

## 2. Best variant counts

| variant | n |
|---|---:|
| `baseline_sauvola` | 1 |
| `global_dark_120` | 9 |
| `global_dark_145` | 13 |

## 3. Fix grade counts

| grade | n |
|---|---:|
| `bad_fix` | 2 |
| `good_fix` | 19 |
| `partial_fix` | 2 |

## 4. Verdict

A candidate preprocessing fix appears viable. The next step is to implement the winning variant as a deterministic preprocessing function and rerun H2 audit metrics on school-notebooks.