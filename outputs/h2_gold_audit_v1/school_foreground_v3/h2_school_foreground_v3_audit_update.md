# H2 school-notebooks foreground v3 audit update

## 1. Previous H2 school-notebooks status

| metric | old value |
|---|---:|
| n | 23 |
| critical topology error rate | 0.957 |
| skeleton follows ink rate | 0.000 |
| border artifact rate | 1.000 |
| mean graph quality 0–3 | 0.826 |

## 2. Foreground v3 audit

| metric | v3 value |
|---|---:|
| n | 23 |
| selected method | `school_dark_auto` |
| good fix rate | 0.826 |
| partial fix rate | 0.087 |
| bad fix rate | 0.087 |
| real ink erased rate | 0.000 |
| background artifact after rate | 0.174 |
| skeleton follows ink after rate | 0.826 |

## 3. Interpretation

Foreground v3 substantially repairs the school-notebooks preprocessing failure on the audited subset. The fix should be treated as a graph-extraction preprocessing improvement, not as HTR accuracy evidence.

The original H2 conclusion remains historically valid for the old pipeline, but the improved pipeline now has a viable path for school-notebooks foreground extraction.