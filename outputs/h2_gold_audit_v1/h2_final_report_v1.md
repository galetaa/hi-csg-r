# H2 final report — v1

## 1. Verdict

```text
H2-v1: partial support with preprocessing exception
```

The graph extraction pipeline preserves visible stroke structure reasonably well on HKR and Cyrillic samples in the manual audit. However, the school-notebooks subset is dominated by crop/border artifacts that are binarized as foreground. Those failures are upstream preprocessing failures and should not be interpreted as pure graph-topology failures.

## 2. Overall manual audit

| metric | value |
|---|---:|
| n | 100 |
| usable rate | 0.920 |
| critical topology error rate | 0.240 |
| skeleton follows ink rate | 0.740 |
| border artifact rate | 0.230 |
| mean graph quality 0–3 | 2.400 |

## 3. Dataset split

| dataset | n | critical | follows ink | border artifact | mean quality | failure stages |
|---|---:|---:|---:|---:|---:|---|
| `cyrillic_handwriting` | 37 | 0.054 | 0.919 | 0.000 | 2.757 | ok:37 |
| `hkr_words` | 40 | 0.000 | 1.000 | 0.000 | 2.975 | ok:40 |
| `school_notebooks_clean` | 23 | 0.957 | 0.000 | 1.000 | 0.826 | binarization:23 |

## 4. HKR + Cyrillic structural preservation

| subset | n | critical | follows ink | mean quality |
|---|---:|---:|---:|---:|
| `hkr_words + cyrillic_handwriting` | 77 | 0.026 | 0.961 | 2.870 |

This is the subset that should be used for the H2-v1 graph-topology preservation claim.

## 5. School notebooks exception

The school-notebooks subset should be reported separately. Manual staging found that all audited school-notebooks samples were affected by binarization-stage failure with border artifacts. The issue is not the canonical graph abstraction itself, but the upstream conversion from cropped image to foreground mask.

```text
school_notebooks_clean:
  n = 23
  critical topology error rate = 0.957
  skeleton follows ink rate = 0.000
  border artifact rate = 1.000
  failure stage = binarization
```

## 6. Border suppression v1

A simple border-connected component suppression check was attempted and rejected. Visual inspection showed that it either failed to remove the artifact or removed handwriting together with the border. Therefore it is not integrated.

See: `outputs/h2_gold_audit_v1/border_suppression_v1/rejection_note.md`

## 7. Consequence

For the thesis/report:

- do not claim that H2 is fully solved across all datasets;
- do claim partial H2 support on HKR/Cyrillic audited samples;
- report school-notebooks as a preprocessing limitation and failure mode;
- do not aggregate school-notebooks into pure graph-topology error statistics;
- do not tune HTR architecture based on this finding.
