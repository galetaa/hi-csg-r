# Seed-42 Validation Gate

Status: **STOP**

- M0-FT CER: `0.079537`
- M3 CER: `0.082196`
- relative improvement: `-3.342%`

| condition | passed |
|---|---:|
| `relative_improvement_2pct` | **False** |
| `two_domains_not_worse` | **False** |
| `max_domain_degradation` | **True** |
| `correct_better_shuffle` | **True** |
| `gate_variable` | **True** |
| `adapter_gradient` | **True** |

## Domain CER deltas

- `cyrillic_handwriting`: `+0.002104`
- `hkr_words`: `+0.002841`
- `school_notebooks_clean`: `+0.003033`

## Decision

M2, seeds 43/44 and test are blocked by the frozen protocol. H4 remains exploratory.
