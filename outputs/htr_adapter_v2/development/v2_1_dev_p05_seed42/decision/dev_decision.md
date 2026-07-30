# HI-CSG-R Late Correction v2 dev decision

**Status:** `STOP`

| Model | CER | WER | Exact |
|---|---:|---:|---:|
| baseline | 0.139370 | 0.479907 | 0.423000 |
| correct | 0.139106 | 0.481402 | 0.422667 |
| shuffle | 0.139106 | 0.480737 | 0.423000 |
| zero | 0.139370 | 0.479907 | 0.423000 |

- relative CER improvement: `0.1890%`
- correct-shuffle CER delta: `0.000000`
- domain CER deltas: `{'cyrillic': 0.0005340453938584705, 'hkr': 0.0007731294562322855, 'school': -0.0012208067940552014}`
- empty correction max: `0.000e+00`

## Conditions

- FAIL `relative_cer_improvement_at_least_1_percent`
- FAIL `correct_better_shuffle`
- PASS `empty_correction_invariant`
- PASS `no_domain_degrades_over_0_003`
- PASS `exact_drop_at_most_0_005`
