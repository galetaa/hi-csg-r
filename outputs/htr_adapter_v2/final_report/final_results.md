# HI-CSG-R Late Correction v2: результаты

**Execution status:** `complete_negative_development`

**H4-v2:** `not_supported`

## Зафиксированные результаты v1

V1 остается завершенным отрицательным экспериментом: `M0-FT CER=0.079537`, `M3 CER=0.082196`, `M3-shuffle CER=0.083004`. V2 этот вывод не перезаписывает.

## V2 technical evidence

- preflight: `CONTINUE_FULL`
- fixed decode/correction: `blank penalty=-0.4`, `alpha_max=0.25`
- preflight D2/D3 CER gains: `+0.000808` / `-0.000191`
- split audit: `PASS`
- independent split: `train=35498`, `dev=3000`, `holdout=1500`
- split overlap counts: `0` (sample/path/group/SHA1)
- smoke gate: `PASS`
- fresh B0-dev: `COMPLETE`
- B0 best dev micro-CER / epoch: `0.139475` / `80` completed epochs
- holdout evaluated: `False`
- final test evaluated: `False`
- image-only parameters: `3900892`
- v2 trainable parameters: `196156` (`5.028%`)

## Development metrics

| Run | Gate | Baseline CER | Correct CER | Shuffle CER | Zero CER | Relative improvement |
|---|---:|---:|---:|---:|---:|---:|
| v2_1_dev_p05_seed42 | STOP | 0.139370 | 0.139106 | 0.139106 | 0.139370 | 0.189% |
| v2_2_dev_p05_seed42 | STOP | 0.139370 | 0.139053 | 0.138711 | 0.139370 | 0.227% |

### v2_1_dev_p05_seed42

- absolute / relative CER change vs B0: `-0.000263` / `-0.189%` (adapter minus baseline)
- correct minus shuffle CER: `+0.000000`
- domain CER deltas (Cyrillic/HKR/School): `+0.000534` / `+0.000773` / `-0.001221`
- alpha / empty correction max: `0.000870217` / `0`
- gate conditions: `relative_cer_improvement_at_least_1_percent=False`, `correct_better_shuffle=False`, `empty_correction_invariant=True`, `no_domain_degrades_over_0_003=True`, `exact_drop_at_most_0_005=True`

### v2_2_dev_p05_seed42

- absolute / relative CER change vs B0: `-0.000316` / `-0.227%` (adapter minus baseline)
- correct minus shuffle CER: `+0.000342`
- domain CER deltas (Cyrillic/HKR/School): `+0.000267` / `+0.000945` / `-0.001327`
- alpha / empty correction max: `0.001067763` / `0`
- gate conditions: `relative_cer_improvement_at_least_1_percent=False`, `correct_better_shuffle=False`, `empty_correction_invariant=True`, `no_domain_degrades_over_0_003=True`, `exact_drop_at_most_0_005=True`

## Development paired statistics

### V2-2 correct vs B0

- delta CER / relative delta: `-0.000316` / `-0.227%`
- paired bootstrap CI95 / p: `[-0.001296, +0.000663]` / `0.544946`
- WER / Exact delta: `+0.002159` / `+0.000000`
- wins/losses/ties: `140/134/2726`

### V2-2 correct vs shuffle

- delta CER / relative delta: `+0.000342` / `+0.247%`
- paired bootstrap CI95 / p: `[-0.000188, +0.000892]` / `0.239376`
- WER / Exact delta: `+0.000664` / `+0.000333`
- wins/losses/ties: `40/50/2910`

## Intervention diagnostics (best-CER development variant)

- alpha: `0.001067763`
- gate mean/std/P90/non-empty/empty: `0.014766` / `0.057859` / `0.016581` / `0.015530` / `0.000000`
- intervention / strong intervention / changed prediction: `7.052%` / `3.750%` / `12.767%`
- intervention precision (improves edit distance): `36.554%`
- improved/hurt samples: `140` / `134`
- correction/base L2 ratio / empty correction max: `0.000845` / `0`
- visual uncertainty mean/P90: `0.024387` / `0.029704`
- structural risk mean/P90: `0.161421` / `0.398590`


## Protected stages

- `lambda_pres=0.10` repeat: **not run** (no dev PASS candidate).
- independent holdout: **not opened**.
- final seeds 42/43/44: **not run**.
- canonical test/page-disjoint/robustness: **not opened**.

## Решение

H4-v2 не подтверждена. Оба разрешенных development-варианта не прошли заранее установленный dev gate: снижение CER было меньше 1%, а correct graph не превзошел matched shuffle. Согласно frozen protocol p10, holdout, final seeds и test не запускались. Отрицательный вывод v1 сохранен отдельно.
