# Полный отчет по эксперименту HI-CSG-R Late Correction v2

- **Протокол:** `crnn_ctc_hi_csg_r_late_correction_protocol_v2`
- **Предыдущий протокол:** `crnn_ctc_hi_csg_r_adapter_protocol_v1`
- **Версия признаков:** `hi_csg_r_xaligned_v1`
- **Дата фиксации результата:** 2026-07-30
- **Фактически достигнутый статус:** завершенный отрицательный результат на независимом development split
- **Статус H4-v2:** `not_supported`
- **Holdout использован:** нет
- **Финальный test использован:** нет
**Финальный коммит исполнения:** `c107f18`

## 1. Краткий итог

HI-CSG-R Late Correction v2 полностью реализован и выполнен до
заранее установленного development stopping gate.

Техническая часть прошла:

- протокол v2 и допустимое пространство моделей зафиксированы до обучения;
- выполнены preflight-анализы D1-D3 на артефактах v1;
- создан независимый group-aware split `35 498 / 3 000 / 1 500`;
- пересечения sample ID, image path, hierarchy group и exact SHA1 равны нулю;
- normalizer и risk quantiles рассчитаны только по reduced train;
- реализованы post-normalization masking, masked multiscale pooling,
  visual uncertainty, structural-risk attenuation и bounded late correction;
- visual CNN, projection, BiLSTM и classifier полностью заморожены;
- strict empty-bin invariant выполняется: graph correction на пустых bins равна
  нулю;
- one-sample и 128-sample smoke/overfit gates прошли;
- fresh image-only `B0-dev-v2` обучен без holdout leakage;
- выполнены оба разрешенных development run: `V2-1-p05` и `V2-2-p05`;
- для обоих выполнены correct, matched shuffle и zero controls;
- рассчитаны paired bootstrap statistics и intervention/failure analysis;
- полностью исполнен experiment notebook с сохраненными outputs.

Оба разрешенных варианта получили `STOP`:

| Модель | B0 CER | Correct CER | Shuffle CER | Zero CER | Relative improvement | Gate |
|---|---:|---:|---:|---:|---:|---|
| `V2-1-dev-p05` | 0.139370 | 0.139106 | 0.139106 | 0.139370 | 0.189% | STOP |
| `V2-2-dev-p05` | 0.139370 | 0.139053 | 0.138711 | 0.139370 | 0.227% | STOP |

Главное сравнение лучшего по correct CER варианта `V2-2` с B0:

- absolute delta CER: `-0.000316`;
- relative CER improvement: `0.227%`;
- delta WER: `+0.002159`, то есть WER ухудшился;
- delta Exact: `0.000000`;
- paired bootstrap CI95: `[-0.001296, +0.000663]`;
- two-sided bootstrap `p = 0.544946`;
- wins/losses/ties: `140 / 134 / 2726`.

Correct graph не оказался лучше matched shuffled graph:

- correct CER: `0.139053`;
- shuffled CER: `0.138711`;
- `correct - shuffle = +0.000342 CER`;
- paired CI95: `[-0.000188, +0.000892]`;
- `p = 0.239376`.

Следовательно:

> H4-v2 не подтверждена. После устранения раннего fusion, обновления visual
> backbone и вмешательства на пустых bins используемое 20-мерное x-aligned
> представление HI-CSG-R не обеспечило воспроизводимого sample-specific
> снижения CER относительно frozen image-only CRNN-CTC.

По frozen stopping rules не запускались:

- повтор лучшего варианта с `lambda_pres=0.10`;
- independent holdout;
- final seeds `42/43/44`;
- canonical test;
- page-disjoint evaluation;
- robustness evaluation.

Это не является незавершенным исполнением. Это обязательный конечный результат
протокола после двух development `STOP`.

## 2. Связь с отрицательным результатом v1

V1 не изменялся и не перезаписывался.

Зафиксированные validation-результаты v1:

| Модель | micro-CER | WER | Exact |
|---|---:|---:|---:|
| `M0-FT seed42` | 0.079537 | 0.332398 | 0.626833 |
| `M3 correct graph` | 0.082196 | 0.337477 | 0.620667 |
| `M3 shuffled graph` | 0.083004 | 0.341486 | 0.615167 |

V1 показал:

- `M3 - M0-FT = +0.002658 CER`;
- relative degradation `+3.342%`;
- correct graph лучше shuffled на `0.000808 CER`;
- graph branch технически активна, но не превосходит matched image-only model.

V2 является новой заранее зафиксированной итерацией. Он проверяет другую
гипотезу:

```text
v1:
graph residual -> visual sequence -> BiLSTM -> logits

v2:
frozen visual model -> baseline logits
graph branch -> small bounded correction directly to logits
```

Итоговые научные статусы должны сохраняться раздельно:

```text
v1 pre-BiLSTM x-aligned residual adapter: negative
v2 bounded late logit correction: negative at independent development gate
```

## 3. Исследовательская постановка v2

### 3.1. Основной вопрос

Проверялось, может ли bounded late correction на основе HI-CSG-R снизить CER
сильной CRNN-CTC, если:

- visual backbone полностью заморожен;
- graph contribution строго нулевой на пустых bins;
- коррекция допускается преимущественно при повышенной visual uncertainty;
- correction ограничивается глобальным `alpha`;
- correct graph сравнивается с matched shuffled graph.

### 3.2. Рабочая гипотеза

Sample-specific сигнал, обнаруженный в v1 correct-vs-shuffle сравнении, мог
оказаться полезным как малая локальная коррекция baseline logits, даже если
постоянное вмешательство перед BiLSTM ухудшало распознавание.

### 3.3. Нулевая гипотеза

Даже после frozen backbone, strict masking, late fusion и bounded gating
20-мерное x-aligned представление не дает воспроизводимого снижения CER.

### 3.4. Критерий положительного development результата

Вариант должен был одновременно выполнить:

```text
relative CER improvement vs B0 >= 1%
correct graph better than shuffle
no domain degradation > 0.003 CER
Exact drop <= 0.005
empty correction max < 1e-7
```

Ни `V2-1`, ни `V2-2` не выполнили первые два условия.

## 4. Зафиксированный экспериментальный протокол

### 4.1. Архитектурная граница

Базовая модель:

```text
grayscale image
-> frozen CNN
-> frozen visual projection
-> frozen BiLSTM
-> frozen classifier
-> baseline CTC logits
```

Единственный новый trainable модуль:

```text
HI-CSG-R x-aligned features [T,20]
-> strict post-normalization mask
-> masked pooling, kernels 1/5/9
-> graph adapter [T,128]
-> uncertainty/risk-aware gate
-> bounded correction [T,C]
-> residual addition to baseline logits
```

Итог:

```text
Z_final = Z_base + alpha * gate * delta_logits
```

### 4.2. Фиксированные значения

| Параметр | Значение |
|---|---|
| Split seed | `20260730` |
| Model seed development | `42` |
| Blank logit penalty | `-0.4` |
| Alpha maximum | `0.25` |
| Alpha logit initialization | `-6.0` |
| Graph embedding dimension | `128` |
| Batch size | `16` |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Gradient clipping | `5.0` |
| Preservation temperature | `1.5` |
| Primary `lambda_pres` | `0.05` |
| Permitted repeat | только лучший PASS с `0.10` |
| Max development epochs | `20` |
| Min development epochs | `8` |
| Early stopping patience | `5` |
| Primary metric | development micro-CER |
| Development variants | `V2-1-p05`, `V2-2-p05`, максимум один p10 |
| Visual backbone | полностью frozen |
| Holdout/test tuning | запрещен |

### 4.3. Auxiliary schedule

```text
epochs 1-3: lambda_aux = 0.15
epochs 4-6: lambda_aux = 0.05
epochs 7+:  lambda_aux = 0.00
```

Общий loss:

```text
L = L_main_ctc
  + lambda_pres * L_baseline_preservation
  + lambda_aux(epoch) * L_graph_aux_ctc
```

### 4.4. Максимум training runs

Разрешалось:

| ID | Назначение | Запуски |
|---|---|---:|
| `B0-dev-v2` | fresh reduced-train image-only baseline | 1 |
| `V2-1-dev-p05` | mask + uncertainty + late correction | 1 |
| `V2-2-dev-p05` | V2-1 + risk attenuation | 1 |
| `V2-best-dev-p10` | repeat лучшего PASS | максимум 1 |

Фактически научно валидных запусков:

```text
B0-dev-v2
V2-1-dev-p05
V2-2-dev-p05
```

P10 не разрешен, поскольку PASS-кандидата не было.

## 5. Реализованная кодовая база

### 5.1. Документы и конфигурации

| Назначение | Файл |
|---|---|
| Frozen protocol | `docs/crnn_ctc_hi_csg_r_late_correction_protocol_v2.md` |
| Protocol freeze | `outputs/htr_adapter_v2/protocol_freeze/protocol_freeze.md` |
| Sampler amendment 001 | `outputs/htr_adapter_v2/protocol_freeze/amendment_001.md` |
| Sampler amendment 002 | `outputs/htr_adapter_v2/protocol_freeze/amendment_002.md` |
| Preflight config | `configs/htr_adapter_v2/preflight.yaml` |
| Fresh baseline config | `configs/htr_adapter_v2/b0_dev_seed42.yaml` |
| V2-1 config | `configs/htr_adapter_v2/v2_1_dev_p05_seed42.yaml` |
| V2-2 config | `configs/htr_adapter_v2/v2_2_dev_p05_seed42.yaml` |
| P10 template | `configs/htr_adapter_v2/v2_best_dev_p10_seed42.yaml` |
| Final templates | `configs/htr_adapter_v2/final_seed42/43/44.yaml` |

### 5.2. Core modules

| Назначение | Файл |
|---|---|
| Dataset, masks, collate, sampler | `src/htr/dataset_adapter_v2.py` |
| Frozen backbone и late correction model | `src/htr/model_hi_csg_r_late_correction_v2.py` |
| Masked multiscale pooling | `src/htr/masked_pooling.py` |
| Visual uncertainty | `src/htr/uncertainty.py` |
| Preservation KL и auxiliary schedule | `src/htr/losses_adapter_v2.py` |
| Runtime/checkpoint helpers | `src/htr/adapter_runtime_v2.py` |

### 5.3. Data, training и evaluation tools

| Назначение | Файл |
|---|---|
| V1 preflight D1-D3 | `tools/diagnose_hi_csg_r_adapter_v1_for_v2.py` |
| Group-aware split | `tools/create_hi_csg_r_adapter_v2_split.py` |
| Split audit | `tools/audit_hi_csg_r_adapter_v2_split.py` |
| Feature/normalizer preparation | `tools/prepare_hi_csg_r_adapter_v2_features.py` |
| Smoke manifests | `tools/create_hi_csg_r_adapter_v2_smoke_manifests.py` |
| Smoke gate | `tools/check_hi_csg_r_adapter_v2_smoke.py` |
| Fresh B0 trainer | `tools/train_crnn_ctc_adapter_v2_baseline.py` |
| V2 trainer | `tools/train_crnn_ctc_hi_csg_r_late_correction_v2.py` |
| V2 evaluator | `tools/evaluate_crnn_ctc_hi_csg_r_late_correction_v2.py` |
| Matched shuffle map | `tools/build_hi_csg_r_adapter_v2_shuffle_map.py` |
| Development/holdout gate | `tools/compare_hi_csg_r_adapter_v2_results.py` |
| Candidate selection | `tools/select_hi_csg_r_adapter_v2_candidate.py` |
| Baseline prediction materialization | `tools/materialize_hi_csg_r_adapter_v2_baseline_predictions.py` |
| Paired bootstrap | `tools/paired_bootstrap_hi_csg_r_adapter_v2.py` |
| Failure analysis | `tools/analyze_hi_csg_r_adapter_v2_failures.py` |
| Final config freeze | `tools/resolve_hi_csg_r_adapter_v2_final_configs.py` |
| Final statistics | `tools/summarize_hi_csg_r_adapter_v2_final_statistics.py` |
| Final report | `tools/make_hi_csg_r_adapter_v2_final_report.py` |
| Notebook builder | `tools/build_hi_csg_r_late_correction_v2_notebook.py` |

### 5.4. Tests

Реализованы:

```text
tests/test_htr_adapter_v2_split.py
tests/test_htr_adapter_v2_masking.py
tests/test_htr_adapter_v2_pooling.py
tests/test_htr_adapter_v2_uncertainty.py
tests/test_htr_adapter_v2_risk.py
tests/test_htr_adapter_v2_model.py
tests/test_htr_adapter_v2_losses.py
tests/test_htr_adapter_v2_checkpoint.py
```

## 6. WP0-WP14: фактический статус

| WP | Требование | Статус | Фактический результат |
|---|---|---|---|
| WP0 | Заморозить protocol v2 | PASS | Protocol, configs и hashes зафиксированы |
| WP1 | Preflight D1-D3 | PASS | `CONTINUE_FULL`, penalty `-0.4`, alpha max `0.25` |
| WP2 | Independent split | PASS | `35 498 / 3 000 / 1 500`, overlaps 0 |
| WP3 | Features и train-only normalizer | PASS | 39 998 records, NaN/Inf 0 |
| WP4 | Dataset/collate/masks | PASS | Empty, time и padding masks разделены |
| WP5 | Late correction model | PASS | Frozen backbone, 196 156 trainable params |
| WP6 | Tests | PASS | 24/24 v2 tests, 39/39 repository tests |
| WP7 | Trainer/evaluator | PASS | Resume, gates, controls, metadata |
| WP8 | Smoke/overfit | PASS | 8/8 conditions |
| WP9 | Fresh B0-dev-v2 | PASS | 80 epochs, no holdout leakage |
| WP10 | Development runs | STOP | V2-1 и V2-2 не прошли gate |
| WP11 | Holdout | BLOCKED BY PROTOCOL | Не открыт после dev STOP |
| WP12 | Final seeds | BLOCKED BY PROTOCOL | Не разрешены |
| WP13 | Final test/statistics | BLOCKED/PARTIAL | Test закрыт; dev paired bootstrap выполнен |
| WP14 | Failure/intervention analysis | PASS ON DEV | 3 000 samples, четыре группы и 80 cases |

## 7. WP1: preflight diagnostics D1-D3

Итоговый preflight status:

```text
CONTINUE_FULL
selected blank penalty = -0.4
selected alpha_max = 0.25
allow V2-2 = true
```

### 7.1. D1: blank-penalty sweep

| Penalty | M0-FT CER | M3 CER |
|---:|---:|---:|
| -0.8 | 0.079792 | 0.082217 |
| -0.6 | 0.079771 | 0.082302 |
| -0.5 | 0.079559 | 0.082153 |
| -0.4 | **0.079537** | 0.082196 |
| -0.3 | 0.079559 | 0.082174 |
| -0.2 | 0.079601 | 0.082238 |
| 0.0 | 0.079686 | 0.082323 |

Вывод D1:

- decode calibration не объясняет отрицательный результат v1;
- `-0.4` сохранен как единый fixed penalty;
- официальный вывод v1 не изменен.

### 7.2. D2: graph-strength sweep v1 M3

| Scale | CER | WER | Exact |
|---:|---:|---:|---:|
| 0.00 | **0.081387** | 0.332531 | 0.625000 |
| 0.10 | 0.143762 | 0.516306 | 0.433500 |
| 0.25 | 0.125941 | 0.471665 | 0.478667 |
| 0.50 | 0.100591 | 0.395349 | 0.558833 |
| 0.75 | 0.087342 | 0.355386 | 0.601833 |
| 1.00 | 0.082196 | 0.337477 | 0.620667 |

Лучшее значение внутри inference sweep: `scale=0`.

Важно: `scale=0` является visual path внутри уже jointly fine-tuned M3, а не
fair `M0-FT` baseline. Поэтому это диагностирует избыточную силу v1 residual,
но не заменяет matched image-only сравнение.

D2 gain относительно v1 `scale=1`:

```text
0.000808 CER
```

Это послужило основанием ограничить v2 параметром:

```text
alpha_max = 0.25
```

### 7.3. D3: strict empty-bin mask

| Вариант | CER | WER | Exact |
|---|---:|---:|---:|
| M3 original | 0.082196 | 0.337477 | 0.620667 |
| M3 strict mask | 0.082387 | 0.336675 | 0.619167 |
| M3 strict shuffle | 0.085449 | 0.350174 | 0.606333 |

Strict mask:

- полностью устранил empty-bin contribution:
  `9.681376 -> 0.000000`;
- не улучшил CER:
  D3 gain `-0.000191`;
- сохранил correct-vs-shuffle signal.

Таким образом, пустые bins были реальной технической проблемой v1, но их
исправление само по себе не обеспечило распознавательный выигрыш.

## 8. WP2: independent train/dev/holdout split

Split seed:

```text
20260730
```

Размеры:

| Split | Total | Cyrillic | HKR | School |
|---|---:|---:|---:|---:|
| train | 35 498 | 8 500 | 8 500 | 18 498 |
| dev | 3 000 | 1 000 | 1 000 | 1 000 |
| holdout | 1 500 | 500 | 500 | 500 |

Group resolution:

```text
writer_id
-> page_id
-> source_group / line_group
-> normalized source path
-> sample_id fallback
```

Leakage audit:

| Проверка | train/dev | train/holdout | dev/holdout |
|---|---:|---:|---:|
| sample ID overlap | 0 | 0 | 0 |
| image path overlap | 0 | 0 | 0 |
| hierarchy group overlap | 0 | 0 | 0 |
| exact image SHA1 overlap | 0 | 0 | 0 |

Дополнительно:

- duplicate sample IDs: 0 на всех splits;
- missing images: 0;
- missing x-aligned features: 0;
- perceptual near-duplicate audit отсутствует как frozen infrastructure;
- exact SHA1 audit выполнен и зафиксирован.

Holdout не участвовал:

- в обучении B0;
- в обучении V2;
- в checkpoint selection;
- в выборе `alpha_max`;
- в выборе `lambda_pres`;
- в stopping decision.

## 9. WP3: features, normalizer и structural risk

### 9.1. Переиспользованные x-aligned records

V2 сохраняет те же 20 признаков `hi_csg_r_xaligned_v1`.

| № | Признак | Группа |
|---:|---|---|
| 1 | `ink_fraction` | geometry |
| 2 | `skeleton_density` | geometry |
| 3 | `edge_length_density` | geometry |
| 4 | `stroke_width_mean` | geometry |
| 5 | `stroke_width_std` | geometry |
| 6 | `curvature_mean` | geometry |
| 7 | `orientation_horizontal` | geometry |
| 8 | `orientation_vertical` | geometry |
| 9 | `orientation_diag_pos` | geometry |
| 10 | `orientation_diag_neg` | geometry |
| 11 | `node_density` | topology |
| 12 | `endpoint_density` | topology |
| 13 | `junction_density` | topology |
| 14 | `loop_edge_fraction` | topology |
| 15 | `component_count_norm` | topology/risk |
| 16 | `short_branch_fraction` | topology/risk |
| 17 | `boundary_crossings_norm` | topology |
| 18 | `ambiguous_edge_fraction` | diagnostic |
| 19 | `graph_occupancy` | presence |
| 20 | `warning_density` | risk |

### 9.2. Feature coverage

| Split | Records | Valid bins | Non-finite | Name mismatches |
|---|---:|---:|---:|---:|
| train | 35 498 | 2 575 073 | 0 | 0 |
| dev | 3 000 | 229 404 | 0 | 0 |
| holdout | 1 500 | 120 132 | 0 | 0 |

Общий record count:

```text
39 998
```

### 9.3. Train-only normalizer

Normalizer fit выполнен только по `adapter_v2_train`.

Преобразование:

```text
z = (x - mean_train) / max(std_train, 1e-6)
z = clip(z, -5, 5)
```

После преобразования применяется обязательный post-normalization mask.

`ambiguous_edge_fraction` имел ненулевую variance на reduced train, но не
использовался в primary gate. Это соответствует frozen design: risk vector
содержит только:

```text
component_count_norm
short_branch_fraction
warning_density
```

### 9.4. Train-only risk quantiles

| Feature | q05 | q50 | q95 |
|---|---:|---:|---:|
| `component_count_norm` | 6.250000 | 25.000000 | 87.500000 |
| `short_branch_fraction` | 0.000000 | 0.000000 | 0.214758 |
| `warning_density` | 0.000000 | 0.000000 | 0.229038 |

Risk formula:

```text
risk =
  0.30 * component_count_scaled
  + 0.40 * short_branch_fraction
  + 0.30 * warning_density_scaled

reliability = exp(-2 * risk)
```

Этот показатель называется structural-risk attenuation, а не вероятностью
корректности графа.

## 10. WP4: Dataset, collate и masks

Batch возвращает:

```text
images
widths
targets
target_lengths
texts
sample_ids
datasets
raw_graph_features [B,T,20]
normalized_graph_features [B,T,20]
time_mask [B,T]
nonempty_graph_mask [B,T]
padding_mask [B,T]
structural_risk_raw [B,T,3]
```

Семантика:

| Timestep | time_mask | nonempty_graph_mask | padding_mask |
|---|---:|---:|---:|
| real non-empty | 1 | 1 | 0 |
| real empty | 1 | 0 | 0 |
| batch padding | 0 | 0 | 1 |

Обязательный порядок:

```python
x_norm = normalizer.transform(x_raw)
x_norm = x_norm * nonempty_graph_mask[..., None]
```

Реализованные invariants:

- normalized graph input empty bin = 0;
- graph embedding empty bin = 0;
- delta logits empty bin = 0;
- correction logits empty bin = 0;
- padding не входит в pooling и losses;
- collate pads до max output `T`, а не max image width.

Domain-balanced sampler:

- выбирает примерно `1/3` каждого core domain на batch;
- использует фиксированный sampler seed;
- не использует target text или model errors;
- width-matching выполняется после выбора domain streams;
- порядок готовых batches детерминированно перемешивается.

## 11. WP5: модель Late Correction v2

### 11.1. Frozen backbone

Backbone возвращает:

```text
base_logits [B,T,C]
visual_hidden [B,T,Dh]
output_lengths [B]
```

Параметры с `requires_grad=False`:

- CNN;
- visual projection;
- BiLSTM;
- baseline classifier.

Hash backbone во всех V2-run:

```text
02a6772823cb334055060a3025801051bba6344134568b7664d26ed4b542195a
```

Hash до и после обучения совпал.

### 11.2. Masked multiscale pooling

Использованы окна:

```text
local  = 1
medium = 5
wide   = 9
```

Для каждого окна:

```text
numerator   = conv(features * mask)
denominator = conv(mask).clamp_min(eps)
pooled      = numerator / denominator
```

Выход:

```text
20 + 20 + 20 = 60 features per timestep
```

### 11.3. Temporal graph adapter

```text
LayerNorm(60)
Linear(60,96)
GELU
Dropout(0.10)
Conv1d(96,96,kernel=3,padding=1)
GELU
Dropout(0.10)
Linear(96,128)
LayerNorm(128)
```

### 11.4. Visual uncertainty

Вычисляется только из detached frozen baseline logits:

```text
p = softmax(Z_base.detach())

entropy_norm = -sum(p log p) / log(C)
margin_uncertainty = 1 - (p_top1 - p_top2)
u = clip(0.5 * entropy_norm + 0.5 * margin_uncertainty, 0, 1)
```

Target, dataset ID и error label не используются.

### 11.5. Gate

V2-1:

```text
gate =
  time_mask
  * nonempty_graph_mask
  * visual_uncertainty
  * learned_gate
```

V2-2:

```text
gate =
  time_mask
  * nonempty_graph_mask
  * visual_uncertainty
  * structural_risk_attenuation
  * learned_gate
```

Обязательные свойства:

```text
padding -> gate = 0
empty graph bin -> gate = 0
uncertainty = 0 -> gate = 0
```

### 11.6. Correction head и bounded alpha

```text
concat(visual_hidden.detach(), graph_embedding)
-> LayerNorm
-> Linear(...,128)
-> GELU
-> Dropout(0.10)
-> Linear(128,C), zero initialized
```

```text
alpha = 0.25 * sigmoid(alpha_logit)
alpha_logit_init = -6
```

Initial equivalence:

```text
max_abs(Z_final - Z_base) < 1e-6
```

### 11.7. Параметры

| Компонент | Параметры |
|---|---:|
| Image-only CRNN-CTC | 3 900 892 |
| Trainable v2 module | 196 156 |
| Full wrapped model | 4 097 048 |
| Relative increase | 5.028% |

Ограничение `<400 000` выполнено. Предпочтительное `<250 000` также выполнено.

## 12. WP6: тесты и статическая проверка

Обязательные v2-тесты:

1. group overlap = 0;
2. normalization fit train-only;
3. raw zero после normalization без mask становится ненулевым;
4. post-normalization mask возвращает empty bin в ноль;
5. masked pooling игнорирует padding;
6. empty-bin correction строго нулевая;
7. alpha bounded в `[0, alpha_max]`;
8. alpha=0 дает baseline logits;
9. correction output zero-initialized;
10. backbone frozen;
11. graph module получает gradient;
12. preservation KL равен нулю для одинаковых logits;
13. uncertainty находится в `[0,1]`;
14. risk attenuation находится в `(0,1]`;
15. shuffle меняет только graph inputs;
16. checkpoint round-trip сохраняет logits;
17. padded timesteps не влияют на loss;
18. intervention metrics конечны;
19. domain-balanced sampler воспроизводим;
20. auxiliary schedule равен нулю после epoch 6;
21. sampler соблюдает domain balance;
22. width matching снижает padding без изменения sample selection;
23. selected candidate STOP не открывает holdout;
24. два STOP-варианта дают terminal negative report status.

Результаты:

```text
pytest -q tests/test_htr_adapter_v2_*.py
24 passed

pytest -q
39 passed
```

Targeted Ruff для всех v2 modules/tools/tests:

```text
All checks passed
```

Полный исторический репозиторий содержит 237 ранее существовавших Ruff
нарушений вне v2. Они не были массово исправлены, поскольку это создало бы
несвязанный refactor и metadata churn.

## 13. Notebook и воспроизводимость

Основной notebook:

```text
notebooks/htr_adapter_v2_late_correction_full_experiment.ipynb
```

Фактический execution audit:

| Показатель | Значение |
|---|---:|
| Всего cells | 33 |
| Code cells | 16 |
| Executed code cells | 16 |
| Output blocks | 901 |
| Error outputs | 0 |

Notebook:

- автоматически находит repository root из `notebooks/`;
- использует `.venv/bin/python`;
- выводит subprocess stdout по мере выполнения;
- сохраняет отдельные logs;
- повторно использует завершенные artifacts;
- поддерживает resume незавершенного training run;
- принимает scientific `exit=2` как STOP, а не software failure;
- не выполняет holdout/test при `selected_candidate.status=STOP`.

Notebook сохраняет:

- preflight tables;
- split/feature audits;
- smoke gate;
- B0 history;
- V2-1/V2-2 histories;
- correct/shuffle/zero summaries;
- gate decisions;
- selection STOP;
- paired statistics;
- figures и final report.

## 14. WP8: smoke и overfit

Smoke gate status:

```text
PASS
```

Условия:

| Condition | Result |
|---|---|
| one-sample CER zero | PASS |
| finite losses | PASS |
| graph gradient nonzero | PASS |
| gate nonconstant | PASS |
| correct not worse than shuffle | PASS |
| empty correction zero | PASS |
| backbone unchanged | PASS |
| no blank collapse | PASS |

One-sample:

```text
CER = 0
Exact = 1
```

128-sample:

```text
CER = 0
Exact = 1
gate std = 0.031591
correct CER = 0
shuffle CER = 0
empty correction max = 0
blank ratio = 0.768160
```

Smoke подтверждает техническую обучаемость, но не является доказательством
H4-v2.

## 15. WP9: fresh B0-dev-v2

### 15.1. Leakage control

`B0-dev-v2` обучен с нуля только на reduced train `35 498`.

Не использованы:

- adapter_v2_holdout;
- canonical v1 validation для development selection;
- canonical checkpoint, обученный на holdout samples.

### 15.2. Training

| Параметр | Значение |
|---|---:|
| Seed | 42 |
| Epochs completed | 80 |
| Runtime | 14 200 s |
| Parameters | 3 900 892 |
| Device | CUDA |
| Torch | 2.6.0+cu124 |

Лучший checkpoint по epoch-specific development evaluation:

```text
epoch = 76
micro-CER = 0.139475
macro-CER = 0.129285
WER = 0.480571
Exact = 0.423667
```

Best checkpoint SHA256:

```text
77551ce69064621b844b5870d7d04aa93ce7d4a639d8d7e4dfc021a708c5ff2d
```

### 15.3. Fixed-penalty baseline для fair V2 comparison

Во время B0 training blank penalty следовал canonical schedule. Best checkpoint
был выбран на эпохе 76 при penalty около `-0.481`.

Все V2 correct/shuffle/zero evaluations используют frozen final penalty
`-0.4`. Поэтому fair baseline, вычисленный внутри V2 evaluator из того же
backbone, равен:

| Metric | Fixed-penalty B0 |
|---|---:|
| micro-CER | 0.139370 |
| macro-CER | 0.129354 |
| WER | 0.479907 |
| Exact | 0.423000 |
| Char edits | 5 292 / 37 971 |
| Word edits | 2 890 / 6 022 |

Именно этот fixed-penalty baseline используется во всех gate decisions.

### 15.4. Domain metrics B0 checkpoint

| Dataset | N | CER | WER | Exact |
|---|---:|---:|---:|---:|
| Cyrillic | 1 000 | 0.122563 | 0.479391 | 0.515000 |
| HKR | 1 000 | 0.108238 | 0.405459 | 0.369000 |
| School clean | 483 | 0.108136 | 0.388430 | 0.619048 |
| School line | 517 | 0.175833 | 0.562526 | 0.170213 |

### 15.5. Invalidation provenance

Два ранних sampler-run не являются научными результатами.

**Invalidated width-batched partial run:**

- нарушал frozen requirement domain-balanced batches;
- остановлен после 11/80 epochs;
- ни один V2 module из него не обучался;
- сохранен только для provenance.

**Aborted random-width domain-balanced run:**

- остановлен до завершения первой эпохи;
- checkpoint и validation result отсутствуют;
- padding overhead был `2.1766x`.

Финальный sampler:

- сохраняет domain-balanced sample selection;
- width-matches три domain streams;
- padding overhead снижен до `1.7564x`;
- научную конфигурацию не меняет.

Изменения зафиксированы protocol amendments 001 и 002 до V2 training.

## 16. WP10: обучение development variants

### 16.1. V2-1-dev-p05

```text
variant = v2_1
lambda_pres = 0.05
risk attenuation = disabled
seed = 42
```

Training:

| Показатель | Значение |
|---|---:|
| Best epoch | 1 |
| Epochs completed | 8 |
| Runtime | 667.07 s |
| Best development CER | 0.139106 |
| Trainable parameters | 196 156 |
| Backbone unchanged | true |

Best checkpoint SHA256:

```text
b9b63c54265ab3b1e0d55aceb33f7ed68ffd2bcbcea906fdfd5adeba1826ea61
```

### 16.2. V2-2-dev-p05

```text
variant = v2_2
lambda_pres = 0.05
risk attenuation = enabled
seed = 42
```

Training:

| Показатель | Значение |
|---|---:|
| Best epoch | 3 |
| Epochs completed | 8 |
| Runtime | 665.01 s |
| Best development CER | 0.139053 |
| Trainable parameters | 196 156 |
| Backbone unchanged | true |

Best checkpoint SHA256:

```text
b66f4ba89a747a518580b14adb4aefe7c34e6d696d27411da5fac13adc2c1da7
```

Оба run завершились minimum epoch 8 по заранее заданному early stopping
patience. Training budget после просмотра метрик не расширялся.

## 17. Полные development метрики

### 17.1. Overall

| Вариант | micro-CER | macro-CER | WER | Exact | Char edits | Word edits |
|---|---:|---:|---:|---:|---:|---:|
| B0 fixed `-0.4` | 0.139370 | 0.129354 | 0.479907 | 0.423000 | 5 292 | 2 890 |
| V2-1 correct | 0.139106 | 0.129361 | 0.481402 | 0.422667 | 5 282 | 2 899 |
| V2-1 shuffle | 0.139106 | 0.129441 | 0.480737 | 0.423000 | 5 282 | 2 895 |
| V2-1 zero | 0.139370 | 0.129354 | 0.479907 | 0.423000 | 5 292 | 2 890 |
| V2-2 correct | 0.139053 | 0.129579 | 0.482066 | 0.423000 | 5 280 | 2 903 |
| V2-2 shuffle | **0.138711** | 0.129654 | 0.481402 | 0.422667 | 5 267 | 2 899 |
| V2-2 zero | 0.139370 | 0.129354 | 0.479907 | 0.423000 | 5 292 | 2 890 |

Наблюдения:

- V2-1 уменьшает 10 character edits, но добавляет 9 word edits;
- V2-2 уменьшает 12 character edits, но добавляет 13 word edits;
- best raw CER принадлежит shuffled V2-2, а не correct graph;
- zero graph точно воспроизводит B0 metrics;
- small CER deltas практически не меняют Exact.

### 17.2. Дельты против B0

| Вариант | Absolute ΔCER | Relative improvement | ΔWER | ΔExact |
|---|---:|---:|---:|---:|
| V2-1 correct | -0.000263 | 0.189% | +0.001495 | -0.000333 |
| V2-2 correct | -0.000316 | 0.227% | +0.002159 | 0.000000 |

Оба relative improvements меньше обязательного `1%`.

## 18. Domain-wise development metrics

Core-domain comparison:

| Domain | B0 CER | V2-1 CER | Δ V2-1 | V2-2 CER | Δ V2-2 |
|---|---:|---:|---:|---:|---:|
| Cyrillic | 0.122830 | 0.123364 | +0.000534 | 0.123097 | +0.000267 |
| HKR | 0.108410 | 0.109183 | +0.000773 | 0.109355 | +0.000945 |
| School | 0.165074 | 0.163854 | -0.001221 | 0.163747 | -0.001327 |

V2-2 detailed:

| Domain | N | CER | macro-CER | WER | Exact |
|---|---:|---:|---:|---:|---:|
| Cyrillic | 1 000 | 0.123097 | 0.130057 | 0.479391 | 0.516 |
| HKR | 1 000 | 0.109355 | 0.124384 | 0.409926 | 0.361 |
| School | 1 000 | 0.163747 | 0.134297 | 0.533379 | 0.392 |

Интерпретация:

- оба graph variants слегка улучшают School;
- оба ухудшают Cyrillic и HKR по CER;
- ухудшения меньше hard threshold `0.003`;
- overall small gain обусловлен главным образом более длинными School samples;
- нельзя заявлять устойчивое multi-domain improvement.

## 19. Length-bucket analysis V2-2 vs B0

| Target length | Samples | ΔCER |
|---|---:|---:|
| 1-5 | 565 | -0.001836 |
| 6-10 | 1 195 | +0.001403 |
| 11-20 | 821 | +0.000085 |
| 21+ | 419 | -0.001492 |

Эффект меняет знак между length buckets. Это не поддерживает гипотезу
равномерной структурной пользы.

## 20. Correct, shuffle и zero dependency controls

### 20.1. V2-1

```text
correct CER = 0.139106
shuffle CER = 0.139106
zero CER    = 0.139370
```

Correct и shuffle имеют одинаковый micro-CER. Это нарушает обязательный
sample-specific criterion.

### 20.2. V2-2

```text
correct CER = 0.139053
shuffle CER = 0.138711
zero CER    = 0.139370
```

Shuffled graph лучше correct graph на `0.000342 CER`.

Вывод:

- graph-conditioned correction меняет predictions;
- zero dependency check подтверждает архитектурную эквивалентность B0;
- но правильная sample-specific структура не превосходит matched чужую;
- observed small gain нельзя приписать именно HI-CSG-R topology конкретного
  образца.

## 21. Paired bootstrap statistics

### 21.1. V2-2 correct vs B0

| Показатель | Значение |
|---|---:|
| Samples | 3 000 |
| Seeds | 1 |
| Baseline CER | 0.139370 |
| V2-2 CER | 0.139053 |
| ΔCER | -0.000316 |
| Relative delta | -0.227% |
| CI95 | [-0.001296, +0.000663] |
| Two-sided p | 0.544946 |
| Wins | 140 |
| Losses | 134 |
| Ties | 2 726 |
| ΔWER | +0.002159 |
| ΔExact | 0.000000 |

CI пересекает ноль. Practical delta значительно меньше development threshold.

### 21.2. V2-2 correct vs shuffled

| Показатель | Значение |
|---|---:|
| Shuffled CER | 0.138711 |
| Correct CER | 0.139053 |
| Correct - shuffle | +0.000342 |
| Relative delta | +0.247% |
| CI95 | [-0.000188, +0.000892] |
| Two-sided p | 0.239376 |
| Wins | 40 |
| Losses | 50 |
| Ties | 2 910 |

Correct graph не показал статистического или практического преимущества.

Holm correction для final primary family не рассчитывалась, поскольку
holdout gate не был пройден и final test family не существовала. Выполнять
Holm над post-stop exploratory comparisons как замену final statistics было бы
методологически неверно.

## 22. Gate, uncertainty и correction diagnostics

Ниже приведены показатели лучшего correct-CER варианта `V2-2`.

### 22.1. Alpha

```text
alpha_max = 0.25
learned alpha = 0.001067763
```

Модель сохранила очень малую глобальную силу вмешательства. Alpha не равен
нулю, но составляет около `0.427%` от разрешенного maximum.

### 22.2. Gate distribution

| Metric | Value |
|---|---:|
| Mean | 0.014766 |
| Std | 0.057859 |
| P10 | 1.85e-11 |
| P50 | 4.03e-7 |
| P90 | 0.016581 |
| Max | 0.795233 |
| Empty-bin mean | **0.000000** |
| Non-empty-bin mean | 0.015530 |

Gate по доменам:

| Domain | Mean gate |
|---|---:|
| Cyrillic | 0.011891 |
| HKR | 0.010841 |
| School | 0.022463 |

Gate по uncertainty terciles:

| Tercile | Mean gate |
|---|---:|
| Low | 2.69e-9 |
| Middle | 2.68e-6 |
| High | 0.044294 |

Gate соответствует intended uncertainty-aware behavior и не открыт
повсеместно, в отличие от v1 mean gate около `0.408`.

### 22.3. Visual uncertainty

| Metric | Value |
|---|---:|
| Mean | 0.024387 |
| Std | 0.091129 |
| P10 | 1.43e-10 |
| P50 | 7.65e-7 |
| P90 | 0.029704 |
| Max | 0.864210 |
| Blank argmax frames | 0.012213 |
| Nonblank argmax frames | 0.072582 |
| Baseline-correct samples | 0.013665 |
| Baseline-error samples | 0.032780 |

Uncertainty связана с ошибками baseline в ожидаемом направлении.

### 22.4. Structural risk

| Metric | Value |
|---|---:|
| Mean | 0.161421 |
| Std | 0.152065 |
| P10 | 0.046154 |
| P50 | 0.069231 |
| P90 | 0.398590 |
| Max | 0.930769 |

### 22.5. Correction magnitude

| Metric | Value |
|---|---:|
| Mean absolute correction | 0.035461 |
| P90 absolute correction | 0.039803 |
| Max absolute correction | 2.048553 |
| Mean correction L2 | 0.110738 |
| Mean base logit L2 | 131.074081 |
| Correction/base L2 ratio | 0.000845 |
| Empty correction max | **0.000000** |

### 22.6. Intervention

| Metric | Value |
|---|---:|
| Top-1 frame change rate | 0.4259% |
| Decoded prediction change rate | 12.7667% |
| Gate > 0.05 | 7.0518% |
| Gate > 0.15 | 3.7497% |
| Improved samples | 140 |
| Hurt samples | 134 |
| Intervention precision | 36.5535% |

Changed predictions имеют:

```text
CER = 0.263044
WER = 0.747860
Exact = 0.073107
```

Unchanged predictions:

```text
CER = 0.104171
WER = 0.409964
Exact = 0.474207
```

Модуль действительно фокусируется на сложных samples, но не умеет достаточно
надежно отличать полезную коррекцию от вредной.

## 23. Failure/intervention analysis

Группы:

| Group | N | CER | Mean uncertainty | Mean gate | Mean correction L2 |
|---|---:|---:|---:|---:|---:|
| A: baseline correct, V2 wrong | 28 | 0.093023 | 0.023642 | 0.017236 | 0.129373 |
| B: baseline wrong, V2 correct | 28 | 0.000000 | 0.024008 | 0.016328 | 0.125113 |
| C: both wrong | 1 703 | 0.199665 | 0.032925 | 0.018706 | 0.140611 |
| D: both correct | 1 241 | 0.000000 | 0.013440 | 0.009842 | 0.072441 |

Ключевой результат:

```text
baseline correct -> V2 wrong: 28
baseline wrong -> V2 correct: 28
```

Число exact исправлений и exact повреждений одинаково.

Сохранены:

- 20 `graph_helps`;
- 20 `graph_hurts`;
- 20 `high_intervention_unchanged`;
- 20 `low_intervention_errors`.

Артефакты:

```text
outputs/htr_adapter_v2/failure_analysis/graph_helps.jsonl
outputs/htr_adapter_v2/failure_analysis/graph_hurts.jsonl
outputs/htr_adapter_v2/failure_analysis/high_intervention_unchanged.jsonl
outputs/htr_adapter_v2/failure_analysis/low_intervention_errors.jsonl
outputs/htr_adapter_v2/final_report/figure_d_helps_hurts.png
```

## 24. Development gate и stopping decision

### 24.1. V2-1 decision

| Condition | Result |
|---|---|
| Relative CER improvement >= 1% | FAIL |
| Correct better than shuffle | FAIL |
| Empty correction invariant | PASS |
| No domain degradation > 0.003 | PASS |
| Exact drop <= 0.005 | PASS |

Decision:

```text
STOP
```

V2-2 был разрешен как последняя preflight-approved risk attenuation проверка,
поскольку V2-1 не ухудшился более чем на 1% relative.

### 24.2. V2-2 decision

| Condition | Result |
|---|---|
| Relative CER improvement >= 1% | FAIL |
| Correct better than shuffle | FAIL |
| Empty correction invariant | PASS |
| No domain degradation > 0.003 | PASS |
| Exact drop <= 0.005 | PASS |

Decision:

```text
STOP
```

### 24.3. Candidate selection

```text
selection metric = development micro-CER
holdout used = false
test used = false
selected candidate = null
selection status = STOP
```

Оба candidates записаны в selection manifest вместе с checkpoint SHA256 и
gate conditions.

## 25. Что намеренно не запускалось

### 25.1. P10 repeat

Не запускался.

Причина:

```text
разрешен только для лучшего PASS-кандидата;
PASS-кандидатов нет.
```

### 25.2. Independent holdout

Не открывался.

Зафиксировано:

```text
Status: NOT_EVALUATED_PROTOCOL_STOP
```

Holdout не использовался для изменения scientific configuration.

### 25.3. Final seeds

Не запускались:

```text
M0-FT-final seeds 42/43/44
V2-final seeds 42/43/44
```

Причина: holdout positive gate отсутствует.

### 25.4. Canonical test

Не открывался:

- mixed test;
- Cyrillic test;
- HKR test;
- School test;
- page-disjoint HKR+School;
- robustness package.

Test predictions и test-derived calibration для v2 отсутствуют.

## 26. Definition of Done

| Требование | Статус | Комментарий |
|---|---|---|
| V1 сохранен неизменным | PASS | Отрицательный вывод v1 сохранен |
| Protocol v2 заморожен | PASS | Freeze и amendments сохранены |
| Preflight выполнен | PASS | D1-D3 завершены |
| Alpha max и blank penalty зафиксированы | PASS | `0.25`, `-0.4` |
| Fresh split создан | PASS | `35 498 / 3 000 / 1 500` |
| Split audit | PASS | Все exact overlaps 0 |
| Fresh B0-dev без leakage | PASS | 80 epochs |
| Post-normalization masking | PASS | Empty input/correction 0 |
| Masked multiscale pooling | PASS | Kernels 1/5/9 |
| Frozen backbone wrapper | PASS | Hash неизменен |
| Uncertainty estimator | PASS | Entropy + margin |
| Risk attenuation | PASS | Fixed formula |
| Late correction head | PASS | Zero-init output |
| Bounded alpha | PASS | `[0,0.25]` |
| Preservation KL | PASS | Confidence-weighted |
| Auxiliary annealing | PASS | 0 после epoch 6 |
| Минимум 20 tests | PASS | 24 v2 tests |
| Smoke/overfit | PASS | 8/8 |
| Development run limit | PASS | B0 + V2-1 + V2-2 |
| Selected config до holdout | STOP/NOT APPLICABLE | Нет PASS candidate |
| Holdout оценен один раз | BLOCKED BY PROTOCOL | Не открыт |
| При STOP test не открыт | PASS | Test не использован |
| Final seeds при PASS | BLOCKED BY PROTOCOL | Holdout PASS отсутствует |
| Final test при PASS | BLOCKED BY PROTOCOL | Не разрешен |
| Paired statistics | PASS ON DEV | 10 000 bootstrap iterations |
| Correct/shuffle control | PASS | Выполнен для V2-1/V2-2 |
| Failure analysis | PASS | 3 000 samples, 80 cases |
| Final report | PASS | Human и machine-readable |
| H4-v2 классифицирована | PASS | `not_supported` |
| Negative results сохранены | PASS | V1 и V2 отражены отдельно |

Эксперимент завершен в полном соответствии с conditional Definition of Done:
последующие стадии не должны выполняться после development STOP.

## 27. Основные артефакты

### 27.1. Human-readable

```text
outputs/htr_adapter_v2/preflight/preflight_report.md
outputs/htr_adapter_v2/split_audit/split_audit.md
outputs/htr_adapter_v2/feature_audit/feature_audit.md
outputs/htr_adapter_v2/smoke/smoke_gate.md
outputs/htr_adapter_v2/development/v2_1_dev_p05_seed42/decision/dev_decision.md
outputs/htr_adapter_v2/development/v2_2_dev_p05_seed42/decision/dev_decision.md
outputs/htr_adapter_v2/development/selected_candidate.md
outputs/htr_adapter_v2/final_report/final_results.md
outputs/htr_adapter_v2/final_report/full_execution_report_ru.md
outputs/htr_adapter_v2/final_report/holdout_decision.md
outputs/htr_adapter_v2/final_report/failure_analysis.md
outputs/htr_adapter_v2/final_report/limitations.md
outputs/htr_adapter_v2/final_report/method_and_results_sections_ru.md
```

### 27.2. Machine-readable

```text
outputs/htr_adapter_v2/preflight/preflight_report.json
outputs/htr_adapter_v2/split_audit/split_audit.json
outputs/htr_adapter_v2/feature_audit/feature_audit.json
outputs/htr_adapter_v2/smoke/smoke_gate.json
outputs/htr_adapter_v2/b0_dev_seed42/train_summary.json
outputs/htr_adapter_v2/v2_1_dev_p05_seed42/train_summary.json
outputs/htr_adapter_v2/v2_2_dev_p05_seed42/train_summary.json
outputs/htr_adapter_v2/development/p05_selection.json
outputs/htr_adapter_v2/development/selected_candidate.json
outputs/htr_adapter_v2/statistical_analysis/dev_v2_2_vs_b0_bootstrap.json
outputs/htr_adapter_v2/statistical_analysis/dev_v2_2_correct_vs_shuffle_bootstrap.json
outputs/htr_adapter_v2/failure_analysis/failure_analysis.json
outputs/htr_adapter_v2/final_report/final_report.json
outputs/htr_adapter_v2/final_report/final_evidence_manifest.json
outputs/htr_adapter_v2/final_report/SHA256SUMS
```

### 27.3. Checkpoints

```text
outputs/htr_adapter_v2/b0_dev_seed42/best.pt
outputs/htr_adapter_v2/v2_1_dev_p05_seed42/best.pt
outputs/htr_adapter_v2/v2_2_dev_p05_seed42/best.pt
```

### 27.4. Figures

| Figure | Файл |
|---|---|
| A: architecture | `figure_a_architecture.png` |
| B: intervention | `figure_b_intervention.png` |
| C: development results | `figure_c_results.png` |
| D: helps/hurts | `figure_d_helps_hurts.png` |

### 27.5. Integrity

Evidence manifest содержит path, SHA256 и размер каждого включенного
артефакта.

Проверка:

```text
sha256sum -c outputs/htr_adapter_v2/final_report/SHA256SUMS
all listed artifacts: OK
```

## 28. Научная интерпретация

### 28.1. Что технически доказано

V2 устранил основные неоднозначности v1:

- graph correction больше не распространяется через BiLSTM;
- visual backbone не меняется;
- empty bins не создают residual после standardization;
- gate ограничен visual uncertainty;
- V2-2 отдельно ослабляет high-risk structure;
- global alpha удерживает correction малой;
- auxiliary graph CTC выключается после epoch 6;
- zero graph точно возвращает baseline.

Следовательно, отрицательный результат нельзя объяснить:

- изменением visual CNN/BiLSTM;
- ненулевым empty-bin signal;
- отсутствием gradient flow;
- полным закрытием graph branch;
- несоответствием CTC lengths;
- holdout/test leakage.

### 28.2. Что не доказано

Нельзя утверждать:

- что HI-CSG-R улучшает CRNN-CTC;
- что correct local topology полезнее matched чужой topology;
- что structural-risk proxy оценивает истинное качество графа;
- что небольшое снижение dev CER воспроизводится по seeds;
- что результат переносится на holdout, test или page-disjoint data;
- что v2 подтверждает восстановление траектории пера.

### 28.3. Почему small CER gain недостаточен

Номинальное снижение CER `0.227% relative`:

- ниже заранее заданного порога `1%`;
- имеет CI, пересекающий ноль;
- сопровождается ухудшением WER;
- отсутствует в двух из трех core domains;
- не является sample-specific, потому что shuffle лучше;
- получено на одном development seed.

Поэтому оно трактуется как практически пренебрежимое development fluctuation,
а не как подтверждение H4-v2.

### 28.4. Диагностическая роль HI-CSG-R

Несмотря на отрицательный recognition result, HI-CSG-R остается полезен для:

- анализа локальной структуры;
- стратификации по graph occupancy;
- визуализации endpoints, junctions, loops и short branches;
- анализа сложных School samples;
- диагностики случаев, где baseline uncertainty повышена.

Это диагностическая, а не доказанная распознавательная роль.

## 29. Ограничения

- Development conclusion основан на seed 42, поскольку protocol остановил
  дальнейшие seeds до holdout.
- Holdout и test намеренно отсутствуют, поэтому нельзя оценивать final
  generalization.
- Perceptual near-duplicate infrastructure отсутствовала; exact SHA1 audit
  выполнен.
- Structural-risk attenuation является фиксированным proxy.
- 20 x-aligned features являются сжатым представлением и могут терять
  информацию исходного графа.
- Matched shuffle ограничивает domain, width и ink properties, но не является
  единственно возможным negative control.
- Negative result относится к текущему extractor, feature builder и трем
  русскоязычным доменам.

## 30. Итоговый вывод

V2 является технически корректным, воспроизводимым и завершенным
экспериментом с отрицательным научным результатом.

Финальная формулировка:

> Несмотря на устранение раннего слияния, пустых-bin вмешательств и обновления
> visual backbone, HI-CSG-R late correction не дала воспроизводимого снижения
> CER. Номинальное улучшение development CER было значительно ниже
> зафиксированного порога, не подтверждалось paired bootstrap и не являлось
> sample-specific, поскольку matched shuffled graph показал не худший
> результат. H4-v2 не подтверждена; holdout и test корректно не открывались.

Объединенный результат v1+v2:

```text
v1:
sample-specific graph signal присутствовал,
но early residual fusion ухудшал matched image-only baseline.

v2:
strict masked bounded late correction технически работала,
но correct graph не превосходил shuffle и practical improvement отсутствовал.
```

Дальнейший архитектурный поиск в рамках этой ветки остановлен.
