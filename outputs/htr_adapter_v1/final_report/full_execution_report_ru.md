# Полный отчет по эксперименту HI-CSG-R x-aligned adapter

**Протокол:** `crnn_ctc_hi_csg_r_adapter_protocol_v1`  
**Версия признаков:** `hi_csg_r_xaligned_v1`  
**Версия feature builder:** `1.0.1`  
**Дата фиксации результата:** 2026-07-30  
**Фактически достигнутый статус:** завершенный отрицательный результат на seed-42 validation  
**Статус H4:** `exploratory`  
**Финальный test использован:** нет

## 1. Краткий итог

Техническая часть локального x-aligned HI-CSG-R adapter реализована и проверена:

- входные checkpoint, manifests, изображения, графы и vocab прошли аудит;
- построено 141 006 x-aligned feature records без failures;
- основной feature audit прошел на 51 561 train/val/test-образце;
- реализованы dataset/collate, temporal adapter, quality-aware gate, auxiliary CTC head, trainer, evaluator, shuffle-контроль и статистические инструменты;
- 15 обязательных unit/integration tests проходят;
- one-sample и 128-sample smoke/overfit gates проходят;
- обучены matched `M0-FT seed42` и полный `M3 seed42`;
- выполнены validation evaluation с правильным и matched shuffled графом;
- validation gate вычислен строго до test.

На seed 42 полный adapter не превзошел matched image-only fine-tuning:

| Модель | micro-CER | macro-CER | WER | Exact |
|---|---:|---:|---:|---:|
| `M0-FT` | 0.079537 | 0.084963 | 0.332398 | 0.626833 |
| `M3`, correct graph | 0.082196 | 0.087691 | 0.337477 | 0.620667 |
| `M3-shuffle` | 0.083004 | 0.088702 | 0.341486 | 0.615167 |

Главное сравнение `M3 - M0-FT`:

- absolute delta CER: `+0.002658`;
- relative CER degradation: `+3.342%`;
- delta WER: `+0.005079`;
- delta Exact: `-0.006167`, или `-0.617` процентного пункта;
- ухудшение CER присутствует во всех трех validation-доменах.

Правильный граф лучше shuffled на `0.000808` CER, gate имеет ненулевую
вариативность, а graph adapter получает градиенты. Следовательно, ветвь не
коллапсировала технически, но не обеспечила требуемого преимущества над
сильным matched baseline.

По заранее зафиксированному stopping rule получен `STOP` на фазе `pre_m2`.
Поэтому `M2`, seeds 43/44, test, page-disjoint и robustness evaluation не
запускались. Это не пропуск исполнения, а обязательное решение протокола,
исключающее подбор по test и продолжение неуспешной архитектурной ветки.

## 2. Зафиксированный экспериментальный протокол

Сохранено исходное ядро:

```text
grayscale image
-> existing CNN
-> visual projection, 256
-> existing BiLSTM
-> existing classifier
-> CTC
```

Добавлен только один модуль:

```text
HI-CSG-R
-> x-aligned sequence [T, 20]
-> temporal graph adapter [T, 256]
-> quality-aware scalar gate
-> residual fusion before existing BiLSTM
```

Фиксированные условия:

| Параметр | Значение |
|---|---|
| Primary metric | validation/test micro-CER |
| Secondary metrics | macro-CER, WER, Exact |
| Seeds | 42, 43, 44 |
| Blank logit penalty | `-0.4` |
| Batch size | 16 |
| Optimizer | AdamW |
| Weight decay | `1e-4` |
| Gradient clipping | `5.0` |
| Warm-up `M3` | 5 epochs |
| Joint fine-tuning | 25 epochs |
| Auxiliary loss | `L_fused_ctc + 0.15 * L_graph_aux_ctc` |
| Checkpoint selection | только validation micro-CER |
| Test tuning | запрещен |

Learning rates:

| Группа | LR |
|---|---:|
| Graph adapter | `3e-4` |
| Gate | `3e-4` |
| Auxiliary graph head | `3e-4` |
| BiLSTM | `5e-5` |
| Classifier | `5e-5` |
| Last CNN block | `1e-5` |

Все семь конфигураций созданы:

- `m0_ft_seed42.yaml`, `m0_ft_seed43.yaml`, `m0_ft_seed44.yaml`;
- `m2_geometry_seed42.yaml`;
- `m3_full_seed42.yaml`, `m3_full_seed43.yaml`, `m3_full_seed44.yaml`.

Каждая seed-конфигурация ссылается на canonical checkpoint того же seed.

## 3. Реализованная кодовая база

### 3.1. Основные модули

| Назначение | Файл |
|---|---|
| Frozen protocol | `docs/crnn_ctc_hi_csg_r_adapter_protocol_v1.md` |
| X-aligned features и normalizer | `src/htr/xaligned_hi_csg_r.py` |
| Dataset и collate | `src/htr/dataset_adapter.py` |
| Adapter, gate, fused CRNN-CTC | `src/htr/model_hi_csg_r_adapter.py` |
| Input audit | `tools/audit_adapter_inputs_v1.py` |
| Feature builder | `tools/build_xaligned_hi_csg_r_features_v1.py` |
| Feature/normalizer audit | `tools/audit_xaligned_features_v1.py` |
| Feature visual browser | `tools/visualize_xaligned_features_v1.py` |
| Trainer | `tools/train_crnn_ctc_hi_csg_r_adapter_v1.py` |
| Evaluator | `tools/evaluate_crnn_ctc_hi_csg_r_adapter_v1.py` |
| Matched shuffle map | `tools/build_hi_csg_r_shuffle_map_v1.py` |
| Validation comparison/gate | `tools/compare_hi_csg_r_adapter_results_v1.py` |
| Paired bootstrap/Holm | `tools/paired_bootstrap_hi_csg_r_adapter_v1.py` |
| Final report generator | `tools/make_hi_csg_r_adapter_final_report_v1.py` |
| Full experiment notebook | `notebooks/htr_adapter_v1_full_experiment.ipynb` |

### 3.2. Реализованные свойства

- `T = max(width // 4, 1)` согласован с output length CRNN.
- Узлы назначаются в x-bin по нормированной координате.
- Ребра распределяются по bins по midpoint каждого элементарного сегмента,
  взвешенному длиной сегмента.
- Фиксированное сглаживание `0.25/0.50/0.25`.
- Реальный пустой bin отличается от batch padding.
- M2 обнуляет признаки 11-20 после нормализации, сохраняя архитектуру.
- Quality gate получает только признаки 18-20.
- Последняя graph projection имеет нулевую инициализацию.
- Финальный bias gate инициализируется значением `-1.5`.
- Strict loader проверяет ключи canonical checkpoint и сохраняет SHA256.
- Feature records пишутся атомарно; поврежденный cache перестраивается.
- Для node-free isolated loops введен детерминированный обход пиксельного
  контура; это исправило ранее обнаруженную ошибку edge-length audit.
- Normalizer разрешен для smoke subset только после явной проверки, что subset
  является точным подмножеством исходного train manifest.

## 4. WP0-WP12: фактический статус

| WP | Требование | Статус | Результат |
|---|---|---|---|
| WP0 | Заморозить протокол/configs | PASS | Документ и 7 configs зафиксированы в git |
| WP1 | Аудит checkpoint/manifests | PASS | Все критические проверки прошли |
| WP2 | Построить x-aligned features | PASS | 141 006/141 006, failures 0 |
| WP3 | Автоматический и визуальный audit | PASS | 51 561 main records и 30 визуальных примеров |
| WP4 | Dataset/collate | PASS | Padding идет до max output T, не image width |
| WP5 | Model/loader | PASS | Adapter, gate, aux CTC, strict provenance |
| WP6 | Unit/integration tests | PASS | 15/15 |
| WP7 | Trainer/evaluator | PASS | Train/eval, logs, checkpoints, metadata |
| WP8 | Smoke/overfit | PASS | Все 6 smoke gate conditions истинны |
| WP9 | Development seed42 | STOP | M3 хуже M0-FT; pre-M2 stop |
| WP10 | Финальные 3 seeds | BLOCKED BY PROTOCOL | Seeds 43/44 запрещены после STOP |
| WP11 | Финальный test | BLOCKED BY PROTOCOL | Test не открыт и не использован |
| WP12 | Статистический анализ | PARTIAL/FINAL NEGATIVE | Validation deltas готовы; paired test CI/Holm неприменимы без test |

## 5. WP1: аудит исходных данных

Итоговый статус: `PASS`.

| Split | Samples | Missing images | Missing graphs | ID mismatch | Duplicates |
|---|---:|---:|---:|---:|---:|
| Train | 39 998 | 0 | 0 | 0 | 0 |
| Validation | 6 000 | 0 | 0 | 0 | 0 |
| Test | 5 563 | 0 | 0 | 0 | 0 |

Пересечения:

- train/val sample overlap: 0;
- train/test sample overlap: 0;
- val/test sample overlap: 0;
- train/val/test image path overlap: 0.

Checkpoint audit:

| Seed | Canonical epoch | Seed match | Model state | Config |
|---:|---:|---|---|---|
| 42 | 73 | PASS | present | present |
| 43 | 76 | PASS | present | present |
| 44 | 79 | PASS | present | present |

Vocab SHA256 во всех checkpoint совпал:

```text
a5914689766f1923c1b7538b73c87dc3efd3f1b08454e60a5b6875120c18710d
```

Полный артефакт: `outputs/htr_adapter_v1/input_audit/report.md`.

## 6. WP2-WP3: x-aligned feature records

### 6.1. Покрытие

Построен 21 manifest:

- main train, validation, test;
- page-disjoint;
- clean-core и hard-real;
- 15 robustness-вариантов: blur, low contrast, noise, thick strokes и thin
  strokes на mild/medium/strong уровнях.

Итог:

| Показатель | Значение |
|---|---:|
| Feature build summaries | 21 |
| Summaries со статусом PASS | 21 |
| Expected records | 141 006 |
| Written records | 141 006 |
| Failures | 0 |
| Feature dimension | 20 |
| Builder version | 1.0.1 |

Robustness features построены из соответствующих distorted images, а не из
clean graph.

### 6.2. Двадцать признаков

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
| 11 | `node_density` | typed topology |
| 12 | `endpoint_density` | typed topology |
| 13 | `junction_density` | typed topology |
| 14 | `loop_edge_fraction` | typed topology |
| 15 | `component_count_norm` | typed topology |
| 16 | `short_branch_fraction` | typed topology |
| 17 | `boundary_crossings_norm` | typed topology |
| 18 | `ambiguous_edge_fraction` | quality |
| 19 | `graph_occupancy` | quality |
| 20 | `warning_density` | quality |

Quality-вектор gate состоит только из признаков 18-20.

### 6.3. Train-only normalizer

- fit выполнен только на enhanced train manifest;
- `clip_value = 5.0`;
- `missing_policy = raw_zero`;
- empty real bin и padded bin имеют разные политики;
- graph version: `hi_csg_r_v1`;
- train manifest SHA256:
  `59962ec63236fe862909a707c4e5c34bed9bf8f61587acbb8f559143e976f893`.

Validation и test не участвовали в fit.

### 6.4. Основной feature audit

| Split | Records | Total bins | Zero-bin fraction | T min | T mean | T max |
|---|---:|---:|---:|---:|---:|---:|
| Train | 39 998 | 2 924 609 | 0.036697 | 9 | 73.119 | 400 |
| Validation | 6 000 | 363 764 | 0.044853 | 16 | 60.627 | 404 |
| Test | 5 563 | 376 176 | 0.044235 | 16 | 67.621 | 297 |

Consistency:

- node count max delta: `0.0` на всех splits;
- endpoint count max delta: `0.0`;
- junction count max delta: `0.0`;
- effective edge-length max delta:
  train `2.64e-11`, val `2.96e-12`, test `5.00e-12`;
- float32 record reconstruction max delta:
  train `2.22e-5`, val `1.52e-5`, test `1.54e-5`;
- NaN/Inf или record failures: 0;
- target-derived field violations: 0.

Аудит корреляций выявляет заметные dataset-correlations у части плотностных
признаков. Это ожидаемый диагностический риск доменного proxy, а не утечка:
dataset ID не подается модели или gate, target-derived поля отсутствуют, а
matched shuffle выполняется внутри того же domain.

Полные min/max/mean/std, zero fractions и correlations всех 20 признаков:
`outputs/htr_adapter_v1/feature_audit/feature_audit.md`.

### 6.5. Визуальный audit

Создан browser на 30 примерах:

- 10 Cyrillic;
- 10 HKR;
- 10 School;
- внутри доменов представлены clean/medium/hard случаи.

Для каждого примера показаны image, foreground, skeleton, graph overlay,
x-bin boundaries и feature curves. Визуальная проверка принята:
endpoint/junction/loop peaks локализованы, edge lengths не смещены в чужие
bins, пустые интервалы и padding не смешаны.

Browser: `outputs/htr_adapter_v1/feature_audit/browser/browser.html`.

## 7. WP4-WP5: данные и модель

Batch содержит:

```text
images, widths,
graph_features [B, Tmax, 20],
graph_quality [B, Tmax, 3],
graph_mask [B, Tmax],
targets, target_lengths,
texts, sample_ids, datasets
```

`Tmax` равен максимальному CRNN output length в batch.

Архитектура adapter:

```text
LayerNorm(20)
Conv1d(20, 64, k=3) -> GELU -> Dropout(0.10)
Conv1d(64, 128, k=3) -> GELU -> Dropout(0.10)
Linear(128, 256) -> LayerNorm(256)
```

Gate:

```text
concat(V[256], G[256], Q[3])
-> Linear(515, 64)
-> GELU
-> Linear(64, 1)
-> sigmoid
```

Параметрический бюджет:

| Модель/часть | Parameters |
|---|---:|
| Base image-only CRNN-CTC | 3 900 892 |
| Adapter + gate + auxiliary head | 119 429 |
| Full M3 | 4 020 321 |
| Relative increase | 3.062% |
| Allowed adapter budget | 400 000 |
| Budget status | PASS |

## 8. WP6: тесты и статическая проверка

Фактический результат:

```text
15 passed
ruff: All checks passed
```

Проверены:

1. node bin assignment;
2. long-edge distribution;
3. canonicalization node-free isolated loop;
4. CRNN output length и feature dimension;
5. fixed smoothing/resampling;
6. train-only normalization и serialization;
7. нормализатор только для проверенного exact train subset;
8. collate padding до output T;
9. topology-off обнуляет признаки 11-20 после normalization;
10. shuffle меняет только graph и корректно resample-ит width;
11. initial equivalence с image-only baseline;
12. отсутствие padding leakage в реальные logits;
13. warm-up freeze и joint gradient flow;
14. strict canonical loader и provenance;
15. checkpoint round-trip сохраняет logits.

## 9. Notebook и воспроизводимость

Главный notebook содержит 39 cells, из них 22 code cells. Исходный notebook
не хранит outputs, что нормально для чистого воспроизводимого source artifact.
Выполненные notebook-артефакты содержат вывод во всех code cells и не содержат
ошибок:

| Выполненный notebook | Code cells with output | Errors |
|---|---:|---:|
| `htr_adapter_v1_prepare.executed.ipynb` | 21/21 | 0 |
| `htr_adapter_v1_smoke.executed.ipynb` | 21/21 | 0 |
| `htr_adapter_v1_seed42.executed.ipynb` | 21/21 | 0 |
| `htr_adapter_v1_report.executed.ipynb` | 22/22 | 0 |
| `htr_adapter_v1_check.executed.ipynb` | 22/22 | 0 |

Notebook разбит на стадии и использует marker-файлы. После `STOP` он не
выбрасывает искусственную ошибку, а формирует отрицательный отчет и явно
показывает, какие дальнейшие стадии заблокированы.

## 10. WP8: smoke и overfit

Итоговый smoke gate: `PASS`.

| Условие | Результат |
|---|---|
| One sample near-zero CER | PASS |
| Auxiliary loss decreases | PASS |
| Adapter gradient nonzero | PASS |
| Gate not constant | PASS |
| No blank collapse | PASS |
| Finite losses | PASS |

### 10.1. Один sample

| Показатель | Значение |
|---|---:|
| Epochs | 5 warm-up + 20 joint |
| Best CER | 0.000000 |
| WER | 0.000000 |
| Exact | 1.000000 |
| Runtime | 8.54 s |
| Device | CUDA |

### 10.2. Auxiliary-only, 128 samples

| Показатель | Начало | Конец |
|---|---:|---:|
| Auxiliary CTC loss | 7.940975 | 2.847737 |
| Auxiliary CER | 0.980534 | 0.922692 |

Graph adapter gradient был ненулевым. Этот run диагностирует обучаемость
ветви, а не качество финального recognizer.

### 10.3. Full fused, 128 samples

| Показатель | Начало | Конец/best |
|---|---:|---:|
| CER | 0.287542 | 0.070078 |
| Total loss | 7.940975 | 0.668044 |
| Blank ratio | - | 0.769950 |
| Gate mean | - | 0.186346 |
| Gate std | - | 0.027767 |
| Exact | - | 0.765625 |
| Runtime | - | 19.84 s |

Stop condition WP8 не сработал: данные, CTC lengths, masks и gradients
признаны пригодными для полного seed42 run.

## 11. WP9: обучение seed 42

| Run | Epochs | Best val CER | Runtime | Device |
|---|---:|---:|---:|---|
| `M0-FT seed42` | 25 joint | 0.079537 | 3131.19 s / 52.19 min | CUDA |
| `M3 seed42` | 5 warm-up + 25 joint | 0.082196 | 3752.34 s / 62.54 min | CUDA |

Среда:

- Python 3.11.12;
- PyTorch 2.6.0+cu124;
- CUDA 12.4;
- Linux 6.12.96.

Для обоих runs сохранены `best.pt`, `last.pt`, `config.json`, history в JSON и
JSONL, train/validation summaries, validation predictions, stdout/stderr и
runtime metadata.

## 12. Seed42 validation: полные основные метрики

Validation содержит 6 000 одинаковых samples, 47 022 target characters и
7 482 target words.

| Метрика | M0-FT | M3 correct | M3 shuffle | Delta M3-M0 |
|---|---:|---:|---:|---:|
| micro-CER | 0.079537 | 0.082196 | 0.083004 | +0.002658 |
| macro-CER | 0.084963 | 0.087691 | 0.088702 | +0.002729 |
| WER | 0.332398 | 0.337477 | 0.341486 | +0.005079 |
| Exact | 0.626833 | 0.620667 | 0.615167 | -0.006167 |
| Char edits | 3 740 | 3 865 | 3 903 | +125 |
| Word edits | 2 487 | 2 525 | 2 555 | +38 |
| Prediction length mean | 7.7870 | 7.7843 | 7.7847 | -0.0027 |
| CTC loss | 0.684212 | 0.671758 | 0.678732 | -0.012453 |
| Blank ratio | 0.832383 | 0.833692 | 0.833532 | +0.001309 |
| Graph auxiliary CTC loss | n/a | 2.708626 | 4.231663 | n/a |
| Graph auxiliary CER | n/a | 0.771277 | 0.915104 | n/a |

Снижение CTC loss у M3 при одновременном ухудшении decoded CER не считается
успехом: primary selection metric по протоколу является micro-CER.

Correct graph против shuffle:

- CER лучше на `0.000808`;
- macro-CER лучше на `0.001011`;
- WER лучше на `0.004010`;
- Exact выше на `0.0055`, или `0.55` процентного пункта;
- self-pairs в shuffle map: 0;
- transcription и model errors для matching не использовались.

## 13. Доменные validation-метрики

| Domain | N | M0 CER | M3 CER | Abs delta | Relative degradation | M0 WER | M3 WER | M0 Exact | M3 Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cyrillic | 2 000 | 0.101731 | 0.103834 | +0.002104 | +2.068% | 0.431560 | 0.426103 | 0.5635 | 0.5680 |
| HKR Words | 2 000 | 0.067098 | 0.069938 | +0.002841 | +4.234% | 0.290787 | 0.298745 | 0.6020 | 0.5900 |
| School | 2 000 | 0.073384 | 0.076418 | +0.003033 | +4.133% | 0.291667 | 0.303571 | 0.7150 | 0.7040 |

M3 ухудшил CER во всех трех core-доменах. Все absolute degradations меньше
порогового ограничения `0.005`, но условие “не ухудшены два или более домена”
не выполнено.

## 14. Метрики по длине target

| Length | N | M0 CER | M3 CER | Abs delta | M0 WER | M3 WER | M0 Exact | M3 Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-5 | 2 066 | 0.089870 | 0.092998 | +0.003128 | 0.261205 | 0.258795 | 0.742498 | 0.744434 |
| 6-10 | 2 634 | 0.084401 | 0.086993 | +0.002592 | 0.386371 | 0.396934 | 0.594533 | 0.580866 |
| 11-20 | 1 252 | 0.068371 | 0.071101 | +0.002730 | 0.324830 | 0.329932 | 0.513578 | 0.510383 |
| 21+ | 48 | 0.086124 | 0.085167 | -0.000957 | 0.391667 | 0.391667 | 0.375000 | 0.354167 |

Единственный CER выигрыш наблюдается в очень малой группе `21+` (`N=48`) и
не компенсирует ухудшение в трех крупных length buckets.

## 15. Метрики по типу token

| Token type | N | M0 CER | M3 CER | Abs delta | M0 WER | M3 WER | M0 Exact | M3 Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alphabetic | 3 885 | 0.078843 | 0.081789 | +0.002946 | 0.335393 | 0.341828 | 0.668726 | 0.661004 |
| Digits | 25 | 0.090909 | 0.090909 | 0.000000 | 0.200000 | 0.200000 | 0.800000 | 0.800000 |
| Mixed | 998 | 0.089827 | 0.094735 | +0.004908 | 0.388778 | 0.404810 | 0.618236 | 0.604208 |
| Multiword | 1 092 | 0.075932 | 0.077040 | +0.001108 | 0.307304 | 0.306138 | 0.481685 | 0.488095 |

## 16. Gate, gradients и graph strata

Gate на M3 validation:

| Диагностика | Значение |
|---|---:|
| Mean | 0.407748 |
| Std | 0.029756 |
| P10 | 0.374099 |
| P50 | 0.403488 |
| P90 | 0.449322 |
| Empty bin | 0.442301 |
| Non-empty bin | 0.401705 |
| Cyrillic mean | 0.408862 |
| HKR mean | 0.403809 |
| School mean | 0.414175 |
| Max graph adapter grad norm | 4.845424 |
| Max gate grad norm | 0.810443 |

Gate не является константой, не закрыт и не насыщен на 0/1. Adapter и gate
получают ненулевые градиенты.

M3 CER по terciles graph diagnostics:

| Diagnostic | Low CER | Medium CER | High CER |
|---|---:|---:|---:|
| Graph occupancy | 0.083031 | 0.079871 | 0.083464 |
| Short-branch fraction | 0.071306 | 0.081127 | 0.095370 |
| Warning density | 0.071994 | 0.081890 | 0.093244 |

Высокие short-branch и warning strata связаны с более тяжелыми samples.
Это диагностическая ассоциация, а не доказательство причинности или пользы
adapter.

`ambiguous_edge_fraction` на validation равен нулю для всех samples, поэтому
его low/high strata пусты и статистически неинформативны.

## 17. Matched shuffled-graph контроль

Shuffle map содержит 6 000 пар:

| Свойство matching | Значение |
|---|---|
| Domain | exact match |
| Width bucket | 8 output steps |
| Ink-fraction bucket | 0.05 |
| Self-pairs | 0 |
| Mean absolute T difference | 2.411 |
| Mean absolute ink difference | 0.01522 |
| Transcription used | no |
| Model error used | no |

Correct graph лучше matched shuffled graph, что указывает на небольшой
sample-specific сигнал. Однако correct M3 остается хуже M0-FT, поэтому этот
сигнал недостаточен для подтверждения H4.

## 18. Validation gate и stopping decision

| Условие pre-gate | Результат |
|---|---|
| M3 лучше M0-FT минимум на 2% relative CER | FAIL |
| Не ухудшены два или более core domains | FAIL |
| Любое domain ухудшение не превышает 0.005 CER | PASS |
| Correct graph лучше shuffled | PASS |
| Gate имеет ненулевую вариативность | PASS |
| Graph adapter gradient не коллапсирует | PASS |

Итог:

```text
status = STOP
phase = pre_m2
decision = M2, seeds 43/44 and test are blocked; H4 is exploratory
```

Причины STOP:

1. M3 не достиг обязательного `>=2%` relative validation improvement;
2. вместо улучшения получено `3.342%` relative degradation;
3. CER ухудшился во всех трех core validation domains.

## 19. Что намеренно не запускалось

Следующие действия заблокированы frozen protocol:

- `M2 geometry/topology-off seed42`;
- `M0-FT seeds 43 and 44`;
- `M3 seeds 43 and 44`;
- финальная mixed test evaluation;
- отдельные test-domain evaluations;
- page-disjoint evaluation;
- robustness recognition evaluation;
- correct-vs-shuffled ablation на test;
- paired test bootstrap confidence interval;
- Holm correction основных test-сравнений;
- трехseedовые mean/SD и seed wins/losses.

Важно различать:

- x-aligned records для test/page-disjoint/robustness построены и
  провалидированы как данные;
- recognition metrics на этих наборах не вычислялись, поскольку открытие
  test после failed validation gate нарушило бы протокол;
- отсутствие M2 соответствует фазе `pre_m2`: исходный порядок разрешал M2
  только после прохождения основного seed42 gate.

Нельзя подставлять вымышленные значения, считать отсутствующие test metrics
нулевыми или продолжать seeds 43/44 задним числом.

## 20. Definition of Done

| Требование исходного DoD | Статус | Комментарий |
|---|---|---|
| Протокол заморожен и закоммичен | DONE | Commit history сохранен |
| Canonical checkpoints 42/43/44 подтверждены | DONE | Seeds, metadata, vocab PASS |
| X-aligned features для основных splits | DONE | Train/val/test PASS |
| Automatic feature audit | DONE | 51 561 records PASS |
| Visual audit минимум 30 | DONE | 30/30 |
| Unit tests | DONE | 15/15 |
| Initial equivalence | DONE | Test PASS |
| One-sample overfit | DONE | CER 0 |
| Small-subset overfit | DONE | Smoke gate PASS |
| M0-FT seed42 | DONE | Best CER 0.079537 |
| M3 seed42 | DONE | Best CER 0.082196 |
| Validation gate | DONE/STOP | Frozen stop применен |
| Seeds 43/44 при positive gate | NOT APPLICABLE | Gate отрицательный |
| Topology-off M2 | NOT APPLICABLE | Blocked at pre-M2 |
| Matched shuffled graph | DONE ON VAL | Correct лучше shuffle |
| Test только после freeze | PRESERVED | Test не использован |
| Paired bootstrap CI | NOT APPLICABLE | Test comparison отсутствует |
| Main 3-seed table | NOT APPLICABLE | 3-seed запуск запрещен |
| Domain-wise table | DONE ON VAL | Test domains не открывались |
| Page-disjoint evaluation | NOT APPLICABLE | Заблокирована |
| Robustness без clean-graph leakage | DATA DONE, EVAL BLOCKED | Distorted features rebuilt |
| Configs/manifests/checkpoints/predictions | DONE FOR EXECUTED RUNS | Seed42 artifacts полные |
| Final report | DONE | Отрицательный отчет сформирован |
| H4 classified | DONE | Exploratory |
| Architecture search stopped | DONE | Новые ветки не добавлялись |

Таким образом, полный положительный трехseedовый DoD не достигнут, но
эксперимент завершен корректно как protocol-compliant negative result.

## 21. Основные артефакты

| Артефакт | Путь |
|---|---|
| Frozen protocol | `docs/crnn_ctc_hi_csg_r_adapter_protocol_v1.md` |
| Input audit | `outputs/htr_adapter_v1/input_audit/report.md` |
| Feature audit | `outputs/htr_adapter_v1/feature_audit/feature_audit.md` |
| Feature browser | `outputs/htr_adapter_v1/feature_audit/browser/browser.html` |
| Train normalizer | `data/experiments/htr_adapter_v1/normalizer/train_stats.json` |
| M0-FT seed42 | `outputs/htr_adapter_v1/m0_ft_seed42/` |
| M3 seed42 | `outputs/htr_adapter_v1/m3_full_seed42/` |
| M0 validation metrics | `outputs/htr_adapter_v1/validation_seed42/m0_ft/` |
| M3 validation metrics | `outputs/htr_adapter_v1/validation_seed42/m3_correct/` |
| Shuffle validation metrics | `outputs/htr_adapter_v1/validation_seed42/m3_shuffle/` |
| Validation gate | `outputs/htr_adapter_v1/statistical_analysis/validation_gate/validation_gate.md` |
| Machine-readable final result | `outputs/htr_adapter_v1/final_report/final_report.json` |
| Concise final report | `outputs/htr_adapter_v1/final_report/final_report.md` |
| Architecture figure | `outputs/htr_adapter_v1/final_report/figure_a_architecture.png` |
| Seed42 figure | `outputs/htr_adapter_v1/final_report/figure_b_seed42_validation.png` |
| Method/results text | `outputs/htr_adapter_v1/final_report/method_and_results_sections_ru.md` |
| Executed notebooks | `outputs/htr_adapter_v1/notebook/` |

## 22. Научная интерпретация

Полученные данные поддерживают ограниченный вывод:

1. x-aligned HI-CSG-R branch технически работоспособен;
2. правильный локальный граф содержит небольшой sample-specific сигнал,
   поскольку correct graph лучше matched shuffled graph;
3. сигнал не приводит к улучшению сильной CRNN-CTC относительно matched
   image-only fine-tuning;
4. отрицательный результат наблюдается overall и во всех core validation
   domains;
5. topology-specific claim нельзя делать, поскольку M2 не был разрешен после
   pre-M2 STOP;
6. воспроизводимость по трем seeds и test generalization не проверялись по
   заранее заданным этическим и статистическим ограничениям протокола.

Финальная формулировка:

> H4 остается поисковой: локально выровненное структурное слияние HI-CSG-R
> не продемонстрировало превосходства над matched image-only fine-tuning на
> seed-42 validation. Дальнейшие модели и test не запускались в соответствии
> с заранее зафиксированными stopping rules.

