# CRNN-CTC + x-aligned HI-CSG-R: финальный отрицательный отчёт

Статус H4: **exploratory**

## Решение протокола

Заранее зафиксированный validation gate на seed 42 вернул **STOP**. Это финальный результат ветки adapter; test set не использовался.

## Результаты validation

| model | CER | WER | Exact | Macro-CER |
|---|---:|---:|---:|---:|
| M0-FT | 0.079537 | 0.332398 | 0.626833 | 0.084963 |
| M3 | 0.082196 | 0.337477 | 0.620667 | 0.087691 |
| M3-shuffle | 0.083004 | 0.341486 | 0.615167 | 0.088702 |

Relative CER improvement M3 относительно M0-FT: **-3.342%** (ухудшение на **3.342%**).

## Основные домены

| domain | M0-FT CER | M3 CER | Delta CER | M0-FT Exact | M3 Exact |
|---|---:|---:|---:|---:|---:|
| cyrillic_handwriting | 0.101731 | 0.103834 | +0.002104 | 0.563500 | 0.568000 |
| hkr_words | 0.067098 | 0.069938 | +0.002841 | 0.602000 | 0.590000 |
| school_notebooks_clean | 0.073384 | 0.076418 | +0.003033 | 0.715000 | 0.704000 |

## Критерии остановки

| criterion | passed |
|---|---:|
| `relative_improvement_2pct` | **False** |
| `two_domains_not_worse` | **False** |
| `max_domain_degradation` | **True** |
| `correct_better_shuffle` | **True** |
| `gate_variable` | **True** |
| `adapter_gradient` | **True** |

## Контроли и диагностика

- Правильный graph CER лучше shuffled на `0.000808`, однако M3 не превосходит M0-FT.
- Стандартное отклонение gate: `0.029756`.
- Максимальная норма градиента graph adapter при joint training: `4.845424`.
- Topology-off M2 не обучался, поскольку pre-M2 gate не пройден.

![Adapter architecture](figure_a_architecture.png)

![Seed-42 validation](figure_b_seed42_validation.png)

## Намеренно не запускалось

- M2 seed42
- M0-FT seeds 43 and 44
- M3 seeds 43 and 44
- final test/domain/page-disjoint/robustness evaluation
- paired test bootstrap and Holm analysis

## Вывод

H4 остаётся поисковой: локальное слияние HI-CSG-R не продемонстрировало превосходства над matched image-only fine-tuning на seed-42 validation. Согласно заранее заданным stopping rules дальнейшие модели и test не запускались.

Для этого вывода не использовалась ни одна test-derived метрика.
