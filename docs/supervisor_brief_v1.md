# Краткое резюме для научного руководителя

Дата: 2026-07-02

## Что сделано

Исследование начиналось с идеи HI-CSG-R: представить рукописный текст не только как изображение, но и как структурный граф видимых штрихов, соединений и дефектов. В ходе проверки стало ясно, что сильное утверждение "графовая модель устойчиво улучшает распознавание" пока не доказано. Поэтому научная рамка была уточнена: HI-CSG-R используется прежде всего как диагностический слой качества, а основной устойчивый результат связан с контролируемым улучшением HTR-пайплайна.

Сейчас работа выглядит так:

- построен воспроизводимый offline HTR pipeline для русскоязычных рукописных данных;
- исправлена критическая предобработка School Notebooks через line-aware foreground;
- проведено обучение CRNN-HTR по 3 seed;
- проверено добавление 10k natural-line samples;
- выполнены same-size controls: random crops и School-word controls;
- проведены fixed-penalty evaluation, paired confidence intervals, dose-response, leakage/duplicate audits;
- добавлены TrOCR baseline, strong internal baselines, selective prediction, annotation reliability checks;
- завершен строгий page-disjoint HKR+School base-vs-line retraining по 3 seed;
- подготовлены и запущены page-disjoint same-size controls, подготовлен пакет для независимой разметки.

## Главный результат

Добавление 10k natural-line samples улучшает image-only CRNN относительно базовой модели на 3 seed:

| model | mean CER | std CER | mean WER | mean exact |
|---|---:|---:|---:|---:|
| baseline | 0.152431 | 0.009247 | 0.528311 | 0.424232 |
| +10k natural-line | 0.135479 | 0.001521 | 0.488990 | 0.465037 |

В publication v3 fixed-penalty протоколе результат остается близким:

| variant | mean CER | std CER | mean WER | mean exact |
|---|---:|---:|---:|---:|
| `tri10k_base` | 0.1528 | 0.0089 | 0.5293 | 0.4232 |
| `line_context_10k` | 0.1358 | 0.0013 | 0.4898 | 0.4641 |
| `random_crops_10k_control` | 0.1358 | 0.0017 | 0.4894 | 0.4644 |
| `school_words_10k_control` | 0.1366 | 0.0012 | 0.4869 | 0.4661 |

Вывод должен быть аккуратным:

> Natural-line augmentation улучшает baseline, но same-size controls показывают, что уникальное преимущество именно natural-line context пока не доказано. Более строгий вывод: релевантное расширение данных и исправленная предобработка улучшают HTR.

На строгом page-disjoint HKR+School split эффект также подтвержден:

| variant | mean CER | std CER | mean WER | mean exact |
|---|---:|---:|---:|---:|
| `page_base` | 0.1483 | 0.0119 | 0.4764 | 0.3946 |
| `page_line_10k` | 0.1271 | 0.0057 | 0.4227 | 0.4522 |

Mean `page_line_10k - page_base`: ΔCER = -0.0212, ΔWER = -0.0537, Δexact = +0.0576. Улучшение CER есть во всех 3/3 seed, paired bootstrap CI по каждому seed ниже нуля.

Page-disjoint same-size controls уже подготовлены с тем же train size 28 014 и без train/test page overlap; сейчас они обучаются по 3 seed. До завершения этих controls нельзя утверждать, что natural-line context уникально лучше других page-disjoint same-size добавок.

## Роль HI-CSG-R

HI-CSG-R сейчас лучше всего обоснован как структурно-диагностический слой:

- показывает, когда preprocessing сохраняет или разрушает видимые штрихи;
- помогает выявлять structural usability образцов;
- используется в анализе надежности и selective prediction;
- не должен подаваться как доказанный восстановитель траектории пера или порядка письма.

Корректный claim:

> HI-CSG-R полезен для диагностики качества и контроля структурной пригодности рукописных образцов.

Некорректный claim:

> HI-CSG-R восстанавливает истинную траекторию письма и устойчиво улучшает HTR как основной recognizer.

## Что пока не закрыто

Критические ограничения перед сильной публикацией:

- formal independent inter-annotator agreement пока не готов;
- сильный внешний Russian/Cyrillic HTR baseline пока отсутствует;
- page-disjoint same-size controls запущены, но итоговые результаты и paired line-vs-control CI пока не готовы;
- уникальность natural-line context относительно page-disjoint same-size controls пока не доказана;
- graph-fusion остается exploratory;
- writer-disjoint validation невозможен без writer_id metadata.

## Как позиционировать работу

Сильная и честная формулировка:

> Работа исследует структурно контролируемый HTR pipeline для русскоязычного рукописного текста: исправление предобработки, data/context augmentation, HI-CSG-R диагностику качества и selective prediction. Главный подтвержденный результат - улучшение HTR относительно базовой модели при добавлении релевантных данных; главный диагностический вклад - использование структурного слоя для контроля качества и ограничений модели.

Не стоит позиционировать работу как:

> Новая графовая модель, которая доказанно превосходит image-only HTR и SOTA.

## Что показать как доказательства

Полный предварительный отчет: `docs/preliminary_scientific_report_for_supervisor_v1.md`

Ключевые evidence files:

- `outputs/htr_publication_v3/publication_v3_status_report.md`
- `outputs/htr_publication_v3/remaining_addendum_v1/report.md`
- `outputs/htr_publication_v3/external_baseline_availability_v1/report.md`
- `outputs/htr_publication_v3/annotation_reliability_addendum_v1/report.md`
- `outputs/final_result_package_v1/seed_confirmation_summary.md`
- `outputs/final_result_package_v1/thesis_tables/table_5_selective_prediction.md`
- `outputs/final_result_package_v1/thesis_tables/table_6_graph_fusion_exploratory.md`
