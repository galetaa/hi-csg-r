# Финальный экспериментальный протокол v1

## HI-CSG-R: структурно-диагностическое графовое представление для offline HTR русского рукописного текста

## 0. Статус документа

Этот документ фиксирует финальную экспериментальную рамку проекта и запрещает дальнейшее расширение архитектур без отдельного методологического обоснования.

Документ нужен для того, чтобы:

1. зафиксировать, что именно доказывается в работе;
2. отделить основные результаты от exploratory-результатов;
3. предотвратить дальнейший перебор моделей;
4. задать правила использования train/validation/test;
5. определить, какие таблицы и артефакты должны войти в итоговую научную работу.

---

# 1. Рабочая тема

**Структурно-диагностическое графовое представление рукописного текста для контроля качества и повышения надёжности offline HTR**

Расширенное техническое название:

**HI-CSG-R: канонический граф видимых штрихов для структурной диагностики, контроля данных и selective prediction в offline-распознавании русского рукописного текста**

---

# 2. Методологическая позиция

Работа сохраняет графовую рамку, но не утверждает, что прямое graph-fusion распознавание устойчиво превосходит сильный image-only baseline.

Основная позиция:

> HI-CSG-R рассматривается как структурно-диагностическое представление offline-рукописи, пригодное для контроля качества foreground/skeleton/graph extraction, анализа ошибок HTR и risk-aware/selective prediction. Распознавание текста проверяется через сильный image-only baseline и natural-line context augmentation. Graph-fusion рассматривается как exploratory/limited result.

---

# 3. Что работа доказывает

Работа доказывает или проверяет следующие положения:

1. Для русских рукописных изображений можно построить диагностически пригодный pipeline:

   ```text
   image → foreground → skeleton → graph → diagnostics
   ```

2. HI-CSG-R может использоваться как диагностический слой для оценки пригодности структурного представления рукописи.

3. Исправление preprocessing и использование natural-line context augmentation улучшают offline HTR по сравнению с baseline.

4. Confidence-aware selective prediction, дополненная graph/quality diagnostics, позволяет снижать CER на принятой части выборки.

5. Простое прямое graph-fusion распознавание даёт доменно-зависимый эффект и не является главным доказанным механизмом улучшения HTR.

---

# 4. Что работа не доказывает

В работе не заявляется:

1. восстановление реальной траектории пера;
2. восстановление реального порядка написания штрихов;
3. судебно-почерковедческая идентификация автора;
4. медицинская, психологическая или биометрическая диагностика;
5. точное восстановление нажима или темпа письма;
6. pixel-level или stroke-level ground truth для всех графов;
7. универсальное превосходство graph-aware HTR над image-only HTR;
8. SOTA по всем русским рукописным датасетам;
9. перенос выводов на все типы рукописей, исторические документы и page-level документы.

---

# 5. Объект и предмет исследования

## 5.1. Объект

Офлайн-изображения русского рукописного текста и методы их автоматического распознавания.

## 5.2. Предмет

Методы построения, диагностической оценки и использования канонического графа видимой штриховой структуры HI-CSG-R в задачах контроля качества, анализа ошибок и повышения надёжности offline HTR.

---

# 6. Центральный исследовательский вопрос

Можно ли использовать HI-CSG-R как структурно-диагностический слой для контроля качества рукописных данных, анализа ошибок и повышения надёжности offline HTR на русском рукописном тексте?

---

# 7. Подвопросы исследования

## RQ1. Structural diagnostic usability

Можно ли построить диагностически пригодное графовое представление HI-CSG-R для русских рукописных изображений?

## RQ2. Source of HTR errors

Какая часть ошибок HTR связана со структурными дефектами foreground/skeleton/graph extraction, а какая — с визуальной неоднозначностью, модельным декодированием, редкими форматами и языковой неопределённостью?

## RQ3. Natural-line context

Улучшает ли восстановление естественного строкового контекста качество offline HTR по сравнению с baseline?

## RQ4. Selective prediction

Позволяют ли confidence и graph/quality diagnostics выявлять ненадёжные предсказания и снижать CER на принятой части выборки?

## RQ5. Graph-fusion

Даёт ли прямое добавление graph-derived признаков в HTR-модель устойчивый выигрыш по CER, или его эффект ограничен отдельными доменами?

---

# 8. Гипотезы

## H1. Structural diagnostic hypothesis

HI-CSG-R-графы, построенные из offline-изображений русского рукописного текста, являются диагностически пригодными для контроля качества foreground, skeleton и graph extraction.

### Проверка H1

Проверяется через structural gold diagnostic subset.

Основные показатели:

```text
foreground_ok
skeleton_ok
graph_ok
structural_usable
defect categories
htr_error_explained_by_structure
```

### Интерпретация H1

Structural gold diagnostic не является pixel-level topology benchmark. Он проверяет диагностическую пригодность графа, а не полную топологическую точность всех штрихов.

---

## H2. Natural-line context hypothesis

Natural-line context augmentation улучшает качество offline HTR по сравнению с baseline.

### Проверка H2

Сравниваются:

```text
M0: image-only baseline
M1: image-only + natural-line context augmentation
```

Основная проверка проводится по 3 seeds.

Основная метрика:

```text
CER
```

Вторичные метрики:

```text
WER
exact match
domain-wise CER
domain-wise WER
domain-wise exact
```

Статистическая проверка:

```text
mean/std over seeds
paired bootstrap CI
domain-wise delta
```

---

## H3. Selective prediction hypothesis

Комбинация model confidence и graph/quality diagnostics позволяет лучше выявлять ненадёжные предсказания и снижать CER на принятой части выборки.

### Проверка H3

Сравниваются risk/scoring variants:

```text
R0: confidence-only
R1: graph/quality-only
R2: confidence + graph/quality diagnostics
```

Метрики:

```text
ROC-AUC
PR-AUC
CER at coverage
coverage-risk curve
domain-wise coverage-risk curve
```

---

## H4. Graph-fusion exploratory hypothesis

Прямое добавление graph-derived признаков в HTR-модель может давать доменно-зависимый эффект, но в текущей реализации не является доказанным универсальным улучшением над сильным image-only baseline.

### Проверка H4

Сравниваются:

```text
image-only + natural-line
graph-fusion + natural-line
zero-graph ablation
```

Интерпретация:

```text
graph-fusion is exploratory;
zero-graph ablation is a dependency test, not a fair image-only control;
graph-fusion superiority is not claimed unless it improves across domains and seeds.
```

---

# 9. Роли датасетов

## 9.1. Core RU datasets

Основная экспериментальная база:

```text
Cyrillic Handwriting
HKR Words
School Notebooks
```

### Cyrillic Handwriting

Роль:

```text
clean-ish Russian handwritten crop/word/phrase domain
primary RU crop-domain benchmark
```

Используется для:

```text
HTR baseline
natural-line comparison if applicable
graph diagnostics
selective prediction
```

### HKR Words

Роль:

```text
secondary Russian word-domain benchmark
cross-source validation within Russian handwriting
```

Используется для:

```text
HTR baseline
domain-wise evaluation
selective prediction
graph diagnostics
```

### School Notebooks

Роль:

```text
hard realistic notebook domain
main stress domain for preprocessing and natural-line context
```

Используется для:

```text
preprocessing repair validation
natural-line augmentation
hard-domain HTR evaluation
structural diagnostics
selective prediction
failure analysis
```

---

## 9.2. Diagnostic / stress RU datasets

Дополнительная русская diagnostic/stress база:

```text
HWR200
HKR Forms
School hard/review subsets
```

### HWR200

Роль:

```text
real acquisition variability
scan/photo/light-condition stress
```

Используется для:

```text
diagnostic/stress analysis
robustness discussion
optional external test
```

### HKR Forms

Роль:

```text
form/page-like structure
graph extraction stress
segmentation assumption stress
```

Используется для:

```text
graph diagnostics
visual failure analysis
stress examples
```

### School hard/review/invalid subsets

Роль:

```text
real failure cases
quality-control evaluation
diagnostic examples
```

Используется для:

```text
foreground/skeleton/graph diagnostics
failure taxonomy
selective prediction analysis
```

---

## 9.3. IAM

IAM не входит в основную русскую доказательную базу.

Статус:

```text
optional English sanity-check / background only
```

Причина исключения из core diagnostic/stress evaluation:

```text
IAM is English/Latin-script, while the main study is focused on Russian/Cyrillic handwriting. Including IAM in the main stress comparison would mix script/language effects with structural/image-quality effects.
```

IAM может быть упомянут только как:

```text
external reference
background dataset
future work direction
optional sanity-check
```

---

# 10. Модели и экспериментальные статусы

## 10.1. Primary models

### M0. Image-only baseline

Назначение:

```text
main recognition baseline
```

Архитектура:

```text
image → CNN encoder → sequence model → CTC decoder
```

Используется в primary comparison.

---

### M1. Image-only + natural-line context augmentation

Назначение:

```text
main HTR improvement model
```

Это основной положительный recognition-result.

Сравнивается с M0 по 3 seeds.

---

## 10.2. Diagnostic components

### D0. HI-CSG-R diagnostic pipeline

Назначение:

```text
structural diagnostic layer
foreground/skeleton/graph quality control
failure analysis
risk features
```

Используется в H1 и H3.

---

### D1. Graph/quality feature extraction

Назначение:

```text
features for diagnostics and selective prediction
```

Не трактуется как самостоятельное доказательство recognition superiority.

---

## 10.3. Exploratory models

### E0. Graph-fusion / gated / graph-aware HTR variants

Статус:

```text
exploratory / limited result
```

Используется для ответа на вопрос:

```text
does direct graph conditioning add measurable recognition value after strong preprocessing and natural-line augmentation?
```

Не используется как основной claim, если нет стабильного улучшения across domains and seeds.

---

# 11. Запрещённые дальнейшие расширения

После принятия этого протокола запрещены:

```text
new gated variants
GNN
Graph Transformer
cross-attention fusion
new graph-channel architecture
additional model search
test-driven architecture tuning
new datasets before seed confirmation
```

Исключение возможно только через отдельный protocol amendment с причиной, ожидаемым вкладом и stopping rule.

---

# 12. Train / validation / test protocol

## 12.1. Общие правила

1. Test set не используется для выбора архитектуры.
2. Test set не используется для выбора hyperparameters.
3. Test set не используется для выбора blank penalty.
4. Все model-selection решения принимаются по validation.
5. Test используется для финального reporting.
6. Все exploratory experiments явно маркируются как exploratory.
7. Negative results не скрываются.

---

## 12.2. Penalty selection

Blank/logit penalty выбирается только на validation.

Для финальной таблицы фиксируется:

```text
model
seed
validation penalty
validation CER
test CER
```

---

## 12.3. Seed protocol

Для primary HTR comparison используются 3 seeds:

```text
seed_1
seed_2
seed_3
```

Минимальное обязательное сравнение:

```text
M0 image-only baseline
M1 image-only + natural-line context augmentation
```

Для каждого seed сохраняются:

```text
checkpoint
config
train log
validation summary
test summary
predictions
```

---

# 13. Primary comparison

## 13.1. Основное сравнение

```text
M0 image-only baseline
vs
M1 image-only + natural-line context augmentation
```

## 13.2. Primary metric

```text
CER
```

## 13.3. Secondary metrics

```text
WER
exact match
domain-wise CER
domain-wise WER
domain-wise exact
```

## 13.4. Reporting

Основная таблица:

```text
model | seed | overall CER | Cyrillic CER | HKR CER | School CER | overall WER | School WER | exact
```

Итоговая таблица:

```text
model | mean CER | std CER | mean School CER | std School CER | mean WER | mean exact
```

---

# 14. Structural gold diagnostic protocol

## 14.1. Назначение

Structural gold subset используется для проверки диагностической пригодности foreground/skeleton/graph pipeline.

Он не используется как утверждение полной topology correctness.

---

## 14.2. Минимальные поля разметки

```text
sample_id
dataset
stratum
token_type
foreground_ok
skeleton_ok
graph_ok
structural_usable
line_residual
missed_ink
neighbor_noise
false_ink
false_branches
broken_strokes
segmentation_issue
htr_error_explained_by_structure
comment
```

---

## 14.3. Основные показатели

```text
completed_n
structural_usable_rate
foreground_ok_rate
skeleton_ok_rate
graph_ok_rate
defect distribution
htr_error_explained_by_structure distribution
rates by dataset
rates by stratum
rates by token_type
```

---

## 14.4. Acceptance interpretation

Если показатели высокие:

```text
The structural extraction pipeline is diagnostically usable on the sampled subset.
```

Не писать:

```text
The graph is topologically correct.
The graph fully reconstructs handwriting.
The graph recovers pen trajectory.
```

---

# 15. Selective prediction protocol

## 15.1. Цель

Оценить, можно ли использовать confidence и graph/quality diagnostics для отбора более надёжных HTR-предсказаний.

---

## 15.2. Сравниваемые варианты

```text
R0 confidence-only
R1 graph/quality-only
R2 confidence + graph/quality diagnostics
```

---

## 15.3. Метрики

```text
ROC-AUC
PR-AUC
CER at coverage
WER at coverage
exact at coverage
coverage-risk curve
domain-wise coverage-risk curve
```

Coverage points:

```text
0.90
0.80
0.70
0.60
0.50
0.40
```

---

## 15.4. Основная интерпретация

Допустимые выводы:

```text
confidence+graph diagnostics improves risk-aware filtering;
graph/quality features help characterize hard samples;
selective prediction reduces CER at lower coverage.
```

Недопустимые выводы:

```text
graph features alone solve recognition;
selective prediction improves full-coverage CER;
rejected samples are necessarily structurally invalid.
```

---

# 16. Graph-fusion exploratory protocol

## 16.1. Цель

Закрыть вопрос о том, даёт ли прямое graph-conditioning measurable recognition value после strong image-only baseline и natural-line context augmentation.

---

## 16.2. Сравнения

```text
image-only + natural-line
graph-fusion + natural-line
zero-graph ablation
```

---

## 16.3. Метрики

```text
overall CER
overall WER
overall exact
Cyrillic CER
HKR CER
School CER
School WER
School exact
```

---

## 16.4. Интерпретация zero-graph ablation

Zero-graph ablation проверяет, использует ли обученная graph-fusion модель graph branch.

Корректная интерпретация:

```text
If zero-graph performance is substantially worse, the model depends on graph input.
```

Некорректная интерпретация:

```text
Zero-graph ablation is equivalent to image-only baseline.
```

Zero-graph ablation не является fair image-only control, потому что модель обучалась с graph input.

---

## 16.5. Условие признания graph-fusion successful

Graph-fusion может считаться successful recognition result только если:

```text
1. improves overall CER;
2. improves or does not worsen major domains;
3. does not strongly degrade Cyrillic;
4. improvement survives paired bootstrap;
5. preferably survives seed confirmation.
```

Если эти условия не выполнены, graph-fusion остаётся exploratory/limited result.

---

# 17. Robustness protocol

Robustness не является главным primary result в текущей версии, но может использоваться как secondary/exploratory block.

## 17.1. Artificial distortions

Допустимые controlled distortions:

```text
blur
noise
low_contrast
thin_strokes
thick_strokes
jpeg_artifacts
```

Levels:

```text
mild
medium
strong
```

---

## 17.2. Главное правило для graph-aware robustness

Нельзя использовать graph features, извлечённые из clean images, при оценке distorted images.

Правильно:

```text
clean image
→ distortion
→ distorted image
→ graph extraction from distorted image
→ evaluation
```

Неправильно:

```text
distorted image + clean graph features
```

Это считается утечкой чистого структурного сигнала.

---

## 17.3. Robustness reporting

Таблица:

```text
model | clean CER | distorted CER | absolute ΔCER | relative degradation
```

Интерпретация:

```text
relative degradation alone is not enough;
absolute CER must also be reported.
```

---

# 18. Leakage and validity checks

Перед финальным reporting необходимо проверить:

```text
text_len is not used as input feature;
transcription-derived features are not used;
dataset ID is not directly used as predictive feature;
test set is not used for model selection;
duplicates across splits are checked;
writer leakage is controlled where writer_id exists;
graph features for distorted images are recomputed from distorted images;
all final paths are reproducible;
all checkpoints/configs are saved.
```

---

# 19. Results inventory requirement

Перед написанием итоговой главы создаётся таблица:

```text
outputs/final_result_package_v1/results_inventory.csv
```

Поля:

```text
experiment_id
result_group
model
dataset
seed
train_manifest
val_manifest
test_manifest
checkpoint_path
config_path
summary_path
predictions_path
included_in_thesis
status
notes
```

Статусы:

```text
primary
secondary
exploratory
negative
diagnostic
excluded
```

---

# 20. Required final tables

В итоговую работу должны войти следующие таблицы.

## Table 1. Dataset roles and statistics

```text
dataset | language | script | role | samples | level | used_in_training | used_in_testing | used_in_diagnostics | primary_claim
```

## Table 2. Preprocessing / quality-control summary

```text
dataset | original issue | correction | kept | rejected/review | notes
```

## Table 3. Structural gold diagnostic

```text
dataset/stratum | n | foreground_ok | skeleton_ok | graph_ok | structural_usable | main defects
```

## Table 4. Primary HTR 3-seed comparison

```text
model | mean CER | std CER | mean WER | mean exact | mean School CER | std School CER
```

## Table 5. Domain-wise HTR results

```text
model | seed | Cyrillic CER | HKR CER | School CER | overall CER
```

## Table 6. Bootstrap significance

```text
comparison | domain | ΔCER | CI low | CI high | interpretation
```

## Table 7. Graph-fusion exploratory result

```text
model | overall CER | Cyrillic CER | HKR CER | School CER | interpretation
```

## Table 8. Selective prediction

```text
risk model | ROC-AUC | PR-AUC | CER@90 | CER@80 | CER@70 | CER@50 | notes
```

## Table 9. Failure taxonomy

```text
failure type | count | dataset | example | likely cause | mitigation
```

---

# 21. Required final artifacts

Финальный пакет результатов:

```text
outputs/final_result_package_v1/
  results_inventory.csv
  seed_confirmation_table.csv
  seed_confirmation_summary.md
  structural_gold_final.csv
  structural_gold_report.md
  selective_prediction_table.csv
  selective_prediction_report.md
  graph_fusion_exploratory_table.csv
  graph_fusion_interpretation.md
  result_card.md
```

Документы:

```text
docs/final_experimental_protocol_v1.md
docs/final_dataset_roles_v1.md
docs/final_research_frame_v1.md
```

---

# 22. Stopping rules

После принятия этого протокола работа не расширяется.

Разрешённые следующие действия:

```text
1. inventory existing results;
2. run missing 3-seed confirmation;
3. aggregate final tables;
4. finalize structural gold report;
5. finalize selective prediction analysis;
6. write result card;
7. start thesis writing.
```

Запрещённые действия:

```text
1. train new architecture;
2. tune graph-fusion further;
3. add GNN/Transformer/cross-attention;
4. introduce new primary dataset;
5. choose model by test result;
6. hide negative graph-fusion results;
7. claim graph-fusion superiority without evidence.
```

---

# 23. Final expected claim

Итоговый допустимый claim:

> The work introduces HI-CSG-R as a structural diagnostic representation for Russian offline handwriting recognition. The proposed pipeline supports foreground/skeleton/graph quality control and error-risk analysis. Recognition quality is improved primarily through corrected preprocessing and natural-line context augmentation, while graph-derived diagnostics are useful for structural validation and selective prediction. Direct graph-fusion provides domain-dependent effects and is reported as an exploratory/limited result rather than a universal recognition improvement.

Русская версия:

> В работе предложено HI-CSG-R как структурно-диагностическое представление для offline-распознавания русского рукописного текста. Разработанный pipeline позволяет контролировать качество foreground/skeleton/graph extraction и анализировать риск ошибок HTR. Основной прирост качества распознавания достигается за счёт исправленного preprocessing и natural-line context augmentation, тогда как графовые признаки используются преимущественно для диагностики, анализа ошибок и selective prediction. Прямое graph-fusion распознавание показывает доменно-зависимый эффект и рассматривается как exploratory/limited result, а не как универсальное улучшение над сильным image-only baseline.

---

# 24. Current next step

Следующий обязательный шаг:

```text
build results_inventory.csv
```

После inventory:

```text
run missing 3-seed confirmation for:
  M0 image-only baseline
  M1 image-only + natural-line context augmentation
```

Только после этого можно переходить к написанию итоговой главы результатов.
