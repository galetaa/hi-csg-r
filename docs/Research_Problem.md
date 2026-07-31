# Этап 0. Постановка исследования и фиксирование границ проекта

**Проект:** HI-CSG-R — Handwriting-Informed Canonical Stroke Graphs for Robust Offline Handwritten Text Recognition
---

## 1. Рабочее название темы

### Основной академический вариант

**Метод построения канонического графа видимой штриховой структуры в задаче офлайн-распознавания русско-английского рукописного текста**

### Комментарий по названию

В названии сознательно не используется формулировка «восстановление траектории пера», потому что из офлайн-изображения рукописи нельзя надёжно восстановить реальный порядок движений. Работа посвящена не восстановлению реального письменно-двигательного процесса, а построению канонического графа видимых штрихов.

Слово «робастный» не выносится в русское академическое название как главный термин, чтобы не перегружать тему. Робастность остаётся ключевым экспериментальным аспектом, но не подменяет основной объект исследования — графовую реконструкцию видимой структуры рукописи.

---

## 2. Краткая концепция исследования

Современная HTR/OCR-модель обычно решает задачу:

```text
image → text
```

В данной работе предлагается исследовать более интерпретируемый путь:

```text
image
→ visible stroke structure
→ canonical stroke graph
→ structural graph features
→ text recognition + graph + confidence + structural explanation
```

Ключевой объект проекта — **HI-CSG-R**, то есть канонический граф видимой штриховой структуры рукописного изображения. Такой граф должен представлять видимые элементы рукописи: штриховые сегменты, узлы, соединения, разрывы, петли, пересечения, кривизну, толщину штриха как proxy, зоны строки и признаки структурной неопределённости.

Принципиально важно: HI-CSG-R не является записью настоящей траектории пера. Это не online handwriting representation. Это структурная реконструкция того, что видно на скане или фотографии.

---

## 3. Центральный исследовательский вопрос

**Можно ли использовать канонический граф видимой штриховой структуры как воспроизводимое промежуточное представление офлайн-рукописи для повышения интерпретируемости и устойчивости HTR-модели?**

Этот вопрос является главным фильтром проекта. Все обязательные разделы, эксперименты и метрики должны работать на его проверку.

---

## 4. Исследовательская проблема

Современные offline HTR-модели преимущественно решают задачу прямого преобразования изображения рукописного текста в строку символов. Такой подход позволяет оптимизировать CER/WER, но обычно не формирует явного структурного представления видимых штрихов рукописи. Это создаёт несколько проблем.

Во-первых, ошибки распознавания трудно объяснять структурно: модель может неверно распознать символ, но не показывает, связана ли ошибка с потерей петли, ложным соединением букв, разрывом штриха, шумовой веткой или деформацией зоны строки.

Во-вторых, традиционные OCR/HTR-метрики, такие как CER и WER, не оценивают качество сохранения графовой структуры рукописи. Две модели могут иметь близкий CER, но существенно различаться по тому, насколько они сохраняют петли, соединения, пересечения и другие элементы, важные для интерпретируемого анализа рукописного изображения.

В-третьих, в офлайн-режиме отсутствует надёжная информация о реальном порядке движения пера, поэтому необходима осторожная методологическая постановка: допустимо строить только каноническое представление видимой структуры, а не заявлять восстановление реального письменно-двигательного процесса.

Следовательно, проблема исследования состоит в отсутствии воспроизводимого и количественно оцениваемого способа построения канонического графового представления видимых штрихов offline-рукописи, пригодного для анализа структурных ошибок и использования в HTR-моделях.

---

## 5. Актуальность исследования

Актуальность работы определяется тремя обстоятельствами.

Первое — практическая потребность в более интерпретируемом распознавании рукописного текста. Для архивных, образовательных, исследовательских и прикладных OCR/HTR-систем важно не только получить текстовую строку, но и понимать, почему модель ошиблась и какие структурные свойства изображения повлияли на распознавание.

Второе — ограниченность прямых image-to-text моделей с точки зрения структурного анализа. Даже сильная HTR-модель может не давать информации о том, сохранились ли в изображении петли, соединения, пересечения, нижние и верхние элементы букв, а также насколько надёжен сам входной материал.

Третье — наличие доступных русско-английских датасетов, позволяющих проверить метод не только на одном корпусе, но и на разных типах данных: английские строки, русские слова и фразы, русские страницы/строки в разных условиях съёмки, исторические документы, школьные тетради и изолированные кириллические символы. Это позволяет построить не только демонстрационный pipeline, но и воспроизводимый экспериментальный протокол.

---

## 6. Объект и предмет исследования

### Объект исследования

**Офлайн-изображения рукописного текста на русском и английском языках и методы их автоматического распознавания.**

### Предмет исследования

**Методы построения, оценки и использования канонического графа видимой штриховой структуры рукописного текста в offline HTR.**

### Разграничение объекта и предмета

Объектом являются рукописные изображения и системы их распознавания в целом. Предметом является более узкий аспект: графовое представление видимых штрихов, его построение, метрики качества и применение в распознавании.

---

## 7. Цель работы

**Разработать и экспериментально оценить метод построения канонического графа видимых штрихов offline-рукописи и проверить его применимость для интерпретируемого анализа и повышения устойчивости HTR-модели на русско-английских данных.**

Цель не предполагает создание «лучшей OCR-модели». Работа оценивает применимость графового промежуточного представления и условия, при которых оно помогает, не помогает или ухудшает результат.

---

## 8. Задачи исследования

1. Проанализировать существующие подходы к offline HTR, структурной векторизации рукописей, скелетизации, графовому представлению штрихов и почерковедчески мотивированным признакам изображения.

2. Сформулировать спецификацию HI-CSG-R как канонического графа видимой штриховой структуры offline-рукописи.

3. Подготовить единый data pipeline для русско-английских рукописных датасетов с сохранением метаданных, транскрипций, уровней разметки и режимов предобработки.

4. Разработать classical pipeline построения графа: изображение → бинаризация → скелетизация → pixel graph → узлы/рёбра → pruning → петли/соединения → width proxy → curve simplification → graph.json/SVG.

5. Создать небольшой gold subset с ручной графовой разметкой русских и английских образцов, включая проверку согласованности между двумя аннотаторами.

6. Разработать набор structural metrics для оценки качества графовой реконструкции: Skeleton F1, Endpoint F1, Junction F1, Loop Preservation Score, Critical Topology Error и дополнительные векторные метрики.

7. Реализовать image-only HTR baseline и получить базовые значения CER/WER.

8. Извлечь x-aligned graph feature sequence и global graph features, пригодные для использования в CTC-based HTR pipeline.

9. Реализовать graph-only sanity baseline для проверки информационной достаточности графовых признаков.

10. Реализовать image+graph fusion baseline и сравнить его с image-only моделью на clean и distorted test sets.

11. Провести robustness experiments на искусственных и естественных искажениях: шум, blur, низкий контраст, erosion/broken strokes, dilation/thick strokes, разные условия съёмки.

12. Провести ablation study, чтобы оценить вклад отдельных компонентов: loop/junction features, width proxy, graph quality gate, feature-preserving preprocessing и pruning.

13. Проанализировать failure cases и ограничения метода, включая случаи, когда графовое представление не помогает или ухудшает результат.

---

## 9. Гипотезы исследования

### H1. Основная гипотеза устойчивости

**Добавление признаков канонического графа видимых штрихов к image-only HTR-модели снижает относительную деградацию CER при визуальных и структурных искажениях изображения по сравнению с моделью, использующей только растровое изображение.**

#### Нулевая гипотеза H0-1

Добавление графовых признаков не снижает относительную деградацию CER при искажениях по сравнению с image-only baseline.

#### Критерии проверки H1

H1 считается частично подтверждённой, если выполняются условия:

```text
1. image+graph модель не хуже или несущественно хуже image-only модели на clean test;
2. image+graph модель показывает меньший relative CER degradation на distorted test;
3. эффект наблюдается более чем на одном типе искажений;
4. результаты проверяются через confidence intervals или paired bootstrap/permutation test;
5. случаи ухудшения объясняются через graph quality/failure analysis.
```

Рабочий критерий практической значимости:

```text
снижение среднего relative CER degradation минимум на 10% относительно image-only baseline
```

Этот порог используется как практический ориентир, а не как абсолютный математический критерий.

---

### H2. Структурная гипотеза

**Classical pipeline image → skeleton → graph → canonical graph способен сохранять ключевые видимые структурные элементы рукописи — endpoints, junctions, loops, crossings и connected components — на уровне, достаточном для количественной оценки и анализа ошибок.**

#### Нулевая гипотеза H0-2

Построенный граф не сохраняет видимую структуру рукописи достаточно надёжно и не позволяет проводить устойчивую структурную оценку.

#### Проверка H2

H2 проверяется на gold subset через:

```text
Skeleton F1
Endpoint Precision/Recall/F1
Junction Precision/Recall/F1
Loop Preservation Score
Critical Topology Error
Chamfer distance
Raster reconstruction IoU
```

---

### H3. Гипотеза интерпретируемости

**Показатели graph quality, graph confidence и informativeness позволяют выявлять часть случаев, в которых HTR-модель с высокой вероятностью ошибается.**

#### Нулевая гипотеза H0-3

Graph quality, graph confidence и informativeness не связаны с ошибками HTR и не позволяют отличать проблемные образцы от неп проблемных лучше случайного уровня.

#### Проверка H3

```text
1. correlation(graph_quality_score, CER)
2. correlation(graph_confidence, CER)
3. correlation(informativeness_score, CER)
4. high-error sample detection: label = CER > threshold
5. metrics: ROC-AUC, PR-AUC, F1 for high-error detection
6. qualitative failure analysis
```

---

## 10. Научная новизна

Научная новизна работы состоит не в создании универсально лучшей модели распознавания, а в разработке и проверке структурного промежуточного представления offline-рукописи.

Планируемые элементы новизны:

```text
1. Формальная спецификация HI-CSG-R — канонического графа видимых штрихов offline-рукописи.
2. Воспроизводимый pipeline построения HI-CSG-R из изображения без претензии на восстановление реальной траектории пера.
3. Набор structural metrics для оценки качества графовой реконструкции рукописного изображения.
4. RU/EN gold subset с ручной графовой разметкой и проверкой согласованности двух аннотаторов.
5. Экспериментальная проверка влияния graph features на устойчивость HTR-модели при искажениях.
6. Анализ failure cases, показывающий условия применимости и ограничения graph-based intermediate representation.
```

---

## 11. Практическая значимость

Практическая значимость работы заключается в создании инструментов и протокола, которые могут использоваться для разработки и анализа HTR-систем.

Потенциальные пользователи:

```text
1. исследователи OCR/HTR;
2. разработчики систем распознавания рукописей;
3. исследователи структурной векторизации рукописного текста;
4. специалисты по подготовке и анализу рукописных датасетов;
5. архивные и образовательные проекты, работающие с рукописными материалами.
```

Практические результаты:

```text
1. graph.json как промежуточное структурное представление;
2. SVG/PNG overlay для визуальной отладки;
3. structural metrics для оценки качества графа;
4. annotation guideline для ручной графовой разметки;
5. robustness evaluation protocol;
6. graph features для fusion-моделей;
7. graph confidence и informativeness score для выявления проблемных образцов.
```

---

## 12. Терминологические правила

### Разрешённые формулировки

```text
видимая штриховая структура;
канонический граф видимых штрихов;
каноническая графовая реконструкция видимой структуры;
почерковедчески мотивированные структурные признаки изображения;
width proxy;
pressure-width proxy;
smoothness proxy;
coordination/tempo proxy;
graph confidence;
informativeness score;
uncertain region;
start_candidate;
end_candidate.
```

### Нежелательные или запрещённые формулировки

```text
реальное восстановление траектории пера;
истинный порядок письма;
точный нажим;
настоящий темп письма;
диагностика состояния автора;
судебная идентификация автора;
полная почерковедческая экспертиза;
доказательство индивидуального письменно-двигательного навыка.
```

### Корректные замены

```text
“нажим” → “width proxy” или “pressure-width proxy”
“темп” → “smoothness/tempo proxy”
“начало штриха” → “start_candidate”
“конец штриха” → “end_candidate”
“почерковедческие признаки” → “почерковедчески мотивированные структурные признаки изображения”
“восстановление штрихов” → “каноническая реконструкция видимой штриховой структуры”
```

---

## 13. Датасетная стратегия

### 13.1. Общий принцип

В проекте используются несколько датасетов не для механического увеличения объёма, а для разных экспериментальных ролей. Каждый датасет должен отвечать на отдельный исследовательский вопрос.

```text
IAM → английский benchmark и line-level baseline.
Cyrillic Handwriting Dataset → основной русский word/phrase-level старт.
HWR200 → реальные условия съёмки и robustness.
HKR → крупный line-level русский/кириллический корпус.
School Notebooks → page-level real-world extension.
Digital Peter → historical domain shift.
Russian Handwriting OCR → дополнительный image-to-text корпус.
Cyrillic-MNIST → toy/sanity dataset для изолированных символов.
```

---

### 13.2. Core datasets

#### IAM

Роль:

```text
1. основной английский baseline;
2. line-level HTR;
3. проверка pipeline на классическом корпусе;
4. источник английской части gold subset;
5. clean и distorted experiments.
```

Использование в Core MVP:

```text
обязательно
```

---

#### Cyrillic Handwriting Dataset

Роль:

```text
1. основной русский word/phrase-level dataset;
2. быстрый старт для RU graph extraction;
3. источник русской части gold subset;
4. baseline для CRNN/CTC на коротких фразах;
5. controlled graph evaluation без page detection.
```

Причина включения:

```text
готовые кропы слов/коротких фраз резко снижают технический риск ранней реализации.
```

Использование в Core MVP:

```text
обязательно
```

---

#### HWR200

Роль:

```text
1. проверка на русском рукописном тексте в реальных условиях;
2. scan / good light / poor light robustness;
3. источник сложных образцов для gold subset;
4. проверка graph confidence и informativeness;
5. external robustness test.
```

Использование в Core MVP:

```text
обязательно, но можно начинать с subset
```

---

#### HKR

Роль:

```text
1. крупный line-level RU/Cyrillic dataset;
2. дополнительное обучение русской HTR-модели;
3. writer-level experiments, если metadata позволяет;
4. проверка работы на длинных строках;
5. дополнительная оценка graph extraction на формах.
```

Использование в Core MVP:

```text
рекомендуется как Core-B после IAM + Cyrillic + HWR200 subset
```

Комментарий:

HKR не должен вытеснять Cyrillic Handwriting Dataset на раннем этапе, потому что формы и клетчатые страницы могут усложнить preprocessing. Но после стабилизации classical graph extractor HKR важен как крупный line-level русский корпус.

---

### 13.3. Optional datasets

#### School Notebooks

Роль:

```text
1. future page-level extension;
2. проверка на школьных тетрадях;
3. работа с полигонами слов/строк;
4. detection + HTR setting;
5. сложный real-world layout.
```

Статус:

```text
optional после завершения Core MVP
```

---

#### Digital Peter

Роль:

```text
1. historical domain shift;
2. проверка на старой орфографии;
3. сложные формы и нестандартные рукописные элементы;
4. future stress-test.
```

Статус:

```text
future work / optional stress-test
```

---

#### Russian Handwriting OCR

Роль:

```text
1. дополнительный image-to-text корпус;
2. возможный VLM/OCR fine-tuning source;
3. дополнительная проверка на фото/сканах.
```

Статус:

```text
optional, не использовать для основных выводов до отдельного data audit
```

---

#### Cyrillic-MNIST

Роль:

```text
1. toy/sanity checks;
2. тестирование graph features на изолированных символах;
3. простая классификация букв;
4. debugging skeleton/graph extractor.
```

Статус:

```text
optional, не HTR benchmark
```

---

## 14. Политика публикации кода и данных

Код проекта должен быть публикуемым. Для этого репозиторий не должен содержать raw images сторонних датасетов и не должен распространять переработанные изображения, если это не предусмотрено условиями конкретного набора.

Публикуется:

```text
1. исходный код;
2. конфиги;
3. scripts/download или instructions/download;
4. converters from raw datasets to unified metadata;
5. JSON schema HI-CSG-R;
6. annotation guideline;
7. scripts for metrics;
8. trained model configs;
9. агрегированные метрики;
10. synthetic toy examples, если они созданы нами;
11. small manually created annotations only if это допустимо для выбранных исходных данных.
```

Не публикуется по умолчанию:

```text
1. raw images датасетов;
2. processed image copies;
3. redistributed dataset archives;
4. large derived image datasets.
```

Для воспроизводимости публикуется pipeline, который пользователь запускает локально после получения датасетов из официальных источников.

---

## 15. Gold subset

### 15.1. Назначение

Gold subset нужен для оценки качества графовой реконструкции, а не для обучения большой модели.

Он решает четыре задачи:

```text
1. даёт ground truth для structural metrics;
2. позволяет сравнивать graph extraction variants;
3. позволяет валидировать Critical Topology Error;
4. даёт визуально проверяемые примеры для отчёта и защиты.
```

---

### 15.2. Объём gold subset

Базовый вариант:

```text
Gold subset v1 = 400 samples

80 IAM EN word/short-line samples
100 Cyrillic Handwriting RU word/phrase samples
80 HWR200 RU crop/short-line samples
60 HKR RU line/word crops
40 numeric/mixed samples
40 hard/ambiguous samples
```

Минимально допустимый вариант:

```text
Gold subset minimal = 250 samples

60 IAM
70 Cyrillic Handwriting
50 HWR200
30 HKR
20 numeric/mixed
20 hard/ambiguous
```

Так как есть второй аннотатор, предпочтителен вариант на 400 samples.

---

### 15.3. Стратегия отбора

Gold subset должен быть сбалансирован не только по датасетам, но и по структурным свойствам.

Критерии отбора:

```text
1. script: latin / cyrillic / numeric-mixed;
2. data source: IAM / Cyrillic / HWR200 / HKR;
3. visual quality: clean / moderate noise / low contrast / photo artifacts;
4. structure complexity: simple strokes / loops / junctions / crossings / cursive connections / broken strokes;
5. length: short word / medium word / long word / short line;
6. special forms: ovals, ascenders, descenders, repeated stems, digits, punctuation, separators;
7. ambiguity: uncertain connections, touching letters, broken components, possible false loops.
```

---

### 15.4. Что размечается

```text
1. corrected skeleton;
2. endpoints;
3. junctions;
4. crossing_candidates;
5. loop_closures;
6. edge paths / stroke segments;
7. connected components;
8. continuous / broken / uncertain connections;
9. uncertain regions;
10. optional letter/digit regions for selected samples.
```

---

### 15.5. Два аннотатора

Так как есть второй человек, разметка должна включать inter-annotator agreement.

Минимальная схема:

```text
1. 100% samples размечает annotator A.
2. 25% samples независимо размечает annotator B.
3. Спорные cases обсуждаются и формируется adjudicated gold.
4. Для части данных сохраняются both raw annotations + adjudicated version.
```

Метрики согласованности:

```text
Endpoint F1 within radius 3–5 px
Junction F1 within radius 3–5 px
Loop agreement F1
Edge overlap score
Uncertain region agreement
Component agreement
```

---

## 16. Метод HI-CSG-R

### 16.1. Три уровня графа

#### Level 1. Raw Skeleton Graph

Прямой граф из пикселей скелета.

```text
nodes = raw endpoints / raw junctions / crossing candidates
edges = pixel paths between nodes
```

Назначение:

```text
технический промежуточный слой, не финальный научный объект.
```

---

#### Level 2. Canonical Stroke Graph

Очищенный граф видимой штриховой структуры.

```text
stable nodes
pruned edges
loop candidates
crossing candidates
smoothed polylines
width profiles
edge confidence
node confidence
uncertain regions
```

Назначение:

```text
основной объект исследования.
```

---

#### Level 3. Handwriting-Informed Feature Graph

Canonical graph + агрегированные признаки.

```text
slant
baseline
connectivity
loop count
junction density
endpoint density
width proxy
smoothness proxy
informativeness
graph quality
```

Назначение:

```text
HTR fusion, interpretability, error analysis.
```

---

### 16.2. Classical graph extraction pipeline

```text
feature-preserving image
→ binarization
→ skeletonization
→ pixel graph
→ node detection
→ edge tracing
→ pruning
→ loop detection
→ width proxy estimation
→ curve simplification
→ canonical graph
→ graph.json + SVG/overlay
```

Core algorithms:

```text
binarization: Otsu baseline + Sauvola default candidate
skeletonization: skeletonize / thinning baseline
pixel graph: 8-neighborhood
node detection: degree-based with post-processing
edge tracing: path tracing between endpoints/junctions
pruning: fixed + width-aware thresholds
width proxy: distance transform
curve representation: raw polyline + simplified polyline
```

---

## 17. Dual preprocessing

### OCR-normalized image

Используется для image-only HTR baseline.

```text
grayscale
denoise
contrast normalization
deskew if needed
height normalization
padding
optional binarization
```

### Feature-preserving image

Используется для graph branch.

```text
grayscale
light denoise
background estimation
soft binarization
no aggressive deskew
preserve stroke width
preserve local intensity
preserve baseline geometry
```

Принцип: OCR preprocessing может улучшать CER, но не должен использоваться как единственный вход для графовой ветки, потому что агрессивная нормализация может уничтожить признаки наклона, толщины, локальных соединений и линии письма.

---

## 18. Метрики

### 18.1. Primary HTR metrics

```text
CER
WER
clean CER
clean WER
distorted CER
distorted WER
relative CER degradation
```

Формула:

```text
relative_CER_degradation = (CER_distorted - CER_clean) / CER_clean
```

Главное сравнение:

```text
image-only vs image+graph
```

---

### 18.2. Primary graph metrics

```text
Skeleton F1
Endpoint Precision/Recall/F1
Junction Precision/Recall/F1
Loop Preservation Score
Critical Topology Error
```

---

### 18.3. Secondary metrics

```text
Chamfer distance
Raster reconstruction IoU
Fréchet distance for curves
compression ratio
component count error
graph edit distance on small subset
graph confidence calibration
inference time
model size
```

---

### 18.4. Critical Topology Error

CTE — авторская метрика для критических структурных ошибок.

```text
CTE = weighted_critical_errors / number_of_relevant_structures
```

Типы ошибок:

```text
lost_loop
false_letter_connection
broken_cursive_connection
lost_descender
lost_ascender
false_crossing
extra_noise_branch
merged_words
split_character_component
```

Начальные веса:

```text
lost_loop = 3
false_letter_connection = 4
broken_cursive_connection = 3
lost_descender = 3
lost_ascender = 2
false_crossing = 2
extra_noise_branch = 1
merged_words = 4
split_character_component = 3
```

Важное ограничение:

```text
Веса CTE экспертно заданы для MVP и должны проверяться sensitivity analysis.
```

---

## 19. Модели и сравнения

### 19.1. Image-only baseline

Core model:

```text
CRNN + CTC
```

Архитектура:

```text
image
→ CNN encoder
→ sequence features along x-axis
→ BiLSTM or small Transformer encoder
→ CTC classifier
```

Причина выбора:

```text
1. воспроизводимая baseline-модель;
2. проще TrOCR;
3. быстрее обучается;
4. легко совместима с x-aligned graph features;
5. позволяет контролировать вклад graph branch.
```

TrOCR / Transformer OCR:

```text
optional stronger baseline, not required for Core proof
```

---

### 19.2. Graph-only sanity baseline

Назначение:

```text
проверить, сохраняет ли graph feature sequence достаточно информации для частичного распознавания.
```

Модель:

```text
x-aligned graph feature sequence
→ BiLSTM / small Transformer encoder
→ CTC classifier
```

Интерпретация:

```text
graph-only не обязан быть лучше image-only;
его роль — sanity check информационной достаточности графа.
```

---

### 19.3. Image+graph fusion

Core fusion variants:

```text
F0 image-only
F1 graph-only sanity
F2 concat fusion
F3 gated fusion
```

Concat fusion:

```text
fused[t] = concat(image_features[t], graph_features[t])
```

Gated fusion:

```text
gate[t] = sigmoid(MLP([image_features[t], graph_features[t], graph_quality]))
fused[t] = gate[t] * graph_projection[t] + (1 - gate[t]) * image_features[t]
```

Причина gated fusion:

```text
если граф построен плохо, модель должна иметь механизм снижения доверия к graph branch.
```

---

## 20. Robustness experiments

### 20.1. Искусственные искажения

```text
D1 Gaussian blur
D2 motion blur optional
D3 Gaussian noise
D4 low contrast
D5 erosion / broken strokes
D6 dilation / thick strokes
D7 JPEG artifacts
D8 uneven lighting optional
```

Уровни:

```text
mild
medium
strong
```

---

### 20.2. Естественные искажения

HWR200 используется как естественный robustness test:

```text
scan
good light photo
poor light photo
```

Сравнение:

```text
same/similar content under different acquisition conditions, where available
```

---

### 20.3. Основная robustness-таблица

```text
model | clean CER | blur | noise | low contrast | erosion | dilation | HWR200 scan | HWR200 good light | HWR200 poor light | mean relative degradation
```

---

## 21. Ablation study

Core ablations:

```text
A0 full image+graph gated
A1 without graph features
A2 without width_proxy features
A3 without loop/junction features
A4 without graph_quality gate
A5 without feature-preserving preprocessing
A6 without width-aware pruning
A7 without uncertain regions
```

Metrics:

```text
CER clean
CER distorted
relative degradation
Skeleton F1
Loop score
CTE
```

---

## 22. Failure analysis

Обязательные категории failure cases:

```text
1. graph helps: image-only ошибается, image+graph исправляет;
2. graph hurts: image-only прав, image+graph ошибается;
3. both fail: обе модели ошибаются;
4. graph extraction fail: плохой граф из-за preprocessing;
5. low informativeness: образец слишком короткий/бедный;
6. ambiguous handwriting: неоднозначные соединения или формы;
7. noise-driven topology error: шум создаёт ложные ветки;
8. lost loop / broken connection / false junction.
```

Для каждого класса нужно сохранять визуальные примеры:

```text
original image
binary image
skeleton
graph overlay
prediction image-only
prediction image+graph
graph warnings
structural explanation
```

---

## 23. Критерии готовности Core MVP

Core MVP считается готовым, если выполнено:

```text
1. Утверждены тема, проблема, объект, предмет, цель, задачи и гипотезы.
2. Подготовлен единый data pipeline для IAM, Cyrillic Handwriting Dataset, HWR200 и HKR.
3. Реализованы OCR-normalized и feature-preserving preprocessing streams.
4. Реализован CRNN+CTC image-only baseline.
5. Реализован classical graph extractor.
6. Для образца можно сохранить graph.json и SVG/PNG overlay.
7. Создан gold subset с двумя аннотаторами.
8. Реализованы structural metrics.
9. Реализованы graph features и graph-only sanity baseline.
10. Реализована хотя бы concat fusion; желательно gated fusion.
11. Проведены clean и distorted experiments.
12. Проведён HWR200 real-condition robustness test.
13. Проведён ablation study.
14. Подготовлен failure analysis.
15. Сформулированы ограничения и future work.
```

---

## 24. Критерии научной успешности

### Минимально успешный результат

Работа научно состоятельна, если:

```text
1. HI-CSG-R формально определён;
2. classical graph pipeline воспроизводимо работает;
3. graph quality измеряется на gold subset;
4. structural metrics выявляют типовые ошибки;
5. failure analysis показывает условия применимости;
6. результаты честно описывают, где graph branch не помогает.
```

Даже если image+graph не улучшит CER, работа может быть защищаемой при наличии сильного структурного анализа.

---

### Хороший результат

```text
1. image+graph не хуже image-only на clean test;
2. image+graph показывает меньший relative CER degradation при нескольких искажениях;
3. graph quality коррелирует с OCR errors;
4. graph overlay помогает объяснять ошибки;
5. CTE и Loop Score показывают содержательные различия между variants.
```

---

### Сильный результат

```text
1. gated fusion лучше concat fusion;
2. graph_quality gate снижает вред от плохих графов;
3. robustness улучшается на artificial и natural distortions;
4. результаты стабильны на IAM, Cyrillic и HWR200/HKR;
5. failure analysis показывает не только успехи, но и ясные границы метода.
```

---

## 25. Ограничения исследования

```text
1. Из offline-изображения невозможно надёжно восстановить реальный порядок движения пера.
2. HI-CSG-R является каноническим представлением видимой структуры, а не записью реального процесса письма.
3. Width proxy зависит от ручки, бумаги, освещения, скана, бинаризации и толщины линии, поэтому не является настоящим нажимом.
4. Smoothness/tempo proxy не является настоящей скоростью письма.
5. Graph extraction чувствителен к preprocessing.
6. Gold subset ограничен по размеру и не покрывает всё разнообразие рукописей.
7. CTE использует экспертные веса и требует sensitivity analysis.
8. Graph-only baseline не обязан конкурировать с image-only HTR.
9. Данные разных датасетов различаются по уровню: word, phrase, line, page, symbol.
10. Page-level datasets требуют отдельного detection/cropping pipeline и не должны смешиваться с line/word results без контроля.
11. Historical domain shift на Digital Peter не является основным доказательством гипотезы.
12. Графовое представление может ухудшать распознавание при низком graph quality.
```

---

## 26. Риски и меры контроля

### Риск 1. Classical graph extractor даёт слишком шумные графы

Меры:

```text
feature-preserving preprocessing;
width-aware pruning;
uncertain regions;
graph confidence;
visual error reports;
gold subset tuning.
```

---

### Риск 2. Fusion не улучшает CER

Меры:

```text
не заявлять гарантированное улучшение;
оценивать robustness отдельно;
проверять graph quality vs CER;
делать failure analysis;
показывать интерпретируемость как самостоятельный результат.
```

---

### Риск 3. Разметка gold subset неоднозначна

Меры:

```text
annotation guideline;
второй аннотатор;
inter-annotator agreement;
adjudicated gold;
uncertain regions вместо насильственного выбора.
```

---

### Риск 4. Разные уровни данных ломают сравнение

Меры:

```text
не смешивать word/line/page в одной таблице без указания level;
делать отдельные evaluations по уровням;
использовать crops для graph metrics;
использовать line-level для CTC baseline.
```

---

### Риск 5. Терминология будет воспринята как чрезмерно почерковедческая

Меры:

```text
использовать “почерковедчески мотивированные признаки изображения”;
все динамические и нажимные признаки называть proxy;
не делать идентификационных, медицинских или юридических выводов;
в теоретической части добавить таблицу “понятие → observable proxy → limitation”.
```

---

## 27. Структура будущей работы

```text
Введение
  Актуальность
  Проблема исследования
  Объект и предмет
  Цель и задачи
  Гипотезы
  Научная новизна
  Практическая значимость
  Ограничения

Глава 1. Offline HTR и структурные представления рукописи
  1.1. Image-only HTR models
  1.2. Skeletonization and graph extraction
  1.3. Vector/graph representations of handwriting
  1.4. Почерковедчески мотивированные признаки изображения
  1.5. Метрики OCR и структурной реконструкции

Глава 2. Метод HI-CSG-R
  2.1. Формальная постановка
  2.2. Уровни графа
  2.3. Node/edge schema
  2.4. Classical graph extraction pipeline
  2.5. Graph features and confidence
  2.6. Visualization and error reports

Глава 3. Данные и экспериментальный протокол
  3.1. Датасеты
  3.2. Data pipeline
  3.3. Gold subset
  3.4. Annotation protocol
  3.5. Metrics
  3.6. Models and baselines
  3.7. Robustness protocol

Глава 4. Результаты
  4.1. Image-only baseline
  4.2. Graph extraction quality
  4.3. Graph-only sanity baseline
  4.4. Image+graph fusion
  4.5. Robustness experiments
  4.6. Ablation
  4.7. Failure analysis

Глава 5. Обсуждение
  5.1. Проверка гипотез
  5.2. Где граф помогает
  5.3. Где граф не помогает
  5.4. Ограничения
  5.5. Future work

Заключение
Приложения
  A. HI-CSG-R JSON schema
  B. Annotation guideline
  C. Dataset converters
  D. Additional visual reports
```

---

## 28. Ближайшие deliverables после Этапа 0

После утверждения этого документа нужно сразу перейти к Этапу 1.

### Документы

```text
docs/00_research_problem.md
docs/01_hi_csg_r_schema.md
docs/02_annotation_guidelines.md
docs/03_metrics.md
docs/04_experiment_protocol.md
docs/05_limitations.md
```

### Кодовые заготовки

```text
src/datasets/registry.py
src/datasets/iam.py
src/datasets/cyrillic_handwriting.py
src/datasets/hwr200.py
src/datasets/hkr.py
src/datasets/validate_dataset.py
src/preprocessing/ocr_preprocess.py
src/preprocessing/feature_preprocess.py
```

### Первый практический шаг

```text
Сделать data audit для IAM + Cyrillic Handwriting Dataset + HWR200 subset + HKR metadata.
```

Цель первого практического шага — не обучение моделей, а проверка данных: формат, размеры, транскрипции, writer_id, уровни разметки, битые изображения, дубликаты, распределение длины, качество кропов и применимость к graph extraction.

---

## 29. Финальная формулировка проекта

В данной работе разрабатывается метод HI-CSG-R для построения канонического графа видимой штриховой структуры offline-изображений рукописного текста. Метод не восстанавливает реальную траекторию пера, а формирует воспроизводимое структурное представление видимых линий, соединений, петель, пересечений, разрывов и зон рукописи. На основе русско-английских датасетов создаётся gold subset для оценки качества графа, вводятся structural metrics и проверяется, может ли добавление графовых признаков к image-only HTR-модели повысить устойчивость распознавания при шуме, размытии, плохом освещении и повреждениях штрихов. Основной результат работы — экспериментальная оценка применимости графового представления для интерпретируемого и робастного offline HTR, а не утверждение о создании универсально лучшей OCR-модели.
