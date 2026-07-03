# iter1.3.2 plan ready

**User:** Anonymous  
**Created:** 5/24/2026 14:47:23  
**Updated:** 5/24/2026 17:25:37  
**Exported:** 7/2/2026 21:03:01  
**Link:** [https://chatgpt.com/g/g-p-6a118b2fcd64819186d96d0543629e0a/c/6a12e48e-a3cc-83eb-a557-1b5e7144fcf8](https://chatgpt.com/g/g-p-6a118b2fcd64819186d96d0543629e0a/c/6a12e48e-a3cc-83eb-a557-1b5e7144fcf8)  

## Prompt:
23/05/2026, 14:19:33

я хочу написать научную работу, суть которой - распознавание рукописных текстов (рукописей) - например со сканов, акцент на русский/английский

на выходе должно получаться структурная векторизация / графовое штриховое представление рукописного текста (не движение пера, а именно векторизация)

изучи научные статьи и работы использующие этот подход, выясни какой простор для новой научной работы

## Response:
23/05/2026, 14:19:34

Я разберу это как обзор литературы: выделю ветки работ по скелетизации/векторизации штрихов, графовым представлениям, HTR/OCR, и затем сформулирую возможные научные “ниши” именно для русско-английских рукописей со сканов.

> **Browsing for research on handwriting vectorization and recognition**
>
>
> **Searching for Cyrillic handwritten text recognition datasets**
>
>
> **Searching for offline HTR datasets and baselines**
>
>
> **Looking into vectorization methods for line drawings**
>
> I need to explore methods for vectorizing line drawings, maybe focusing on techniques like DeepSVG, DiffVG, or neural stroke approaches. I should search for papers related to line drawing vectorization to find more details on how they handle this kind of process.
>
> **Searching for papers on line drawing vectorization**
>
>
> **Searching for stroke recovery datasets**
>
> I need to look into datasets, both online and offline, that might be useful for stroke recovery research. The goal is to find paired datasets that could help in studying recovery patterns or rehabilitation. I'll search to find options.
>
> **Searching for paired stroke recovery datasets online/offline**
>
>
> **Searching for historical handwritten document recognition methods**
>
> I’m thinking about graph-based methods for keyword spotting in historical handwritten documents. It might help to focus on techniques involving skeleton features for handwriting recognition. I'll search to find relevant information on how graphs are used for this kind of task, particularly for older handwritten data.
>
> **Searching for graph-based handwriting recognition techniques**
>
>
> **Looking for Russian stroke representation papers**
>
> I need to find papers specifically focused on stroke representation, particularly related to Russian writing. I'll search for resources that address this topic in detail, especially any unique approaches or techniques used in representing strokes in the Russian language. Let's see what I can find.
>
> **Searching for Russian stroke representation papers**
>
>
> **Clarifying requirements for analyzing PDFs**
>
> When analyzing PDFs, I need to take screenshots, as it’s important to include them. Since we’ve already parsed the text from the PDF, the instructions say I should use screenshots when analyzing them—possibly for visual inspection. I'll probably focus on the first page and use it as reference for citations.
>
> **Capturing PDF screenshots for analysis compliance**
>
>
> **Searching for HTR model baselines and TrOCR info**
>
> I need to look up more information on HTR model baselines, particularly on TrOCR. It would be useful to check for any official sources or papers by Microsoft on TrOCR to get a clearer understanding of its baseline and performance. Let’s search for that now.
>
> **Searching for sources on TrOCR model baselines**
>
>
> **Considering inclusion of PDF screenshot for TrOCR**
>
> If I use a PDF related to TrOCR, I might need to include a screenshot, as the instructions say it's important. If I include information derived from the PDF, the screenshot should be used as a reference. I need to ensure that's taken care of properly.
>
> **Searching for TrOCR PDF and screenshot references**
>
>
> **Searching for TrOCR PDF citations and screenshots**
>
>
> **Considering structure and sources for HWR200 and Cyrillic models**
>
> I think I need to focus on the HWR200 page while avoiding specific citation but still provide a structured answer in Russian. Adding "простор" could be useful for the new paper. I might need to mention line vectorization broadly, not just for handwriting, and identify the gap. I'll also explore additional sources on Cyrillic models.
>
> **Analyzing handwriting recognition and Cyrillic models**
>
>
> **Searching for complete HKR lines information**
>
>
> **Structuring comprehensive response on HTR research**
>
> Since the user asks for a comprehensive review of scientific articles, my response should be broad. I'll mention "научная работа можно сделать..." and "state of art landscape." It's important to highlight the difference between vectorization and pen movement. I'll clarify that current HTR focuses on raster models, while vectorization is more common in keyword spotting or math symbols. I'll also add that Russian Cyrillic datasets are underdeveloped in terms of stroke graph annotations.
>
> **Structuring new research on stroke graph vectorization**
>
> The new work involves a benchmark method for "language-independent stroke graph vectorization" of offline handwritten lines. It will feature graph/raster mutual learning, self-supervised pseudo-labeling, and topology-preserving evaluation. Currently, datasets like HWR200, HKR, and those for Cyrillic text only provide image/text labels, not graph/ vector labels. I'll include a "what has been done" table, with citations for each fact, and discuss research questions in terms of novelty, for example, on order-free graph vectorization versus trajectory recovery.
>
> Thought for 1m 13s

Да, пространство для новой научной работы есть - особенно если **не восстанавливать “истинное” движение пера**, а строить **структурную векторную/графовую модель видимых штрихов** из скана и использовать её как промежуточное представление для HTR/OCR, поиска, сравнения почерков и генерации.

## 1. Что уже сделано

Современный HTR в основном ушёл в сторону end-to-end нейросетей: CNN/RNN/CTC, encoder-decoder, Vision Transformer, TrOCR и мультимодальные модели. Например, TrOCR использует image Transformer как визуальный энкодер и text Transformer как декодер и показывает сильные результаты на печатном, рукописном и scene text OCR без сложного pre/post-processing. Это сильная baseline-линия, но она почти не даёт явной геометрической структуры штрихов. ([ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/26538/26310))

Отдельная ветка - **stroke/trajectory recovery**: из статического изображения пытаются восстановить последовательность точек, как будто текст был записан онлайн. TRACE, например, расширяет восстановление траекторий до строк произвольной длины и даёт benchmark для IAM-On/IAM-Off; но это всё равно задача восстановления порядка движения, то есть более сильное и часто неоднозначное предположение, чем нужно тебе. ([arXiv](https://arxiv.org/pdf/2105.11559))

Ближе к твоей идее находятся работы, где текст переводится в **скелетный граф**. Работа МГУ “Pen Trace Reconstruction with Skeleton Representation…” формулирует задачу как построение обхода по skeleton graph, покрывающего все рёбра, с декомпозицией графа на структурные элементы. Важно: авторы прямо пишут, что offline → online может уменьшить размерность и шум, но требует реконструкции траектории, а универсального решения нет. ([ceur-ws.org](https://ceur-ws.org/Vol-2744/paper27.pdf))

Есть и более “векторный” подход: в работах по keyword spotting для исторических рукописей слово после бинаризации/скелетизации представляется графом; узлы - ключевые точки, промежуточные точки или элементы сетки, далее используется graph matching / graph edit distance. Это уже не OCR в классическом смысле, а структурное представление для поиска похожих слов. ([iCoSys](https://icosys.ch/wp-content/papercite-data/pdf/stauffer18keyword.pdf))

В 2024 году для русских рукописных документов появилась работа по поиску ключевых слов, где текст представляется как последовательность штрихов - подграфов скелетного графа; штрихи делятся на кольцевые и ветвевые, а расстояние между словами строится через выравнивание последовательностей классов штрихов. Это очень близко к твоей теме, но задача там - keyword spotting, а не универсальная векторизация/HTR. ([isprs-archives.copernicus.org](https://isprs-archives.copernicus.org/articles/XLVIII-2-W5-2024/49/2024/isprs-archives-XLVIII-2-W5-2024-49-2024.pdf))

Параллельно развивается **line drawing vectorization**: модели превращают растровые line art изображения в векторные stroke primitives. Например, AAAI-2022 предлагает end-to-end framework, который напрямую генерирует векторные штрихи из растрового line drawing и обучается под raster + vector supervision. Но это не заточено под рукописный текст, языковые особенности, кириллицу/латиницу и downstream HTR. ([cdn.aaai.org](https://cdn.aaai.org/ojs/20379/20379-13-24392-1-2-20220628.pdf))

Недавняя работа BMVC-2025 по Bézier curve-based stroke extraction показывает ещё один важный путь: не просто скелет, а параметрические кривые Безье, fitted to skeleton. Авторы подчёркивают, что пиксельные методы часто дробят штрихи и плохо сохраняют топологию; их Bézier-представление пригодно для генерации, интерполяции и OCR-аугментации. Однако работа сфокусирована на символах, а не на строках/словах русского и английского текста. ([bmva-archive.org.uk](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_478/paper.pdf))

## 2. Что есть по русскому/кириллическому направлению

Для кириллицы данные и работы заметно беднее, чем для английского. HKR содержит примерно 95% русских и 5% казахских слов/предложений, более 1400 заполненных форм, около 63 000 предложений и более 715 699 символов примерно от 200 авторов. ([arXiv](https://arxiv.org/abs/2007.03579))

HWR200 - открытый датасет русских рукописных текстов: 200 авторов, разные почерки, разные условия фотографирования, всего 30 030 изображений. Авторы прямо позиционируют его как ресурс для HTR, поиска повторов и задач, где нужно работать с разнообразными реальными условиями съёмки. ([Диалог](https://dialogue-conf.org/media/5925/potyashiniplusetal048.pdf))

Есть также украинско-кириллическая работа “On recognition of Cyrillic Text”, где авторы подчёркивают нехватку публичных кириллических датасетов, создают synthetic Cyrillic text и handwritten Cyrillic text, а на handwritten test получают CER 4.8 и WER 21.1. ([OpenReview](https://openreview.net/pdf?id=Ske6GT9c8r))

На уровне готовых моделей уже есть fine-tuned TrOCR для Church Slavonic, Russian и Ukrainian handwriting, но он остаётся image-to-text моделью: ViT encoder + language decoder, без явного штрихового/графового слоя. ([Hugging Face](https://huggingface.co/cyrillic-trocr/trocr-handwritten-cyrillic))

## 3. Главный научный зазор

Существующие направления пока плохо соединены:

| Направление | Что умеет | Ограничение для твоей идеи |
|---|---|---|
| End-to-end HTR / TrOCR / ViT | Хорошо распознаёт текст | Почти нет интерпретируемой геометрии штрихов |
| Trajectory recovery | Пытается восстановить порядок письма | Порядок пера неоднозначен и не всегда нужен |
| Skeleton graph / graph matching | Даёт структурное представление | Часто используется для word spotting, не как универсальный слой HTR |
| Line drawing vectorization | Даёт SVG/Bezier/stroke primitives | Обычно не учитывает рукописный текст, буквы, язык, OCR |
| Cyrillic HTR | Есть датасеты и модели | Почти нет работ по русской stroke-level vectorization |

Отсюда хорошая формулировка новизны:

**“Order-invariant structural stroke graph vectorization for offline Russian/English handwritten text recognition.”**

То есть вместо восстановления “как двигалась рука” ты предлагаешь восстановить **что геометрически написано**: граф видимых штрихов, кривые, узлы, пересечения, петли, толщины, направления касательных, локальную топологию, связи между штрихами и их вероятную принадлежность к символам/словам.

## 4. Самые перспективные темы для новой статьи

### Тема A. Векторно-графовое представление как промежуточный слой HTR

Можно построить pipeline:

**scan/photo → binarization/segmentation → skeleton/medial axis → stroke graph → Bezier/polyline vectorization → graph encoder → text decoder**

Новизна будет не в том, чтобы просто распознать текст, а в том, что модель создаёт **интерпретируемое структурное представление**, которое можно визуализировать, редактировать, сравнивать, хранить и использовать для downstream задач.

Сильная гипотеза: добавление stroke graph branch к raster branch улучшит устойчивость на малых данных, шумных сканах, редких почерках и кросс-языковом переносе RU/EN. Это логично соотносится с IJCAI-2025, где graph-enhanced cross-modal mutual learning мотивируется тем, что обычные HTR-модели игнорируют внутреннюю геометрию символов. ([ijcai.org](https://www.ijcai.org/proceedings/2025/574))

### Тема B. Не trajectory recovery, а “canonical stroke graph”

Это, на мой взгляд, самая чистая ниша. Вместо того чтобы угадывать порядок пера, можно определить каноническое представление:

- узлы: endpoints, junctions, extrema curvature, loop centers, intersections;
- рёбра: скелетные ветви или fitted Bézier-сегменты;
- атрибуты: длина, кривизна, толщина штриха, направление касательной, локальная плотность, loop/branch type;
- граф верхнего уровня: связи штрихов в буквы, буквы в слово, слова в строку.

Такой граф можно сделать **инвариантным к порядку письма**, но чувствительным к форме. Это выгодно: рукопись со скана часто не содержит достоверной информации о lift-up и временном порядке, а попытка восстановить порядок может вносить ложную “уверенность”.

### Тема C. Датасет/benchmark для RU+EN stroke graph vectorization

Сейчас у IAM/HKR/HWR200 есть изображения и текстовые аннотации, но нет общепринятого benchmark-а “image → stroke graph / vector representation”. Можно создать полуавтоматическую разметку:

1. взять IAM для английского и HWR200/HKR для русского;
2. прогнать скелетизацию и Bezier/polyline fitting;
3. вручную проверить небольшой gold subset;
4. остальное использовать как pseudo-labels;
5. оценивать не только CER/WER, но и качество векторизации.

Это может быть самостоятельный вклад: **Russian-English Offline Handwriting Stroke Graph Benchmark**.

### Тема D. Метрики качества векторизации рукописей

Здесь тоже есть простор. Для OCR обычно считают CER/WER, но для твоей задачи нужны метрики представления:

- raster reconstruction IoU/F-score после обратного рендеринга графа;
- Chamfer/Hausdorff distance между скелетом исходника и скелетом реконструкции;
- graph edit distance / topology error: потерянные петли, лишние junctions, разрывы;
- compression ratio: сколько параметров нужно против bitmap;
- downstream CER/WER: помогает ли граф распознаванию;
- writer/style preservation: сохраняет ли граф индивидуальные особенности почерка.

Особенно интересно сделать метрику, которая штрафует не только пиксельные ошибки, но и **топологические ошибки**: потеря петли в “о”, неверное соединение в “т/ш/щ”, ложный разрыв в cursive writing.

### Тема E. Двуязычная русско-английская модель

Русский и английский вместе дают хороший научный материал: латиница и кириллица частично похожи визуально, но имеют разные частотные паттерны, лигатуры и рукописные формы. Новизна может быть в том, чтобы обучать **language-agnostic stroke graph encoder**, а затем подключать разные языковые декодеры.

Можно исследовать transfer learning:

- train graph vectorizer on English IAM;
- adapt to Russian HWR200/HKR;
- compare raster-only vs graph-only vs raster+graph;
- проверить few-shot adaptation на новых авторах.

## 5. Возможная формулировка научной работы

**Название:**  
“Structural Stroke Graph Vectorization for Offline Russian and English Handwritten Text Recognition”

**Цель:**  
разработать метод, который из растрового изображения рукописного текста строит компактное графово-векторное представление видимых штрихов и использует его для распознавания и анализа рукописей.

**Научная новизна:**  
не восстановление движения пера, а построение order-invariant stroke graph; объединение raster branch и graph branch; benchmark и метрики для RU/EN offline handwriting vectorization.

**Методы:**  
skeletonization / medial axis, graph pruning, Bezier/polyline fitting, GNN или Graph Transformer, cross-modal learning, CTC/Transformer decoder, self-supervised reconstruction.

**Эксперименты:**  
IAM для английского; HKR и HWR200 для русского; сравнение с TrOCR/CRNN/ViT baseline; ablation: без графа, только граф, raster+graph; robustness tests на шум, blur, плохое освещение, разные авторы.

## 6. Самый сильный вклад, который можно заявить

Я бы строил работу вокруг трёх вкладов:

1. **Новый тип представления:** canonical stroke graph / Bézier stroke graph для offline рукописного текста без восстановления порядка пера.  
2. **Модель:** hybrid raster-graph HTR, где графовая ветка даёт структурные признаки, а raster ветка - визуальную детализацию.  
3. **Benchmark:** RU+EN набор и метрики для оценки именно качества структурной векторизации, а не только OCR.

Такую работу можно позиционировать между HTR, document analysis, graph representation learning и image vectorization. Она не конкурирует напрямую с TrOCR “кто меньше CER”, а добавляет новый уровень: **распознавание + интерпретируемая структурная векторизация рукописи**.

---

**Sources:**

- [TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models](https://ojs.aaai.org/index.php/AAAI/article/view/26538/26310)
- [](https://arxiv.org/pdf/2105.11559)
- [](https://ceur-ws.org/Vol-2744/paper27.pdf)
- [Keyword spotting in historical handwritten documents based on graph matching](https://icosys.ch/wp-content/papercite-data/pdf/stauffer18keyword.pdf)
- [](https://isprs-archives.copernicus.org/articles/XLVIII-2-W5-2024/49/2024/isprs-archives-XLVIII-2-W5-2024-49-2024.pdf)
- [End-to-End Line Drawing Vectorization](https://cdn.aaai.org/ojs/20379/20379-13-24392-1-2-20220628.pdf)
- [](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_478/paper.pdf)
- [HWR200: New open access dataset of handwritten texts images in Russian](https://dialogue-conf.org/media/5925/potyashiniplusetal048.pdf)
- [](https://openreview.net/pdf?id=Ske6GT9c8r)
- [cyrillic-trocr/trocr-handwritten-cyrillic · Hugging Face](https://huggingface.co/cyrillic-trocr/trocr-handwritten-cyrillic)
- [Structure-Aware Handwritten Text Recognition via Graph-Enhanced Cross-Modal  Mutual Learning | IJCAI](https://www.ijcai.org/proceedings/2025/574)

## Prompt:
23/05/2026, 15:01:57

смотри вот что еще удалось найти по этой теме изучи это и то что ты искал ранее и подумай, какую в итоге работу выбрать (я хочу самую интересную и не простую) и сразу предложи возможные подходы для решения

[Attachment: Исполнительное резюме.pdf]

## Response:
23/05/2026, 15:01:58

Я сопоставлю твой PDF с найденной ранее литературой и отфильтрую идеи по двум критериям: научная новизна и “непростота” реализации. В результате выберу одну основную тему, а не список равнозначных вариантов.

Из PDF важный вывод усиливает предыдущую картину: самый большой пробел не в “ещё одном распознавателе”, а в отсутствии полноценного слоя **image → stroke graph/vector → text** с нормальными метриками и русско-английской проверкой.

> **Searching for relevant papers**
>
> I need to gather information from a web search, possibly focusing on specific, relevant titles. I'll aim to get reliable sources to support my response, making sure to include citations where appropriate. Let's see what I can find — I'll focus on searching extensively but with a targeted approach.
>
> **Searching for papers on handwritten expression recognition**
>
>
> **Searching for vectorization and OCR tools**
>
> I need to look up some information on raster-to-vector conversion, specifically related to DiffVG and deepvectorization. I'll also check out TrOCR, along with datasets like IAM, HKR, or HWR200, to gather relevant details and see how these tools and resources connect to my task.
>
> **Searching for vectorization techniques and related papers**
>
>
> **Searching for HKR and Digital Peter citations**
>
> I should look up HKR and Digital Peter to find relevant citations. This will help ensure I have credible sources to back up my response. Let me search for these references and see what I can find to support my answer accurately.
>
> **Searching for HKR and Digital Peter citations**
>
>
> **Finalizing answer and addressing citation format**
>
> I need to finalize the answer and make sure it's substantial, but not too lengthy. I'll also mention this "непростая" topic. Regarding citations, the output from one tool doesn’t include line numbers, which is needed for exact references. So, I might need to adjust and just use the citation markers instead.
>
> **Final answer in Russian with citations**
>
> I need to finalize the answer in Russian, possibly discussing the paper "Order-invariant differentiable stroke graph vectorization for offline Russian/English handwriting recognition" with the subtitle about Canonical Stroke Graph (CSG). I'll include citations, both web-based and possibly from an uploaded PDF as "в резюме," and mention the novelty of this approach.
>
> **Searching for updated information on graph neural networks**
>
> I need to verify whether "Graph Neural Network based Handwritten Trajectories Recognition" is as important as I thought. I’ll search for the latest research, specifically focusing on a 2024 paper from arXiv about this topic. This will help ensure I'm working with the most recent and relevant information.
>
> **Searching for 2024 arXiv papers on handwritten trajectories**
>
>
> **Formulating detailed approach with citations**
>
> I need to formulate an answer that includes possible approaches broken into phases. I’ll mention that similar citations from both files and the web can be used. I’ll also focus on incorporating key citations, like those for structural vectorization, stroke extraction, and the specific datasets. Additionally, I’ll emphasize risks and mitigation strategies while considering depth without over-verbosity. It should fit the balance the user seems to want.
>
> Thought for 50s

Мой итоговый выбор: **не делать просто OCR и не делать просто восстановление движения пера**. Самая интересная и достаточно сложная работа - это:

# Структурная векторизация офлайн-рукописей через canonical stroke graph для русско-английского HTR

Рабочее название:

**Canonical Stroke Graph Vectorization for Offline Russian and English Handwritten Text Recognition**

Идея: из скана/фото рукописи строить **каноническое графово-векторное представление видимых штрихов**, а затем использовать его для распознавания текста.

Ключевое отличие: мы **не утверждаем, что восстанавливаем реальную траекторию пера**. Мы восстанавливаем **структуру видимого письма**: скелет, ветвления, петли, сегменты, кривые Безье/полилинии, связи между штрихами. Это делает работу более честной научно и более новой.

---

## Почему именно эту тему стоит выбрать

В твоём PDF хорошо сформулирован главный разрыв: есть скелетизация, графы, GNN, векторизация и HTR, но значительная часть работ изучает отдельные шаги, а не полный конвейер “пиксели → структурный граф/вектор → текст”; также нет массовых датасетов с разметкой на уровне штрихов и нет устоявшихся метрик качества извлечённых штрихов. `Исполнительное резюме.pdf`

Ранее найденная литература это подтверждает. TrOCR и похожие модели хорошо решают image-to-text задачу, но не дают явного структурного представления штрихов. TrOCR использует предобученные image Transformer и text Transformer и позиционируется как end-to-end OCR-система для печатного, рукописного и scene text recognition. ([arXiv](https://arxiv.org/abs/2109.10282?utm_source=chatgpt.com))

С другой стороны, Chan строит stroke extraction из офлайн-изображения через скелет: бинарное изображение разбивается на junctions и segments, затем сегменты объединяются в strokes, а порядок нормализуется через recursive projection и topological sort. Но эта работа сфокусирована на математических выражениях и фактически пытается приблизить offline к online-recognition. ([arXiv](https://arxiv.org/abs/1905.06749?utm_source=chatgpt.com))

Работы по графовому HMER тоже показывают, что графовая структура полезна: Xie et al. вводят Edge-weighted GAT для одновременной классификации узлов-символов и рёбер-отношений в рукописных математических выражениях. ([arXiv](https://arxiv.org/abs/2410.18555?utm_source=chatgpt.com)) Но это опять математика, а не обычные русско-английские строки.

Свежая линия с Bézier stroke extraction показывает, что параметрические кривые - перспективный формат: BMVC-2025 работа прямо говорит, что пиксельные методы часто дробят штрихи и плохо сохраняют топологию, а Bézier-подход лучше сохраняет структурные strokes. ([BMVC 2025](https://bmvc2025.bmva.org/proceedings/478/?utm_source=chatgpt.com))

По русскому направлению тоже есть пространство: HKR содержит около 63 000 предложений, более 715 699 символов и примерно 200 авторов на кириллице, а HWR200 - русские рукописные изображения от 200 авторов в разных условиях съёмки. ([arXiv](https://arxiv.org/abs/2007.03579?utm_source=chatgpt.com)) Но эти датасеты в основном дают транскрипты, а не графы/штрихи.

---

# Моя рекомендация

Не выбирать одну из простых тем вроде:

- “GNN для распознавания рукописного текста”;
- “скелетизация русских рукописей”;
- “топологические признаки для классификации букв”;
- “fine-tune TrOCR на русском”.

Это всё либо слишком прикладное, либо слишком узкое, либо будет выглядеть как небольшое улучшение существующих методов.

Лучше выбрать **комбинированную тему**:

> **Разработка order-invariant stroke graph vectorizer для офлайн-рукописей и проверка, улучшает ли такое структурное представление распознавание русско-английского рукописного текста.**

Главный научный тезис:

> Для офлайн-рукописей полезнее восстанавливать не временную траекторию пера, а канонический граф видимых штрихов, который сохраняет топологию письма и может использоваться как интерпретируемое промежуточное представление для HTR.

Это сложнее, интереснее и сильнее, чем просто OCR.

---

## Что именно будет выходом модели

На выходе не просто текст, а две вещи:

```text
input image
   ↓
canonical stroke graph / vector representation
   ↓
recognized text
```

Граф можно задать так:

```text
G = (V, E, A)

V - ключевые точки:
    endpoints,
    junctions,
    loop points,
    extrema curvature,
    sampled control points

E - штриховые сегменты:
    skeleton branches,
    fitted polylines,
    Bézier curves

A - атрибуты:
    thickness,
    curvature,
    tangent direction,
    branch/loop type,
    confidence,
    local raster features
```

Каждое ребро графа можно хранить как:

```text
edge_i = {
    start_node,
    end_node,
    bezier_control_points,
    length,
    curvature,
    stroke_width,
    probability,
    local_direction
}
```

А вся строка превращается в “stroke graph line representation”.

---

# Возможные подходы к решению

## Подход 1. Сильный базовый pipeline: классика + граф + OCR

Это нужно сделать первым, потому что он даст воспроизводимый baseline.

Конвейер:

```text
image
→ preprocessing
→ binarization
→ skeletonization
→ skeleton graph extraction
→ pruning
→ stroke grouping
→ Bézier/polyline fitting
→ graph features
→ recognizer
```

Технически:

1. Предобработка: deskew, denoise, contrast normalization, Sauvola/Otsu/adaptive threshold.
2. Скелетизация: Zhang-Suen, Guo-Hall, medial axis.
3. Граф: пиксели с количеством соседей ≠ 2 становятся вершинами; цепочки между ними - рёбра.
4. Очистка: удаление коротких “усиков”, сглаживание, объединение почти коллинеарных участков.
5. Векторизация: каждое ребро аппроксимируется polyline или cubic Bézier.
6. Распознавание: признаки графа подаются в CTC/Transformer/Graph Transformer.

Это не самая новая часть, но она нужна как foundation. В PDF именно такой общий путь описан как типовой: предобработка → скелет → граф → сегментация штрихов → аппроксимация кривыми → модель распознавания. `Исполнительное резюме.pdf`

---

## Подход 2. Главная научная часть: neural canonical stroke graph

Это самая интересная часть.

Вместо того чтобы полностью полагаться на эвристики скелетизации, можно обучить модель предсказывать структурные элементы:

```text
image encoder
   ↓
dense maps:
   - skeleton probability
   - endpoint heatmap
   - junction heatmap
   - loop heatmap
   - tangent field
   - stroke width map
   ↓
graph decoder
   ↓
Bézier stroke graph
```

Архитектура может быть такой:

```text
CNN / Swin / ViT encoder
        ↓
multi-head decoder
        ├── skeleton map
        ├── keypoint maps
        ├── tangent field
        ├── stroke width
        └── edge existence matrix
        ↓
graph construction layer
        ↓
Bézier fitting / vector decoder
```

Фишка: модель учится не просто “классифицировать пиксели”, а строить **граф штрихов**.

Для этого можно использовать DETR-подобный set prediction:

```text
N learned stroke queries
    ↓
каждый query предсказывает один stroke:
    - существует / не существует
    - control points Bézier-кривой
    - confidence
    - связи с другими strokes
```

Порядок штрихов не важен, поэтому можно использовать Hungarian matching между предсказанными и псевдо-эталонными штрихами. Это хорошо согласуется с твоей идеей: **не движение пера, а векторизация видимой структуры**.

---

## Подход 3. Self-supervised raster ↔ vector learning

Проблема: нет настоящей stroke-разметки.

Решение: обучать через обратный рендеринг.

```text
image → predicted vector graph → differentiable rasterizer → reconstructed image
```

Loss-функции:

```text
L = L_raster
  + L_skeleton
  + L_keypoint
  + L_topology
  + L_text
```

Где:

- `L_raster`: насколько отрендеренный граф похож на исходную рукопись;
- `L_skeleton`: совпадение со скелетом;
- `L_keypoint`: совпадение junction/endpoints;
- `L_topology`: сохранение числа компонент, петель, ветвлений;
- `L_text`: CER/CTC/seq2seq loss по распознаванию.

Это хорошая научная идея, потому что она соединяет line drawing vectorization и HTR. В line drawing vectorization уже есть end-to-end подходы, которые напрямую выдают vector representation из raster line drawing без отдельного tracing/post-processing. ([cdn.aaai.org](https://cdn.aaai.org/ojs/20379/20379-13-24392-1-2-20220628.pdf?utm_source=chatgpt.com)) Есть также self-supervised идея rasterization/vectorization для sketches and handwriting, где vectorization переводит image space в vector coordinates, а rasterization - обратно. ([arXiv](https://arxiv.org/abs/2103.13716?utm_source=chatgpt.com))

Твоя новизна: применить это не к абстрактным sketch-данным, а к **офлайн-рукописному тексту RU/EN**, с графовой структурой и HTR-метриками.

---

## Подход 4. Графовый распознаватель поверх stroke graph

После получения графа можно строить не только векторизацию, но и распознавание.

Варианты:

### Вариант A: Graph Transformer + CTC

```text
stroke graph
→ Graph Transformer
→ sequence features along x-axis
→ CTC decoder
→ text
```

Здесь граф даёт локальные признаки формы, а CTC решает последовательное распознавание строки.

### Вариант B: Dual encoder

```text
raster image → ViT/CNN encoder ┐
                               ├── fusion → text decoder
stroke graph → Graph encoder ──┘
```

Это сильный вариант для статьи: сравнить

```text
raster-only
graph-only
raster + graph
```

Если `raster + graph` лучше на шумных, размытых, исторических и кириллических данных - это будет сильный результат.

### Вариант C: Graph-to-text Transformer

```text
nodes/edges as tokens
→ graph positional encoding
→ Transformer decoder
→ text
```

Здесь можно использовать координатные positional encodings, Laplacian eigenvectors, признаки кривизны и толщины.

---

## Подход 5. Русско-английская cross-script проверка

Это обязательная часть, потому что она делает работу заметнее.

Эксперименты:

```text
train on English IAM
test/fine-tune on Russian HWR200/HKR/Digital Peter

train on Russian
test on English

train mixed RU+EN
compare with monolingual
```

Идея: stroke graph должен быть более language-agnostic, чем чисто raster/image модель. То есть он может лучше переноситься между латиницей и кириллицей.

Датасеты:

- **IAM** - английский benchmark для offline handwriting;
- **HKR** - русский/казахский кириллический корпус, около 63 000 предложений и более 715 699 символов; ([arXiv](https://arxiv.org/abs/2007.03579?utm_source=chatgpt.com))
- **HWR200** - русский корпус от 200 авторов; ([Hugging Face](https://huggingface.co/datasets/AntiplagiatCompany/HWR200?utm_source=chatgpt.com))
- **Digital Peter** - исторический русский набор из 9 694 строк рукописей Петра I. ([arXiv](https://arxiv.org/abs/2103.09354?utm_source=chatgpt.com))

---

# Как сформулировать вклад статьи

Я бы заявил 4 вклада.

## 1. Canonical Stroke Graph

Новый формат промежуточного представления офлайн-рукописи:

```text
not pen trajectory,
not contour vectorization,
but canonical visible stroke graph
```

То есть ты не споришь с тем, что “истинную траекторию пера” невозможно надёжно восстановить из скана. Ты говоришь: нам это и не нужно.

---

## 2. Hybrid raster-graph HTR model

Модель, которая использует и изображение, и граф:

```text
image branch + stroke graph branch → recognition
```

Это сильнее, чем просто GNN или просто TrOCR.

---

## 3. Weak/self-supervised training without stroke labels

Поскольку штриховых аннотаций нет, ты предлагаешь:

```text
pseudo-labels from skeletonization
+ synthetic data
+ online-to-offline rendered data
+ differentiable rasterization
+ small manually corrected subset
```

Это закрывает главный практический разрыв, который отмечен в PDF: отсутствие стандартных датасетов с разметкой штрихов. `Исполнительное резюме.pdf`

---

## 4. Structural metrics beyond CER/WER

Кроме CER/WER, ты вводишь метрики качества векторизации:

```text
Raster reconstruction IoU
Skeleton Chamfer distance
Endpoint/Junction F1
Loop preservation score
Graph edit distance
Bézier fitting error
Topology error
CER/WER after recognition
```

Это важно, потому что CER/WER могут быть хорошими даже при плохом графе, а твоя работа именно про структурное представление.

---

# Основной экспериментальный дизайн

## Эксперимент 1. Качество структурной векторизации

Сравнить:

```text
Potrace / contour vectorization
classical skeleton graph
Chan-like skeleton stroke extraction
neural stroke graph vectorizer
hybrid neural + skeleton approach
```

Метрики:

```text
reconstruction IoU
skeleton Chamfer
junction F1
endpoint F1
loop preservation
number of broken strokes
average Bézier error
```

---

## Эксперимент 2. Помогает ли граф распознаванию

Сравнить:

```text
TrOCR / CRNN baseline
graph-only model
raster-only model
raster + graph model
```

Метрики:

```text
CER
WER
robustness under noise
robustness under blur
writer-independent split
cross-dataset transfer
```

Самый важный результат: не обязательно побить TrOCR на чистом IAM. Достаточно показать, что графовая ветка помогает на:

- шумных сканах;
- русской скорописи;
- исторических рукописях;
- малом количестве данных;
- cross-script transfer.

---

## Эксперимент 3. Cross-script transfer

```text
EN → RU
RU → EN
EN+RU → RU
EN+RU → historical RU
```

Проверить гипотезу:

> Структурные признаки штрихов лучше переносятся между латиницей и кириллицей, чем чистые raster-признаки.

---

## Эксперимент 4. Ablation study

Убрать по очереди:

```text
без topology loss
без Bézier fitting
без graph branch
без raster branch
без synthetic pretraining
без junction prediction
без loop preservation
```

И показать, что именно даёт вклад.

---

# Самый реалистичный план реализации

## Этап 1. Baseline

Сначала сделать простой pipeline:

```text
image → binarization → skeleton → graph → Bézier edges → SVG/JSON
```

Выход сохранить в формате:

```json
{
  "nodes": [
    {"id": 1, "x": 10, "y": 20, "type": "junction"}
  ],
  "edges": [
    {
      "from": 1,
      "to": 2,
      "type": "stroke_segment",
      "bezier": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "width": 3.1,
      "confidence": 0.92
    }
  ]
}
```

Это уже даст демонстрационный результат: можно визуально показать исходный скан и его stroke graph.

---

## Этап 2. Pseudo-label generation

Создать автоматическую разметку:

```text
IAM + HWR200 + HKR
→ skeleton graphs
→ cleaned graphs
→ pseudo-stroke labels
→ vector JSON/SVG
```

Часть данных вручную поправить: например, 300-500 слов/строк. Этого хватит для gold subset.

---

## Этап 3. Neural vectorizer

Обучить модель предсказывать:

```text
skeleton map
junction map
endpoint map
edge existence
Bézier control points
```

Сначала проще сделать U-Net/HRNet-like dense prediction, затем добавить graph decoder.

---

## Этап 4. Recognition model

Проверить три режима:

```text
image-only
graph-only
image+graph
```

Для image-only можно взять CRNN/TrOCR-like baseline. Для graph-only - Graph Transformer/Graph Attention. Для image+graph - fusion.

---

## Этап 5. Статья

Структура статьи:

```text
1. Introduction
2. Related Work
3. Canonical Stroke Graph Representation
4. Weakly/Self-Supervised Vectorization Method
5. Graph-Aware Handwritten Text Recognition
6. Experiments on English and Russian Handwriting
7. Structural Metrics
8. Discussion and Limitations
```

---

# Какую гипотезу проверять

Я бы сформулировал так:

> **H1:** Canonical stroke graph сохраняет топологическую и геометрическую структуру офлайн-рукописи лучше, чем контурная векторизация и эвристическое выделение штрихов.

> **H2:** Добавление stroke graph branch к raster HTR модели улучшает устойчивость распознавания на шумных, малоресурсных и cross-script сценариях.

> **H3:** Для русского и английского рукописного текста order-invariant stroke graph является более переносимым представлением, чем восстановленная последовательность движения пера.

---

# Почему это “непростая” работа

Она сложная сразу на нескольких уровнях:

1. **Computer vision:** скелетизация, графы, keypoints, кривые.
2. **Graph ML:** GNN/Graph Transformer, node/edge features.
3. **Differentiable rendering:** обратное сравнение векторного графа с растровым изображением.
4. **HTR:** CTC/Transformer decoding, CER/WER.
5. **Datasets:** русские и английские корпуса, отсутствие stroke labels.
6. **Metrics:** нужно оценивать не только текст, но и структуру.
7. **Scientific positioning:** работа находится между OCR, document analysis, sketch vectorization и graph representation learning.

Именно поэтому она может быть интересной.

---

# Финальный выбор

Я бы выбрал вот такую формулировку:

> **Разработка слабонаблюдаемого метода канонической графово-векторной реконструкции штрихов офлайн-рукописей и его применение к русско-английскому распознаванию рукописного текста.**

Или короче:

> **Weakly Supervised Canonical Stroke Graph Vectorization for Offline Russian-English Handwriting Recognition.**

Это лучше, чем “распознавание рукописей через GNN”, потому что здесь есть настоящий исследовательский вклад: **новое промежуточное представление**, **новая постановка без восстановления движения пера**, **метрики структурного качества**, **русско-английская проверка**, **связь с downstream HTR**.

---

**Sources:**

- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282?utm_source=chatgpt.com)
- [Bézier Curve-Based Stroke Extraction for Handwritten ...](https://bmvc2025.bmva.org/proceedings/478/?utm_source=chatgpt.com)
- [End-to-End Line Drawing Vectorization](https://cdn.aaai.org/ojs/20379/20379-13-24392-1-2-20220628.pdf?utm_source=chatgpt.com)
- [AntiplagiatCompany/HWR200 · Datasets at ...](https://huggingface.co/datasets/AntiplagiatCompany/HWR200?utm_source=chatgpt.com)

## Prompt:
23/05/2026, 16:01:37

хорошо, тогда берем эту тему, я хочу чтобы ты изучил полностью пару научных материалов по почерковедению и подумал какие фундаментальные идеи или принципы пригодятся нам для разработки модели, я отправлю тебе материалы

## Response:
23/05/2026, 16:01:39

> **Acknowledging user's intent to send materials**
>
> The user mentioned they will send materials, so I should acknowledge that and respond in a way that welcomes the information. "Отлично, присылай" seems appropriate here. I'll note that we're ready to process their materials, mentioning that files, like the "резюме", are already available for reference.
>
> Thought for 8s

Отлично, присылай материалы.

Я буду читать их не как “общую теорию почерка”, а сразу с прицелом на нашу модель:

1. **какие признаки почерка считаются фундаментальными**: нажим, наклон, связность, ритм, интервалы, форма петель, разгоны/торможения, вариативность букв;
2. **что из этого можно наблюдать в офлайн-скане**, а что нельзя честно восстановить;
3. **как перевести почерковедческие признаки в graph/vector representation**;
4. **какие признаки стоит добавить в canonical stroke graph**;
5. **какие идеи можно использовать для архитектуры модели и метрик качества**.

У нас уже есть техническая база по структурной векторизации, скелетизации, графам, GNN и проблеме отсутствия stroke-аннотаций `Исполнительное резюме.pdf`. Теперь нужно добавить второй слой - **почерковедческую теорию**, чтобы наша модель была не просто инженерным OCR-пайплайном, а опиралась на реальные принципы формирования почерка.

## Prompt:
23/05/2026, 16:04:41

исследуй внимательно и в полном объеме максимально полно

[Attachment: Seregin_Pocherkovedenie_2015.pdf]

## Response:
23/05/2026, 16:04:42

Принял. Я разберу PDF как источник понятийной базы для нашей модели: сначала извлеку структуру и ключевые разделы, затем переведу почерковедческие признаки в признаки графа/векторизации/метрики.

> **Analyzing uploaded PDF for research**
>
>
> **Inspecting PDF and extracting text**
>
>
> **Rendering PDF sample output**
>
>
> **Extracting text from PDF file**
>
>
> **Displaying help for PDF extraction and rendering scripts**
>
>
> **Rendering and extracting PDF content**
>
>
> **Extracting table of contents lines**
>
>
> **Extracting table of contents from text**
>
>
> **Considering how to handle document analysis and citations**
>
> I need to extract all headings and paragraphs and check the images—though they might not be necessary. I might process the PDF text and search for core concepts, possibly quoting from page footers. Citing by page number seems like a good idea. The file-search issue could be a hurdle, though.
>
> **Thinking about file citations and current document**
>
> I might need to include file citations, but I’m also thinking about clicking on or accessing the current file to make sure I’m working with the correct version of it. I’ll figure out the best way to approach it in a streamlined way.
>
> **Refining search strategy and markup citations**
>
> I retrieved a large chunk of pages, but I'll need more targeted queries to narrow it down to relevant sections about handwriting, signatures, and diagnostic traits. I think I’ll focus on specific lectures and extract concepts around writing system structure. I’ll cite using full page markers and targeted snippets for relevance.
>
> **Extracting targeted handwriting-related terms from document**
>
>
> **Extracting targeted pages and citations from document**
>
>
> **Focusing on general signs sections**
>
> I should search for pages related to general signs, key concepts that might be discussed under a title like “Общие признаки.” Using a search function like grep should help me isolate these sections more precisely to hone in on the relevant material.
>
> **Extracting general signs from text**
>
>
> **Extracting pages 31-55**
>
> I need to focus on extracting the content from pages 31 to 55. These are likely the specific sections I need for the task, so I'll make sure to grab this range carefully to get the information that's most relevant.
>
> **Extracting pages 31-55 from text**
>
>
> **Continuing with pages 48-55**
>
> I’ll focus on extracting and reviewing pages 48 to 55 next. This range might hold additional relevant details I need to consider, so it’s important to keep gathering the information systematically to make sure I cover all the necessary areas.
>
> **Extracting specific pages from text file**
>
>
> **Extracting mathematical modeling pages 56-79**
>
> I should focus on extracting pages 56 to 79 for the mathematical modeling sections, as they might contain the relevant information I need. These pages seem like they could be crucial in developing a deeper understanding of the material, so it’s worth dedicating time to them.
>
> **Extracting specified pages from text file**
>
>
> **Extracting pages on changed handwriting and diagnostics**
>
> I’m focusing on pages 117-141 and 193-226 for changed handwriting and diagnostic content. I’ll search around terms like "сбивающие" and "измененное" in the text to identify relevant sections. This seems like it could lead to insights into the diagnostic aspects that I need.
>
> **Extracting specific text range from document**
>
>
> **Clarifying query for signs of changed handwriting**
>
> It looks like I need to focus on classic signs of changed handwriting. I'll try a more specific query, either through msearch or grep, to target the exact information related to these signs. That could help me gather the crucial details I'm after.
>
> **Searching for classic signs in text**
>
>
> **Extracting specific text ranges from file**
>
>
> **Extracting information for diagnostic/model purposes**
>
> I need to gather diagnostic or situational data relevant to a specific model. I'll search for pages from 193 and onwards to focus on the right content. This should help me pinpoint the exact information I need for the data or model analysis. Let's continue exploring.
>
> **Extracting lines 9200 to 10450 from text file**
>
>
> **Refining search criteria and diagnostic relevance**
>
> I gathered data from pages 188 to 212, though it's a bit mixed up. I might not need to include the sex or age diagnostic information since it may not be essential, although it's fundamental to some contexts. I'll focus the search on lecture 9, starting from the beginning. Let's proceed.
>
> **Extracting lecture 9 and diagnostic information**
>
>
> **Extracting specific text from provided file**
>
>
> **Planning full analysis and review of lectures**
>
> I need to perform a full analysis—maybe I'll break the text into chunks using Python and search for the top terms. I'll review the lectures:
> 1. Theory: Writing, FDK, properties, tasks.
> 2. Identification features: Handwriting signs—general and private.
> 3. Mathematical modeling.
> 4. Methodology: The general staged approach.
> 5. Changed handwriting, signatures, types, multi-object, diagnostics. 
> I'll summarize key principles for the model and focus on lectures 4, 5, and 9 for methodology details.
>
> **Extracting text for model analysis**
>
>
> **Targeting file extraction with citations**
>
> I think it's best to extract page 93 and nearby pages for relevant information. I'll probably need to use some sort of search or click function for citations, which can guide me toward the right sources from the file. By targeting specific results, I can ensure everything is properly cited.
>
> **Focusing on methodology and mathematics**
>
> I might need to search specifically for content related to methodology and mathematical modeling. That could help streamline things. The "mclick" might not be necessary right now based on what I’m looking for, so I’ll stick to searching the relevant sections for now.
>
> **Separating search for mathematical modeling**
>
> It seems like the search didn’t pull up relevant information for mathematical modeling, possibly because it's not in the parameters. I'll try searching for it separately, focusing more on the specific terms related to mathematical modeling to get better results.
>
> **Reviewing options for file-based and web-based analysis**
>
> It seems the user hasn’t asked for web-based sources, and there are enough resources within the uploaded document. I think comparing with prior literature might not be necessary unless the task involves niche or current info. Reviewing images, such as diagrams on page 14, could be useful, though!
>
> **Rendering pages with key diagrams**
>
> I didn’t render page 14 yet, which has a figure. It would make sense to render that along with page 121, since it probably includes diagrams related to norms. If we're analyzing these diagrams, it's better to render and inspect them. Let’s do that now.
>
> **Rendering selected PDF pages for inspection**
>
>
> **Inspecting images with open_image**
>
> I think I might need to inspect some images, so I should probably use the "open_image" method to do that. This will help me examine the content more closely and ensure I'm catching important details that might be relevant for the task at hand.
>
> **Finalizing the report with citations**
>
> I think I’ll provide the final report in Russian, as the user requested “максимально полно,” while avoiding unnecessary verbosity. I’ll include citations directly from file_search like "turn3file3" or "turn5file0" and ensure they show up inline where needed. No need for web-based citations.
>
> **Structuring the key ideas for the user**
>
> I’m thinking the structure should start with a report on reading 229 pages of 9 lectures. I'll include the main conclusion for our model. Then, I'll break down the handwriting science application, explaining key ideas like information flows, properties, and the sign system. Next, I'll mention the principles and contexts for translation into architecture, especially hierarchical and canonical stroke graphs with specific fields like reliability and variation.
>
> Thought for 3m 29s

Я внимательно разобрал материал **«Почерковедение и почерковедческая экспертиза: курс лекций», 2015, под ред. В. В. Серегина**. Это не статья, а полноценный учебно-методический курс на 229 страниц: теория почерка, система признаков, математическое моделирование, методика экспертного исследования, изменённый почерк, подписи, диагностические и ситуационные задачи. Самое ценное для нашей работы - не криминалистическая часть сама по себе, а **модель почерка как устойчивой, но вариативной двигательной системы**, материально проявленной в рукописи. Это отлично ложится на нашу тему **canonical stroke graph vectorization**.

Ключевой вывод: нашу модель нужно строить не как “нейросеть, которая видит буквы”, а как систему, которая извлекает из изображения **материально наблюдаемый след письменно-двигательного навыка**: форму, связность, ритм, координацию, нажим, вариативность, устойчивые и неустойчивые зоны.

---

# 1. Главная идея из почерковедения

В почерковедении почерк понимается не просто как изображение, а как **зафиксированная система привычных движений**, основанная на письменно-двигательном навыке. В курсе подчёркивается, что письмо включает интеллектуальные, письменно-речевые и письменно-двигательные навыки; последние делятся на технические и графические, а графические включают конфигурационные, пространственно-ориентационные, координационные, скоростные, ритмические, нажимные и другие навыки. `Seregin_Pocherkovedenie_2015.pdf`

Для нас это означает: **графовая векторизация не должна описывать только геометрию контура**. Она должна пытаться восстановить разные слои проявления двигательного навыка:

```text
изображение → видимые следы движения → структурные признаки → текст / стиль / устойчивость
```

Это принципиально отличается от обычного OCR.

Обычный OCR спрашивает:

```text
какие здесь буквы?
```

Наша модель должна спрашивать:

```text
какими устойчивыми движениями, связями, формами и вариациями эти буквы образованы?
```

---

# 2. Самая важная теоретическая связка: три информационных потока письма

В материале описана письменно-двигательная функциональная система: при выполнении рукописи формируются взаимосвязанные потоки обратной информации - **структурно-геометрическая**, **координационно-скоростно-ритмическая** и **нажимная** информация. `Seregin_Pocherkovedenie_2015.pdf`

Это почти готовая архитектурная подсказка для нашей модели.

Я бы прямо заложил три ветви признаков:

```text
1. Structural-geometric branch
   форма, петли, дуги, углы, пересечения, пропорции, положение элементов

2. Coordination-rhythm branch
   плавность, изломы, устойчивость наклона, равномерность размера, связность,
   последовательность и тип соединений

3. Pressure-width branch
   толщина штриха, распределение толщины, дифференцированность нажима,
   локальные утолщения/ослабления
```

То есть наш canonical stroke graph должен быть не просто:

```json
{
  "nodes": [...],
  "edges": [...]
}
```

а примерно:

```json
{
  "nodes": [
    {
      "type": "endpoint | junction | crossing | loop_point | curvature_extreme",
      "x": 0.42,
      "y": 0.31,
      "confidence": 0.93
    }
  ],
  "edges": [
    {
      "type": "visible_stroke_segment",
      "curve": "bezier | polyline",
      "control_points": [...],
      "length": 24.1,
      "curvature": 0.37,
      "direction": "left_to_right | right_to_left | up | down | clockwise | counterclockwise | unknown",
      "width_mean": 2.8,
      "width_variance": 0.4,
      "smoothness": 0.82,
      "angularity": 0.16,
      "connection_type": "continuous | interval | probable_lift",
      "rhythm_score": 0.71
    }
  ]
}
```

Особенно важно: **мы не обязаны восстанавливать реальную траекторию пера**. Но мы можем восстанавливать признаки, которые являются следами двигательной организации письма.

---

# 3. Четыре фундаментальных свойства почерка

В курсе ключевыми свойствами почерка названы:

1. **индивидуальность**;
2. **динамическая устойчивость**;
3. **вариационность**;
4. **избирательная изменчивость**.  

Первые три прямо описаны как фундаментальные для идентификационных, диагностических и ситуационных исследований; индивидуальность связывается с неповторимым взаимодействием субъективных и объективных факторов формирования навыка. `Seregin_Pocherkovedenie_2015.pdf`

Для нашей модели это, пожалуй, самое важное место всего материала.

## 3.1. Индивидуальность

Почерк индивидуален не потому, что каждая буква всегда уникальна, а потому что **совокупность устойчивых отклонений, пропорций, соединений, точек начала/окончания, форм и ритмов** образует индивидуальный комплекс.

Для модели это значит: не надо искать “идеальную букву а” или “идеальную букву д”. Нужно искать:

```text
устойчивую систему отклонений от нормы
```

Например:

- как человек обычно начинает букву;
- где ставит точку соединения;
- как замыкает овал;
- как выполняет подстрочную петлю;
- насколько стабилен наклон;
- как меняется ширина штриха;
- какие соединения между буквами предпочитает.

## 3.2. Динамическая устойчивость

Очень важная мысль: устойчивость почерка не означает механическое повторение одинаковых форм. Почерк устойчив **динамически**, то есть сохраняет структуру навыка при допустимых колебаниях. В курсе прямо подчёркивается, что устойчивость проявляется через вариационность и разброс признака. `Seregin_Pocherkovedenie_2015.pdf`

Для модели это означает: нельзя сравнивать графы как точные шаблоны. Нужна вероятностная зона нормы.

Не так:

```text
эта буква отличается → ошибка
```

А так:

```text
эта буква отличается, но лежит внутри индивидуального диапазона вариации
```

## 3.3. Вариационность

Вариационность - это не шум, а нормальное свойство письменно-двигательной системы. Один и тот же человек может писать одну букву разными устойчивыми вариантами.

Для нас это означает, что модель должна хранить **мультимодальные варианты графов**:

```text
writer_style_model = {
    "letter д": [
        variant_1: looped descender,
        variant_2: simplified descender,
        variant_3: interval connection
    ],
    "letter т": [
        variant_1,
        variant_2
    ]
}
```

То есть canonical stroke graph должен поддерживать не один канон, а **семейство допустимых канонических вариантов**.

## 3.4. Избирательная изменчивость

Избирательная изменчивость - это реакция почерка на сбивающие факторы: усталость, необычную позу, алкоголь, болезнь, попытку маскировки, письмо другой рукой, плохое освещение и т. д. В материале подчёркивается, что конкретных “стопроцентных” признаков конкретного фактора нет, но есть комплексы признаков необычности выполнения рукописи. `Seregin_Pocherkovedenie_2015.pdf`

Для нашей модели это означает: нужно отделять:

```text
стабильные признаки автора
от
ситуативных деформаций
```

Иначе модель будет путать плохой скан, усталость, медленный темп или намеренное изменение с “особенностями автора”.

---

# 4. Очень важный принцип: общие и частные признаки

Почерковедение делит признаки на **общие** и **частные**.

Это прямо просится в архитектуру модели.

```text
общие признаки → глобальный уровень графа строки/страницы
частные признаки → локальный уровень букв, элементов, соединений
```

---

# 5. Общие признаки: что взять в модель

Общие признаки в курсе характеризуют рукопись в целом: пространственную ориентацию движений, степень сформированности письменно-двигательного навыка и структуру движений по траектории. `Seregin_Pocherkovedenie_2015.pdf`

Их можно превратить в глобальные признаки строки, слова или страницы.

## 5.1. Топографические признаки

Это признаки размещения текста:

- поля;
- красная строка;
- интервалы между строками;
- интервалы между словами;
- направление линии письма;
- форма линии письма;
- размещение знаков препинания;
- размещение фрагментов текста.

Для OCR они кажутся второстепенными, но для нашей работы важны. Почему? Потому что они дают уровень **layout-graph**.

Я бы ввёл несколько уровней графа:

```text
page graph
  └── line graph
       └── word graph
            └── letter/allograph graph
                 └── stroke graph
```

Топографические признаки живут на верхних уровнях:

```json
{
  "line_baseline": {
    "slope": 0.04,
    "curvature": 0.12,
    "waviness": 0.08
  },
  "word_spacing": {
    "mean": 14.3,
    "variance": 3.1
  },
  "line_spacing": {
    "mean": 42.0,
    "variance": 5.7
  }
}
```

## 5.2. Выработанность почерка

В курсе выработанность связывается с темпом и координацией движений. Высокая координация проявляется прямизной прямых элементов, округлостью овалов, равномерностью наклона, размера, разгона, нажима и устойчивым размещением точек начала/окончания/соединения; низкая - изломами, извилистостью, угловатостью, смещением точек и неравномерностью признаков. `Seregin_Pocherkovedenie_2015.pdf`

Для модели это можно сделать отдельной задачей:

```text
predict handwriting development / coordination score
```

Не обязательно как криминалистический вывод, а как latent quality factor.

Признаки:

```text
coordination_score =
    smoothness of strokes
    + stability of slant
    + stability of size
    + stability of spacing
    + endpoint consistency
    + connection consistency
    - angular noise
    - broken strokes
    - random tremor
```

## 5.3. Темп письма как наблюдаемый суррогат

У нас нет реальной скорости пера. Но из офлайн-скана можно извлечь косвенные признаки темпа:

Быстрый темп:

- высокая связность;
- упрощение букв;
- удлинённые заключительные штрихи;
- дугообразные переходы;
- уменьшение числа отрывов.

Медленный темп:

- малая связность;
- тупые начала и окончания;
- остановки;
- ретушь;
- угловатость;
- сильный недифференцированный нажим.

Это не “истинная скорость”, но **offline speed proxy**.

В графе это можно хранить так:

```json
{
  "tempo_proxy": {
    "connectivity": 0.78,
    "stroke_smoothness": 0.81,
    "terminal_bluntness": 0.12,
    "retouch_probability": 0.04,
    "unjustified_stops": 0.09
  }
}
```

## 5.4. Наклон

Наклон - это не просто deskew. В OCR часто исправляют наклон как дефект. Для нас наклон - признак.

Нужно различать:

```text
наклон страницы
наклон строки
наклон букв
наклон отдельных элементов
```

Поэтому модель должна нормализовать глобальный наклон изображения, но **не уничтожать локальный наклон почерка**.

## 5.5. Размер и разгон

Размер - вертикальная протяжённость. Разгон - горизонтальная протяжённость относительно высоты.

Для графа:

```json
{
  "height_ratio": 1.0,
  "width_height_ratio": 0.74,
  "upper_zone_ratio": 1.35,
  "lower_zone_ratio": 1.42
}
```

Особенно важно для кириллицы: подстрочные и надстрочные элементы в буквах **д, у, р, в, б, й** могут быть очень информативны.

## 5.6. Связность

Связность - один из центральных признаков для нашей темы. В курсе она определяется как степень непрерывности движений: чем реже пишущий прибор отрывается от бумаги, тем выше связность. `Seregin_Pocherkovedenie_2015.pdf`

Это почти напрямую соответствует графовой структуре:

```text
низкая связность → много компонент / probable pen lifts
высокая связность → длинные connected paths
средняя связность → локальные connected groups
```

Для нашей модели:

```json
{
  "connectivity": {
    "components_per_word": 2.3,
    "mean_connected_letters": 4.1,
    "probable_lifts": 3,
    "longest_connected_path": 0.64
  }
}
```

Связность нужно использовать не только как признак, но и как constraint при векторизации.

## 5.7. Нажим

Нажим в офлайн-скане напрямую не виден, особенно если это фото/скан шариковой ручки. Но можно использовать суррогаты:

- толщина штриха;
- интенсивность пикселей;
- локальное утолщение;
- различие сгибательных/разгибательных участков;
- степень дифференцированности ширины.

В курсе нажим определяется через толщину штрихов и различие сгибательных/разгибательных движений; также подчёркивается различие дифференцированного и недифференцированного нажима. `Seregin_Pocherkovedenie_2015.pdf`

Для модели:

```json
{
  "pressure_proxy": {
    "mean_width": 2.7,
    "width_std": 0.6,
    "downstroke_upstroke_ratio": 1.45,
    "differentiation": 0.52
  }
}
```

---

# 6. Частные признаки: основа stroke graph

Частные признаки - это самое полезное для нашей модели. Они описывают особенности движений при выполнении и соединении отдельных букв и элементов; именно по ним вместе с общими признаками делается вывод о наличии/отсутствии тождества. `Seregin_Pocherkovedenie_2015.pdf`

В курсе выделяются признаки:

- сложность движений;
- форма движений;
- направление движений;
- протяжённость по вертикали и горизонтали;
- количество движений;
- вид соединения;
- последовательность движений;
- относительное размещение точек начала, окончания, соединения, пересечения;
- относительное размещение элементов по вертикали и горизонтали.

Это фактически готовый словарь признаков для canonical stroke graph.

---

# 7. Как перевести частные признаки в машинное представление

## 7.1. Сложность движений

В экспертном смысле сложность - это упрощение или усложнение относительно прописей.

В нашей модели это можно представить как:

```text
complexity = number_of_strokes
           + number_of_curvature_extrema
           + number_of_loops
           + number_of_direction_changes
           + ornamentation_score
           - simplification_score
```

Для каждой буквы/аллографа:

```json
{
  "letter": "д",
  "complexity": {
    "movement_count": 4,
    "loop_count": 1,
    "curvature_extrema": 5,
    "simplified": false,
    "ornamental": true
  }
}
```

## 7.2. Форма движений

В почерковедении формы: прямолинейная, дуговая, угловатая, петлевая, овальная, извилистая, ломаная и т. д.

В графе:

```json
{
  "edge_shape": "arc | line | loop | oval | broken | wavy | angular",
  "curvature_profile": [...],
  "corner_count": 2,
  "loop_probability": 0.84
}
```

Особенно важно не просто классифицировать форму, а иметь **непрерывное числовое описание**:

```text
curvature mean
curvature variance
signed curvature
number of inflection points
angle change distribution
```

## 7.3. Направление движений

Для офлайн-скана истинное направление движения не всегда известно. Но можно хранить не “истинное направление”, а **допустимое/вероятное направление**.

```json
{
  "direction_hypotheses": [
    {"direction": "counterclockwise", "probability": 0.72},
    {"direction": "clockwise", "probability": 0.28}
  ]
}
```

Это лучше, чем жёстко утверждать, что модель восстановила движение пера.

## 7.4. Протяжённость

Для каждого элемента:

```json
{
  "vertical_extent": 12.4,
  "horizontal_extent": 5.8,
  "relative_to_x_height": 1.27,
  "relative_to_neighbor": 0.91
}
```

Для кириллицы это особенно важно, потому что подстрочные и надстрочные элементы несут много информации.

## 7.5. Количество движений

В курсе количество движений связывается с числом интервальных движений или резких изменений направления. Это можно формализовать:

```text
movement_count_proxy =
    connected_components
    + probable_lifts
    + sharp_turns
    + separate_elements
```

Для векторизации:

```json
{
  "movement_count_proxy": 3,
  "sharp_turns": 2,
  "probable_pen_lifts": 1
}
```

## 7.6. Вид соединения

Вид соединения - слитное или интервальное.

Для графа это центральный признак:

```json
{
  "connection": {
    "from": "letter_и",
    "to": "letter_м",
    "type": "continuous | interval | uncertain",
    "connector_shape": "arc | loop | angular | straight",
    "confidence": 0.87
  }
}
```

## 7.7. Последовательность движений

Это сложный момент: мы не хотим восстанавливать реальную траекторию пера. Но последовательность иногда может быть видна косвенно - по пересечениям, наложениям, утолщениям, разрывам, ретуши.

Поэтому я бы не делал обязательный `stroke_order`, а сделал:

```json
{
  "order_constraints": [
    {"edge_a": "before edge_b", "confidence": 0.61},
    {"edge_c": "unknown", "confidence": 0.20}
  ]
}
```

Это честнее.

## 7.8. Точки начала, окончания, соединения, пересечения

Это прямо совпадает с нашими nodes.

В старых кибернетических методах кодирования почерка в качестве исходных точек выделялись точки начала, окончания, соединения и пересечения движений. `Seregin_Pocherkovedenie_2015.pdf`

Это практически готовая основа для graph representation:

```json
{
  "node_types": [
    "start_candidate",
    "end_candidate",
    "junction",
    "crossing",
    "connection_point",
    "loop_closure",
    "baseline_touch",
    "upper_zone_extreme",
    "lower_zone_extreme"
  ]
}
```

Важно: точки начала/окончания в офлайн-изображении часто не наблюдаемы напрямую. Поэтому лучше использовать названия:

```text
start_candidate / end_candidate
```

а не уверенные `start` / `end`.

---

# 8. Принцип устойчивости: признак значим только если повторяется

Очень важная фраза в материале: частный признак становится характерным для почерка конкретного человека только тогда, когда он не просто выполнен с отклонением от прописей, но и устойчиво проявляется в рукописи. `Seregin_Pocherkovedenie_2015.pdf`

Для модели это критично.

Нельзя делать вывод по одному красивому признаку. Нужно считать частотность и устойчивость.

Для каждого признака:

```json
{
  "feature": "looped descender in д",
  "occurrences": 12,
  "possible_occurrences": 15,
  "stability": 0.80,
  "variance": 0.12,
  "writer_specificity": 0.73
}
```

Именно это может стать одним из научных вкладов:

> Мы не просто строим граф рукописи, а оцениваем устойчивость графовых признаков внутри текста.

---

# 9. Принцип “норма → отклонение → устойчивость”

В почерковедении признак часто определяется относительно нормы: школьных прописей, “родного” почерка, обычных условий письма или образцов.

Для нашей модели нужна похожая схема:

```text
script norm
    ↓
observed graph
    ↓
deviation graph
    ↓
stability / variation estimate
```

Для русского и английского это особенно полезно.

Можно иметь:

```text
canonical Cyrillic school-like graph prototypes
canonical Latin school-like graph prototypes
```

И затем модель будет измерять не только абсолютную форму, а отклонение:

```text
observed letter д - canonical д = writer/style deviation
```

Это даст более сильное представление, чем просто эмбеддинг изображения.

---

# 10. Изменённый почерк: как использовать в модели

В материале большой раздел посвящён изменённому почерку. Изменения делятся на естественные и умышленные: внешняя обстановка, внутреннее состояние, непривычная установка, маскировка, письмо другой рукой, подражание печатным буквам, подражание почерку другого лица и т. д. `Seregin_Pocherkovedenie_2015.pdf`

Для нашей задачи это важно по двум причинам.

Во-первых, модель должна быть устойчива к таким изменениям. Во-вторых, она может отдельно предсказывать “необычность исполнения”.

Классические признаки необычности:

- снижение темпа;
- снижение координации;
- интервальные соединения;
- увеличение размера;
- тупые начала и окончания;
- остановки;
- ретушь;
- повторы;
- сильный недифференцированный нажим;
- изломы и извилистость;
- неравномерный размер;
- неравномерный разгон;
- неустойчивый наклон;
- извилистая или ступенчатая линия письма;
- неравномерные интервалы. `Seregin_Pocherkovedenie_2015.pdf`

Это можно сделать отдельной головой модели:

```text
graph → abnormality head
```

Выход:

```json
{
  "execution_condition": {
    "ordinary": 0.62,
    "deautomatized": 0.28,
    "possible_disguise": 0.07,
    "possible_unusual_conditions": 0.03
  },
  "deautomation_features": {
    "tempo_drop": 0.41,
    "coordination_drop": 0.37,
    "retouch": 0.08,
    "blunt_terminals": 0.19,
    "spacing_instability": 0.22
  }
}
```

Это не обязательно делать как судебный диагноз. Для научной работы это можно назвать:

```text
execution stability estimation
```

или

```text
motor regularity estimation
```

---

# 11. Диагностические, ситуационные и идентификационные задачи

Курс делит задачи на идентификационные, диагностические и ситуационные. Идентификационные ищут тождество исполнителя, диагностические определяют свойства/состояния, ситуационные - условия и обстоятельства выполнения рукописи. `Seregin_Pocherkovedenie_2015.pdf`

Для нашей модели это можно превратить в multi-task learning:

```text
main task:
    text recognition

auxiliary tasks:
    stroke graph reconstruction
    writer/style embedding
    stability/variation estimation
    execution condition estimation
    script/language classification
    layout/topographic feature prediction
```

То есть модель обучается не только читать, но и понимать структуру почерка.

---

# 12. Принцип математического моделирования

В курсе подчёркивается, что математические модели в почерковедении должны:

- адекватно отражать свойства почерка и элементы его структуры;
- объективизировать исследование;
- быть несложными в применении;
- соответствовать методике исследования;
- давать объективно оцениваемые показатели. `Seregin_Pocherkovedenie_2015.pdf`

Это почти методологические требования к нашей модели.

Для нашей статьи я бы сформулировал так:

> Модель canonical stroke graph должна быть не просто нейросетевым latent representation, а измеримым структурным представлением, в котором каждый узел, ребро и атрибут имеют интерпретируемый смысл.

То есть плохая идея:

```text
взяли ViT embedding и сказали, что это “структурное представление”
```

Хорошая идея:

```text
извлекли граф с узлами, рёбрами, кривыми, петлями, точками соединения,
толщинами, наклонами, связностью, вариативностью и confidence-score
```

---

# 13. Очень ценная идея: графическое усреднение и область разброса

В материале описан метод графического усреднения: по одноимённым буквам получают усреднённые формы движений, изучают плотность распределения, устойчивые и подверженные помехам участки знака, размер, форму и площадь разброса. `Seregin_Pocherkovedenie_2015.pdf`

Это прямо можно модернизировать.

Для нашей работы:

```text
Graphical averaging → Neural graph prototype learning
```

Идея:

1. Берём все варианты буквы “д” у одного автора.
2. Строим для каждой canonical stroke graph.
3. Выравниваем графы.
4. Получаем:
   - средний граф;
   - зону допустимого разброса;
   - устойчивые узлы;
   - нестабильные узлы;
   - вариативные рёбра;
   - writer-specific признаки.

Это может стать очень сильной частью работы.

Например:

```json
{
  "writer_prototype": {
    "letter": "д",
    "stable_nodes": ["upper_entry", "oval_closure", "descender_loop"],
    "variable_nodes": ["terminal_tail"],
    "mean_graph": "...",
    "variance_field": "...",
    "topological_variants": 2
  }
}
```

Это даст мост между почерковедением и машинным обучением.

---

# 14. Как это меняет нашу исходную тему

Изначально тема была:

> структурная векторизация / графовое штриховое представление рукописного текста

После изучения почерковедения я бы усилил тему так:

> **Почерковедчески-информированная графово-векторная реконструкция офлайн-рукописей для русско-английского HTR.**

Или на английском:

> **Forensic-Handwriting-Informed Canonical Stroke Graph Vectorization for Offline Russian-English Handwritten Text Recognition.**

Это звучит сильнее, чем просто “stroke graph vectorization”, потому что мы добавляем фундаментальную теорию признаков почерка.

---

# 15. Что именно должно войти в canonical stroke graph

Я бы разделил признаки на 5 уровней.

## Уровень 1. Геометрия штриха

```text
edge geometry:
    length
    width
    curvature
    signed curvature
    smoothness
    angularity
    direction
    bezier control points
    polyline points
```

## Уровень 2. Топология

```text
graph topology:
    endpoints
    junctions
    crossings
    loops
    connected components
    cycles
    bridges
    graph degree distribution
    loop closure quality
```

## Уровень 3. Письменно-двигательные признаки

```text
motor proxies:
    connectivity
    probable pen lifts
    terminal bluntness
    retouch probability
    tremor / waviness
    pressure proxy
    rhythm regularity
    coordination score
```

## Уровень 4. Буквенно-элементные признаки

```text
letter/allograph features:
    upper zone elements
    middle zone elements
    lower zone elements
    relative element placement
    start/end/connection/crossing candidates
    loop shape
    descender shape
    ascender shape
    connector type
```

## Уровень 5. Текстово-топографические признаки

```text
layout features:
    baseline slope
    baseline curvature
    line spacing
    word spacing
    margins
    punctuation placement
    paragraph indentation
```

---

# 16. Архитектура модели, которую я теперь считаю лучшей

Я бы делал не один модуль, а систему:

```text
image
  ↓
Visual Encoder
  ↓
Dense Structural Heads
  ├── skeleton map
  ├── stroke width map
  ├── endpoint heatmap
  ├── junction heatmap
  ├── crossing heatmap
  ├── loop heatmap
  ├── tangent field
  └── baseline / x-height zones
  ↓
Graph Builder
  ↓
Canonical Stroke Graph
  ↓
Graph Encoder / Graph Transformer
  ├── HTR decoder
  ├── writer/style embedding
  ├── stability/variation head
  ├── abnormality/deautomation head
  └── reconstruction renderer
```

Особенно важно добавить `x-height zones`:

```text
upper zone
middle zone
lower zone
```

Потому что в почерковедении очень много признаков связано с размещением элементов относительно линии письма, срединной линии, верхней линии строки, подстрочных и надстрочных зон.

---

# 17. Loss-функции

Для такой модели нужны не только OCR-loss.

```text
L = L_text
  + L_skeleton
  + L_keypoints
  + L_width
  + L_tangent
  + L_graph_topology
  + L_vector_reconstruction
  + L_stability
  + L_style_contrastive
```

## 17.1. Text loss

```text
CTC / seq2seq cross entropy
```

## 17.2. Skeleton loss

Сравнение предсказанного скелета с pseudo-label от классической скелетизации.

## 17.3. Keypoint loss

Endpoint, junction, crossing, loop-point detection.

## 17.4. Width/pressure proxy loss

Модель должна предсказывать ширину штриха вдоль рёбер.

## 17.5. Topology loss

Штраф за потерянные петли, лишние компоненты, разорванные соединения.

## 17.6. Vector reconstruction loss

Предсказанный граф рендерится обратно в растр, затем сравнивается с исходным изображением.

## 17.7. Stability loss

Для нескольких образцов одного автора/одной буквы модель должна учиться разделять:

```text
устойчивое ядро
и
вариативные проявления
```

## 17.8. Style contrastive loss

Графовые признаки одного автора должны быть ближе друг к другу, чем признаки разных авторов, но без разрушения распознавания текста.

---

# 18. Самая интересная научная новизна

После изучения почерковедения я бы выбрал не просто “векторизацию”, а такую постановку:

> **Модель строит canonical stroke graph и одновременно оценивает для каждого признака его устойчивость, вариативность и диагностическую чувствительность.**

То есть для каждого признака мы храним:

```json
{
  "feature": "loop closure in д",
  "value": "...",
  "stability": 0.84,
  "variation": 0.12,
  "writer_specificity": 0.77,
  "condition_sensitivity": 0.31,
  "recognition_importance": 0.69
}
```

Это уже не просто OCR. Это структурная модель почерка.

---

# 19. Как использовать русско-английский аспект

Почерковедческая теория в материале в основном ориентирована на русский рукописный текст. Но многие принципы универсальны:

```text
форма движений
направление
связность
размер
разгон
наклон
нажим
петли
точки соединения
точки пересечения
устойчивость
вариативность
```

Значит, для RU/EN можно сделать двухуровневую систему:

```text
universal motor-graph layer
+
script-specific allograph layer
```

То есть:

```text
stroke graph primitives - общие для русского и английского
letter prototypes - разные для кириллицы и латиницы
```

Это красиво для статьи:

> Мы проверяем, какие графово-двигательные признаки переносятся между кириллицей и латиницей, а какие являются script-specific.

---

# 20. Что можно заявить как вклад статьи

Я бы сформулировал 5 вкладов.

## Вклад 1. Почерковедчески-информированное представление

Новый формат:

```text
Canonical Stroke Graph with handwriting-informed attributes
```

Он включает не только кривые и узлы, но и признаки формы, связности, протяжённости, нажима, координации, вариативности.

## Вклад 2. Модель weak/self-supervised vectorization

Так как stroke-разметки нет, используем:

```text
skeleton pseudo-labels
+ differentiable rendering
+ synthetic online-to-offline data
+ small manually corrected subset
```

## Вклад 3. Устойчивость и вариативность признаков

Вводим оценку:

```text
feature stability
feature variance
writer specificity
condition sensitivity
```

Это напрямую вдохновлено почерковедческим принципом динамической устойчивости.

## Вклад 4. Multi-task graph model

Модель решает сразу:

```text
text recognition
stroke graph reconstruction
style embedding
stability estimation
execution abnormality estimation
```

## Вклад 5. RU/EN cross-script benchmark

Проверяем, какие stroke graph features универсальны, а какие зависят от письменности.

---

# 21. Возможные эксперименты

## Эксперимент 1. Качество векторизации

Сравнить:

```text
Potrace / contour vectorization
skeleton graph baseline
Bezier fitting baseline
our handwriting-informed CSG
```

Метрики:

```text
raster reconstruction IoU
skeleton Chamfer distance
endpoint F1
junction F1
crossing F1
loop preservation
component consistency
Bezier fitting error
```

## Эксперимент 2. Помогает ли CSG распознаванию

Сравнить:

```text
image-only HTR
graph-only HTR
image + graph HTR
image + graph + handwriting-informed features
```

Метрики:

```text
CER
WER
robustness under blur/noise
writer-independent split
cross-script transfer
```

## Эксперимент 3. Устойчивость признаков

Для авторов, где есть несколько строк/страниц:

```text
same writer intra-variation
different writer inter-variation
```

Метрики:

```text
same-writer graph distance
different-writer graph distance
stability of allograph prototypes
writer retrieval accuracy
```

## Эксперимент 4. Необычность исполнения

Синтетически или вручную создать искажения:

```text
blur
uneven pressure
broken strokes
slant change
spacing instability
simulated tremor
reduced connectivity
```

Проверить, может ли модель отличать:

```text
нормальную вариативность
от
структурной деградации / деавтоматизации
```

## Эксперимент 5. Cross-script transfer

```text
train EN → test RU
train RU → test EN
train RU+EN → test historical RU
```

Гипотеза:

> Графово-двигательные признаки лучше переносятся между письменностями, чем чисто растровые признаки.

---

# 22. Что особенно пригодится из курса для раздела “Related Work / Theory”

Я бы взял из Серегина следующие теоретические опоры:

1. Почерк - результат письменно-двигательного навыка, а не просто изображение.
2. Графические навыки включают конфигурацию, ориентацию, координацию, скорость, ритм, нажим.
3. Почерк индивидуален, но динамически устойчив.
4. Устойчивость проявляется через вариативность, а не через абсолютную одинаковость.
5. Признаки делятся на общие и частные.
6. Общие признаки описывают рукопись в целом; частные - элементы букв и соединения.
7. Информативность признака зависит от его устойчивости.
8. Математическая модель должна быть объективной, измеримой и интерпретируемой.
9. Нельзя игнорировать необычные условия письма.
10. Нужно различать идентификационные, диагностические и ситуационные аспекты.

---

# 23. Как я бы теперь сформулировал итоговую тему

Лучший вариант:

> **Почерковедчески-информированная каноническая графово-векторная реконструкция штрихов для офлайн-распознавания русско-английских рукописей.**

На английском:

> **Forensic-Handwriting-Informed Canonical Stroke Graph Vectorization for Offline Russian-English Handwritten Text Recognition.**

Более научно и компактно:

> **Canonical Stroke Graphs with Stability-Aware Handwriting Features for Offline Russian-English HTR.**

---

# 24. Самая сильная исследовательская гипотеза

Я бы сделал главную гипотезу такой:

> **Если представлять офлайн-рукопись как canonical stroke graph с признаками формы, связности, протяжённости, нажима, координации и устойчивости, то такое представление будет лучше переноситься между авторами, условиями письма и письменностями, чем чисто растровые признаки.**

Дополнительные гипотезы:

```text
H1: графовые признаки улучшают OCR на шумных и малоресурсных данных.

H2: учёт устойчивости/вариативности признаков улучшает writer-independent recognition.

H3: часть признаков является script-independent и переносится между кириллицей и латиницей.

H4: признаки деавтоматизации помогают отделять нормальную вариативность от искажений.
```

---

# 25. Главная практическая архитектура, которую стоит реализовывать

Я бы выбрал такую:

```text
Stage 1. Classical pseudo-label generator
image → binarization → skeleton → graph → keypoints → Bezier edges

Stage 2. Neural graph vectorizer
image → skeleton/keypoint/width/tangent maps → graph decoder → CSG

Stage 3. Handwriting-informed graph encoder
CSG → Graph Transformer → text + style + stability

Stage 4. Raster-graph fusion
image encoder + graph encoder → HTR decoder

Stage 5. Evaluation
CER/WER + structural metrics + stability metrics
```

Это достаточно сложная, интересная и научно защищаемая работа.

---

# 26. Финальный вывод

Материал по почерковедению сильно усиливает нашу идею. Он показывает, что рукопись нужно понимать как **материальный след устойчивого, вариативного, иерархически организованного двигательного навыка**. Поэтому наша модель должна быть не просто “векторизатором штрихов”, а **структурной моделью признаков почерка**.

Самая перспективная формула работы:

> **строим canonical stroke graph, в котором каждая линия, петля, соединение, точка пересечения, ширина, наклон, связность и вариация имеют интерпретируемый почерковедческий смысл; затем используем этот граф для распознавания текста и анализа устойчивости рукописи.**

Это как раз та тема, которая будет **интересной, непростой и отличимой от обычного HTR/OCR**.

## Prompt:
23/05/2026, 16:16:49

теперь следующая литература - также изучи и дополни всё из прошлой, мы прямо сейчас создаем теоретическое дополнение и идеи для реализации модели

[Attachment: Zhizhina_M_V_Pocherkovedenie_2006.docx]

## Response:
23/05/2026, 16:16:52

Я уже вижу, что Жижина полезна не только как “ещё один учебник”, а как методологическая рамка: объект, задача, признаки, сопоставимость образцов, диагностические факторы. Сейчас упакую это в теоретическое дополнение к нашей модели.

> **Considering how to handle docx content**
>
> I’m thinking about extracting full content from a docx file. I could use Python for that, but would it provide enough detail, like headings or sections, to be useful? The snippets I’ve gotten are often too short, and I need a more targeted approach, especially if the user's asking for details on specific topics. Hmm, looks like I'll have to familiarize myself with processing docx files before proceeding.
>
> **Reading documentation for docx tasks**
>
>
> **Planning to extract docx content with Python**
>
> I’m thinking of using Python to extract text from the docx file, maybe focusing on specific sections around relevant headings. No need for citations on the Python usage itself. I’ll probably work with the python-docx library—seems like it’s mostly set up. Time to check the paragraphs and organize the content!
>
> **Extracting paragraphs from DOCX**
>
>
> **Extracting specific paragraphs by index**
>
> I’m thinking of extracting the relevant paragraphs from the docx file by using their indices. Looks like I can use search indices to pinpoint exactly what I need from the content. This should help focus on the most important parts without digging through everything.
>
> **Extracting relevant paragraphs by index**
>
>
> **Extracting text around specified indices**
>
> I’m thinking of extracting text specifically from around indices 240-430 and 660-690. I can use Python to pull that content out and ensure I'm working with just the relevant parts. This should help narrow things down to exactly what's needed without wasting time on irrelevant sections.
>
> **Extracting text from specified indices**
>
>
> **Extracting full content from specific range**
>
> I noticed the output is showing ellipses because it's too long. But it still looks like it captures enough. I'll focus on extracting the full content from around indices 270-320 to make sure I get all the necessary details. This should give me exactly what I need without cutting anything off.
>
> **Extracting specific text ranges from document**
>
>
> **Filling in missing content**
>
> It looks like the output missed sections from 301-320 and 353-410, though we still have enough for now. To make sure we don't lose anything important, I’ll focus on extracting more content around indices 455-460. That should give me a complete picture of what’s missing.
>
> **Extracting specific paragraph range sections**
>
>
> **Focusing on specific content extraction**
>
> I think I should extract the full content from indices 320-352 and 358-405. This will help ensure that I cover all the necessary sections without missing any important details. I’ll focus on these ranges to make sure everything is included properly.
>
> **Extracting specific paragraphs from document**
>
>
> **Addressing citations and content complexity**
>
> I may struggle to use citations from Python directly, but I can reference snippets from the file search results instead, like turn8file0, which includes content up to chapter 1, and other markers from the docx file. I'll mention Jizhina's work supplementing Seregin’s and clarify the difference between graphology and forensic analysis while keeping the answer comprehensive, but concise.
>
> **Refining search for specific terms**
>
> I should refine my file search by focusing more precisely on terms like "точные науки" (exact sciences), "моделирование" (modeling), and "сопоставимость" (comparability). This will help me gather more relevant content for the task and improve the accuracy of the information I provide.
>
> **Finalizing response with citations**
>
> I'll craft a concise yet detailed response, ensuring it’s well-structured with citations. Since the user has uploaded docs, no internet connections are needed, but everything will be cited from the file search.
> 
> The content includes:
> - Zhizhina's complement to Seregin’s methodology, focusing on differences between properties and features. We also clarify the importance of structured tests, multi-head feature weighting, and a complete sign hierarchy for a coherent classification and diagnosis model.
>
> Thought for 1m 58s

Да. Жижина очень хорошо дополняет Серегина: Серегин дал нам **понятийную и признаковую систему почерка**, а Жижина сильнее подчёркивает **методологию экспертного исследования**: объект, задача, сопоставимость материалов, информативность признаков, роль условий выполнения, требования к образцам и логику перехода от наблюдаемого признака к скрытому свойству письменно-двигательного навыка.

Ниже - уже не просто конспект, а теоретическое дополнение к нашей будущей модели.

---

# 1. Что Жижина добавляет к прошлой теории

Главное усиление: **признак почерка - это не просто видимая форма, а проявление свойства письменно-двигательного навыка**.

Это критично для нашей модели. Мы не должны строить граф как набор случайных геометрических дескрипторов. Каждый node, edge, loop, connector, width-profile или baseline должен быть связан с некоторым более глубоким свойством:

```text
наблюдаемый графовый признак
        ↓
проявление свойства письменно-двигательного ФДК
        ↓
использование для HTR / writer style / stability / diagnostic robustness
```

У Жижиной судебное почерковедение определяется как система знаний о закономерностях почерка и методах его исследования; предмет включает изучение формирования, функционирования и изменения почерка, а также лежащего в его основе письменно-двигательного функционально-динамического комплекса навыков. `Zhizhina_M_V_Pocherkovedenie_2006.docx`

Это значит: наша модель должна быть **не просто vectorizer**, а **модель отображения скрытого письменно-двигательного навыка в наблюдаемую графовую структуру**.

---

# 2. Уточнённая теоретическая основа модели

После Серегина и Жижиной я бы формулировал основу так:

> Рукописное изображение является материальным отображением письменно-двигательного функционально-динамического комплекса. Поэтому задача структурной векторизации должна состоять не только в восстановлении скелета или кривых, а в извлечении наблюдаемых графовых признаков, которые выражают устойчивые, вариативные и ситуативно изменяемые свойства почерка.

Серегин описывает три потока информации в рукописи: структурно-геометрический, координационно-скоростно-ритмический и нажимной. `Seregin_Pocherkovedenie_2015.pdf` Жижина фактически подтверждает этот подход через связку естественных наук, математики, теории распознавания образов, вероятностного анализа и криминалистической методики. `Zhizhina_M_V_Pocherkovedenie_2006.docx`

Для нас это превращается в три ветви признаков:

```text
1. Structural-geometric stream
   форма, петли, соединения, пересечения, наклон, размер, разгон

2. Coordination-rhythm stream
   плавность, связность, равномерность, темп-прокси, координация

3. Pressure-width stream
   ширина штриха, плотность, вариация толщины, нажим-прокси
```

---

# 3. Самая важная идея Жижиной: свойство ≠ признак

Жижина подчёркивает различие между свойством и признаком: свойство относится к более глубокой характеристике объекта, а признак - это его конкретное проявление в рукописи. Для нашей модели это фундаментально.

Плохой путь:

```text
модель нашла много графовых чисел → называем это признаками
```

Правильный путь:

```text
модель извлекает наблюдаемые признаки,
каждый из которых связан с конкретным свойством почерка
```

Например:

```text
Свойство: степень сформированности письменно-двигательного навыка
Признаки:
    темп-прокси
    координация
    равномерность наклона
    равномерность размера
    степень связности
    стабильность точек соединения

Свойство: динамика нажима
Признаки:
    средняя ширина штриха
    вариация ширины
    различие сгибательных/разгибательных участков
    локальные утолщения
```

Отсюда я предлагаю ввести в модель слой:

```text
Observable Graph Features → Latent Handwriting Properties
```

То есть графовые признаки не просто подаются в HTR-декодер, а группируются в интерпретируемые скрытые свойства.

---

# 4. Уточнённая структура canonical stroke graph

Раньше мы говорили:

```text
nodes + edges + Bezier curves + topology
```

Теперь нужно расширить до:

```text
CSG = Canonical Stroke Graph
    + Handwriting Property Layer
    + Stability/Variation Layer
    + Condition Sensitivity Layer
```

Предлагаю такую схему:

```json
{
  "stroke_graph": {
    "nodes": [],
    "edges": [],
    "loops": [],
    "components": []
  },
  "observable_features": {
    "structural_geometric": {},
    "coordination_rhythm": {},
    "pressure_width": {},
    "topographic": {},
    "letter_element": {}
  },
  "latent_properties": {
    "development_level": {},
    "coordination_level": {},
    "tempo_proxy": {},
    "connectivity": {},
    "movement_complexity": {},
    "pressure_proxy": {},
    "layout_orientation": {}
  },
  "stability_model": {
    "within_writer_variance": {},
    "feature_repeatability": {},
    "variant_families": {}
  },
  "diagnostic_context": {
    "possible_deautomation": {},
    "condition_sensitivity": {},
    "uncertainty": {}
  }
}
```

---

# 5. Новая важная категория: “особенные признаки”

У Жижиной есть сильная идея, которой я раньше уделил мало внимания: кроме общих и частных признаков есть промежуточная группа - **особенные признаки**. Это признаки, которые проявляются не в одной конкретной букве, а в группах разноимённых букв с общей конструкцией: например, овальные элементы в “а”, “о”, “ю”, “ф”; надстрочные элементы в “б”, “й”; подстрочные элементы в “д”, “у”.

Это идеально для нашей модели.

Я предлагаю добавить уровень:

```text
Allograph-block graph
```

То есть модель должна выделять не только буквы, но и повторяющиеся графовые блоки:

```text
oval block:
    а, о, ю, ф

upper loop / upper element block:
    б, й, в некоторых вариантах л/т

lower descender block:
    д, у, р, ц, щ

connector block:
    и-н-п-м group
```

Это особенно ценно для русского, потому что многие кириллические буквы состоят из повторяющихся моторно-графических блоков.

---

# 6. Как это использовать в архитектуре

Архитектуру я бы теперь изменил так:

```text
Image
  ↓
Visual Encoder
  ↓
Dense Structural Prediction
  ├── skeleton map
  ├── stroke width map
  ├── endpoint map
  ├── junction/crossing map
  ├── loop map
  ├── tangent field
  ├── baseline / x-height / zones
  └── ink density map
  ↓
Graph Builder
  ↓
Canonical Stroke Graph
  ↓
Handwriting Feature Extractor
  ├── general features
  ├── special block features
  ├── private letter-element features
  ├── stability/variation features
  └── diagnostic/context features
  ↓
Graph Transformer + Raster Fusion
  ├── HTR decoder
  ├── writer/style embedding
  ├── feature stability head
  ├── condition/deautomation head
  └── uncertainty/informativeness head
```

Это уже намного сильнее, чем просто “скелет → граф → OCR”.

---

# 7. Признаки, которые нужно явно добавить в модель

## 7.1. Общие признаки

Общие признаки описывают рукопись в целом. У Серегина они делятся на признаки пространственной ориентации, сформированности письменно-двигательного навыка и структуры движений. `Seregin_Pocherkovedenie_2015.pdf`

Для модели:

```json
{
  "general_features": {
    "baseline_slope": 0.03,
    "baseline_waviness": 0.12,
    "word_spacing_mean": 14.2,
    "line_spacing_mean": 41.7,
    "slant_mean": 72.4,
    "slant_variance": 5.1,
    "size_mean": 3.4,
    "size_variance": 0.6,
    "expansion_ratio": 0.82,
    "connectivity_degree": 0.71,
    "coordination_score": 0.84,
    "tempo_proxy": 0.68,
    "pressure_proxy": 0.52
  }
}
```

## 7.2. Частные признаки

Частные признаки описывают конкретные буквы, элементы, соединения, точки начала/окончания/пересечения, форму и направление движений.

Для модели:

```json
{
  "private_features": {
    "letter": "д",
    "construction": "looped_descender",
    "connection_type": "continuous",
    "loop_shape": "oval",
    "lower_zone_extent": 1.42,
    "start_candidate": [x, y],
    "end_candidate": [x, y],
    "junctions": [],
    "crossings": [],
    "relative_placement": {}
  }
}
```

## 7.3. Особенные признаки

Новая важная категория:

```json
{
  "special_block_features": {
    "oval_group": {
      "letters": ["а", "о", "ю", "ф"],
      "closure_position": "upper_right",
      "loop_roundness": 0.77,
      "closure_stability": 0.82
    },
    "descender_group": {
      "letters": ["д", "у", "р"],
      "descender_depth": 1.36,
      "looped": true,
      "terminal_direction": "rightward"
    }
  }
}
```

Это даст модели возможность учить **общие моторные блоки**, а не только отдельные буквы.

---

# 8. Новая идея: task-dependent информативность признаков

У Жижиной важная мысль: признак может быть идентификационным, диагностическим или классификационно-диагностическим в зависимости от задачи. Один и тот же наблюдаемый признак может иметь разную значимость для разных целей.

Для нас это означает: нельзя иметь один общий feature importance. Нужна матрица:

```text
feature × task
```

Например:

| Признак | Для HTR | Для writer style | Для conditions | Для vector quality |
|---|---:|---:|---:|---:|
| форма петли “д” | высокая | высокая | средняя | высокая |
| baseline slope | низкая | средняя | высокая | средняя |
| ширина штриха | средняя | средняя | высокая | высокая |
| точка соединения букв | высокая | высокая | средняя | высокая |
| интервалы между словами | средняя | средняя | высокая | низкая |

Это можно реализовать через attention-gating:

```text
feature embedding
   ↓
task-specific attention
   ↓
HTR / writer / diagnostic / vectorization outputs
```

---

# 9. Новая идея: редкость признака + устойчивость признака

Из Жижиной следует важный принцип: для идентификационной значимости признак должен быть:

```text
редким между разными людьми
и
устойчивым внутри одного человека
```

Для нашей модели это можно сделать математически:

```text
feature_value_score =
    intra_writer_stability
    × inter_writer_rarity
    × extraction_confidence
```

То есть:

```json
{
  "feature": "loop closure in д",
  "intra_writer_stability": 0.84,
  "inter_writer_rarity": 0.61,
  "extraction_confidence": 0.92,
  "task_importance": {
    "htr": 0.67,
    "writer_style": 0.89,
    "condition": 0.31
  }
}
```

Это можно сделать одним из самых сильных вкладов статьи:

> Мы вводим stability-aware and rarity-aware stroke graph features.

---

# 10. Новая идея: оценка информативности входа

Жижина подробно различает большие тексты, средние, малые, краткие записи и подписи. Смысл: не вся рукопись одинаково информативна.

Для нашей модели это очень важно. Она должна не только выдавать текст, но и оценивать:

```text
насколько достаточно графовой информации для уверенного вывода
```

Добавляем модуль:

```text
Informativeness Estimator
```

Он считает:

```json
{
  "informativeness": {
    "text_volume": "medium",
    "alphabet_coverage": 0.62,
    "repeated_letters": {
      "а": 12,
      "о": 9,
      "д": 3
    },
    "stroke_graph_density": 0.74,
    "feature_repetition": 0.69,
    "expected_reliability": 0.81
  }
}
```

Для HTR это поможет калибровать confidence. Для научной части это особенно красиво: модель не просто ошибается молча, а понимает, что у неё мало структурной информации.

---

# 11. Новая идея: сопоставимость образцов как принцип обучения

У Жижиной много внимания уделено подготовке сравнительных материалов: свободные и экспериментальные образцы, достоверность, сопоставимость по языку, алфавиту, материалу письма, темпу, размеру, наклону, пишущему прибору и условиям. Также подчёркивается, что экспериментальные образцы полезно получать в привычном, медленном и быстром темпе, потому что темп сильно влияет на вариационность почерка. `Zhizhina_M_V_Pocherkovedenie_2006.docx`

Для нашей работы это превращается в протокол создания датасета.

Нужны не просто сканы разных авторов, а controlled handwriting protocol:

```text
Для каждого автора:
1. обычный текст в привычном темпе
2. тот же или похожий текст в медленном темпе
3. текст в быстром темпе
4. текст разными пишущими приборами
5. текст на линованной/нелинованной бумаге
6. короткая запись
7. подпись / подпись-подобная запись, если это этически и юридически допустимо
```

Тогда мы сможем учить модель различать:

```text
устойчивые признаки автора
vs
вариации из-за темпа / инструмента / условий
```

---

# 12. Новая идея: экспериментальные режимы как аугментации

Вместо обычных computer vision аугментаций типа blur/noise/crop можно делать **почерковедчески мотивированные аугментации**.

Например:

```text
tempo augmentation:
    увеличиваем связность
    сокращаем элементы
    сглаживаем кривые
    удлиняем финальные штрихи

slow-writing augmentation:
    уменьшаем связность
    добавляем тупые начала/окончания
    усиливаем локальные утолщения
    добавляем мелкие остановки

coordination degradation:
    добавляем неровность наклона
    волнистость прямых
    угловатость овалов
    нестабильность baseline

pressure variation:
    меняем width profile
    меняем ink density
    делаем нажим более/менее дифференцированным
```

Это гораздо осмысленнее, чем случайная деформация изображения.

---

# 13. Уточнение по нажиму

Нажим - опасный признак для офлайн-модели. Его нельзя напрямую восстановить из скана, особенно если это фото, копия или изображение с неизвестным пишущим прибором.

Но можно использовать **pressure proxy**:

```text
pressure_proxy =
    stroke width
    + ink density
    + width variation
    + local thickening
    + contrast distribution
```

Серегин указывает, что быстрый темп связан с постепенным утолщением начальных и утончением заключительных штрихов, а медленный - с тупыми началами/окончаниями, извилистостью и угловатостью. `Seregin_Pocherkovedenie_2015.pdf` Жижина уточняет, что нажим опосредованно проявляется через ширину штриха и плотность красителя, но зависит от подложки и условий письма, поэтому его нужно использовать осторожно.

В модели это должно быть не поле `pressure`, а именно:

```json
"pressure_proxy": {
  "mean_width": 2.8,
  "width_variance": 0.41,
  "ink_density_mean": 0.63,
  "differentiation_score": 0.52,
  "reliability": 0.44
}
```

Ключевое слово - `reliability`.

---

# 14. Уточнение по темпу

Настоящий темп письма из скана мы не знаем. Но можно извлечь **tempo proxy**.

Признаки быстрого темпа:

```text
большая связность
упрощение букв
сокращение элементов
плавные переходы
удлинённые заключительные штрихи
меньше отрывов
```

Признаки замедленного темпа:

```text
отрывистость
тупые начала и окончания
локальные утолщения
остановки
ретушь
угловатость
недифференцированный нажим
```

В модели:

```json
"tempo_proxy": {
  "connectivity": 0.78,
  "simplification": 0.61,
  "terminal_bluntness": 0.18,
  "local_stops": 0.12,
  "retouch_probability": 0.05,
  "confidence": 0.73
}
```

---

# 15. Уточнение по связности

Связность теперь становится одним из центральных графовых признаков.

Серегин определяет связность как степень непрерывности движений; чем реже отрывается пишущий прибор, тем выше связность. Там же выделяются уровни от отрывистого письма до сплошной связности. `Seregin_Pocherkovedenie_2015.pdf`

В графе:

```json
"connectivity": {
  "components_per_word": 1.7,
  "mean_connected_letters": 4.3,
  "longest_connected_component_ratio": 0.76,
  "probable_lifts": 3,
  "interval_connections": 2,
  "continuous_connections": 11
}
```

Важно: связность должна быть не только признаком, но и **регуляризатором графа**. Например, если модель делает слишком много разрывов в слове, это может быть ошибкой векторизации.

---

# 16. Уточнение по изменённому почерку

Серегин делит изменения почерка на естественные и умышленные: внешняя обстановка, внутреннее состояние, непривычная установка, скорописная маскировка, смена пишущей руки, подражание печатным буквам, подражание чужому почерку и другие формы. `Seregin_Pocherkovedenie_2015.pdf`

Для нашей модели это даёт отдельный исследовательский слой:

```text
Execution Condition Modeling
```

Мы не должны делать судебные выводы вроде “человек был пьян” или “маскировал почерк”. Но мы можем предсказывать:

```text
ordinary execution
deautomatized execution
possible unusual condition
possible intentional distortion
low confidence
```

Для HTR это полезно: изменённый почерк часто снижает точность распознавания. Для модели это может быть сигналом, что нужно больше опираться на raster branch или снижать confidence.

---

# 17. Уточнённая архитектура loss-функций

Теперь loss должен быть не только OCR и reconstruction.

```text
L = L_text
  + L_skeleton
  + L_keypoint
  + L_width
  + L_tangent
  + L_vector_render
  + L_general_features
  + L_private_features
  + L_special_block_features
  + L_stability
  + L_variation
  + L_condition
  + L_informativeness
```

Расшифровка:

```text
L_text:
    CTC / seq2seq loss

L_skeleton:
    совпадение со skeleton pseudo-label

L_keypoint:
    endpoints, junctions, crossings, loop closures

L_width:
    stroke width map

L_tangent:
    локальное направление штриха

L_vector_render:
    граф рендерится обратно в растр

L_general_features:
    наклон, размер, разгон, связность, baseline, spacing

L_private_features:
    признаки букв и элементов

L_special_block_features:
    овалы, подстрочные блоки, надстрочные блоки, connector groups

L_stability:
    один автор / близкие варианты должны иметь близкие устойчивые признаки

L_variation:
    разные темпы и условия должны сохранять ядро, но менять вариативные признаки

L_condition:
    ordinary vs altered/degraded writing proxy

L_informativeness:
    оценка достаточности структурной информации
```

---

# 18. Новая модель данных

Я бы ввёл такой формат аннотации:

```json
{
  "image_id": "sample_001",
  "script": "cyrillic",
  "language": "ru",
  "writer_id": "w_023",
  "text": "пример рукописного текста",
  "metadata": {
    "tempo_condition": "normal | slow | fast | unknown",
    "instrument": "ballpoint | gel | pencil | unknown",
    "paper": "lined | unlined | grid | unknown",
    "scan_type": "scan | photo | copy",
    "condition": "ordinary | unusual | unknown"
  },
  "layout_graph": {},
  "stroke_graph": {},
  "features": {
    "general": {},
    "special": {},
    "private": {}
  },
  "confidence": {
    "graph_quality": 0.82,
    "feature_reliability": 0.74,
    "informativeness": 0.69
  }
}
```

Это сразу учитывает Жижину: нельзя отрывать рукопись от условий, материала, темпа, сопоставимости и объёма.

---

# 19. Новый экспериментальный протокол

## Эксперимент A. Чистая векторизация

Сравнить:

```text
classical skeleton graph
Potrace-like contour vectorization
Bezier stroke extraction
our handwriting-informed CSG
```

Метрики:

```text
skeleton Chamfer distance
junction F1
endpoint F1
loop preservation
component consistency
Bezier fitting error
raster reconstruction IoU
```

Это продолжает технический конвейер, который мы уже выбрали: скан → предобработка → скелет → граф → штрихи → кривые → GNN/CTC. `Исполнительное резюме.pdf`

## Эксперимент B. Почерковедческие признаки

Проверить, насколько модель правильно извлекает:

```text
наклон
размер
разгон
связность
форма движений
петли
точки соединения
точки пересечения
pressure proxy
tempo proxy
coordination score
```

Метрики:

```text
feature classification accuracy
feature regression MAE
human-expert agreement
feature stability score
```

## Эксперимент C. Устойчивость и вариационность

Для одного автора:

```text
normal vs slow
normal vs fast
normal vs different pen
normal vs different paper
```

Проверить:

```text
что сохраняется?
что меняется?
какие признаки являются ядром?
какие признаки ситуативны?
```

## Эксперимент D. HTR benefit

Сравнить:

```text
image-only
graph-only
image + graph
image + graph + handwriting-informed features
```

Метрики:

```text
CER
WER
confidence calibration
robustness under distortion
cross-script transfer RU ↔ EN
```

## Эксперимент E. Информативность

Подать модели:

```text
длинный текст
средний текст
короткую запись
одно слово
подпись-подобный объект
```

Проверить, умеет ли она снижать confidence при малой информативности.

---

# 20. Итоговая научная формулировка после Жижиной

Теперь я бы уточнил тему так:

> **Почерковедчески-информированная каноническая графово-векторная реконструкция офлайн-рукописей с оценкой устойчивости, вариационности и информативности признаков для русско-английского HTR.**

На английском:

> **Handwriting-Forensics-Informed Canonical Stroke Graph Vectorization with Stability- and Informativeness-Aware Features for Offline Russian-English HTR.**

Это уже очень сильная постановка.

---

# 21. Главная новая гипотеза

После Жижиной гипотеза становится точнее:

> **Если графово-векторное представление рукописи строить не как произвольный набор геометрических кривых, а как систему наблюдаемых признаков письменно-двигательного навыка, учитывающую общие, особенные и частные признаки, их устойчивость, вариационность и информативность, то такое представление будет лучше для HTR, переноса между письменностями и устойчивости к изменённым условиям письма.**

---

# 22. Что это даёт нашей модели по сравнению с обычным OCR

Обычный OCR:

```text
image → text
```

Наша модель:

```text
image
→ observable stroke graph
→ handwriting-informed features
→ latent motor-property representation
→ text + uncertainty + stability + style
```

То есть результат будет не только:

```text
"распознанный текст"
```

а:

```text
"распознанный текст + структурное объяснение того, как он написан"
```

---

# 23. Важное ограничение

Нужно явно написать в будущей работе: мы **не занимаемся графологией** и не делаем выводы о характере, личности или психике человека по почерку. Мы используем почерковедение только в той части, где оно описывает:

```text
письменно-двигательный навык
видимые признаки рукописи
устойчивость
вариационность
условия выполнения
сопоставимость материалов
```

Это защитит работу методологически.

---

# 24. Финальное дополнение к нашей идее

Самое ценное, что даёт Жижина:

1. **Признак должен быть связан со свойством**, а не быть произвольным числом.
2. **Информативность признака зависит от задачи**: HTR, writer style, diagnostic robustness, vector quality.
3. **Устойчивость внутри автора и редкость между авторами** - ключ к значимости признака.
4. **Объём и сопоставимость материала** должны учитываться моделью.
5. **Темп, условия, инструмент, материал письма** должны стать metadata и аугментациями.
6. **Особенные признаки** между общими и частными можно превратить в allograph-block graph.
7. **Модель должна оценивать не только результат, но и достаточность информации**.

Итоговая модель теперь должна называться не просто CSG, а, например:

```text
HI-CSG: Handwriting-Informed Canonical Stroke Graph
```

Где HI-CSG - это граф, в котором каждый штрих, петля, соединение, пересечение, ширина, наклон, размер, разгон, связность и точка размещения имеют не только геометрический, но и почерковедческий смысл.

## Prompt:
23/05/2026, 16:32:10

вот еще

[Attachment: Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc]

## Response:
23/05/2026, 16:32:11

По этому файлу ключевое отличие: это уже “особенная часть” — не теория вообще, а методики для сложных случаев: цифры, разрыв во времени, сходные почерки, необычные условия, маскировка, подражание. Именно это хорошо превращается в модули робастности модели.

> **Analyzing document content and cross-referencing**
>
> Alright, I need to provide an integrated analysis and reference the current file. It seems I should also mention previous documents, like Ser/Zh, without adding specifics that aren't supported. I'll look at the current document for unusual conditions and focus on intentional changes. Searching "скорописная маскировка" seems like a good next step.
>
> **Thinking through document analysis and citation details**
>
> I need to mention that the current document is too long to cite every line but can still provide file references. I'll check for "unusual external conditions" in the snippets and include relevant ideas from previous documents like Seregin and Zhizhina. I should also note that some uploaded files have expired, but the current ones seem fine to use. It’s likely the user wants a detailed theoretical addition and analysis without overdoing it, while still emphasizing key applied methods, like imitation and altered conditions.
>
> Thought for 44s

Отлично, этот источник очень важный. Это уже не “общая теория почерка”, а **особенная часть судебно-почерковедческой экспертизы**: как исследовать трудные случаи - цифровой почерк, тексты с разрывом во времени, сходные почерки, необычные условия письма, умышленное изменение, подражание, печатный почерк, диагностические задачи. В книге прямо указано, что она содержит методики решения идентификационных и диагностических задач и сочетает теоретический анализ с большим объёмом экспериментальных данных. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Главный вывод для нашей работы: модель должна быть не только **stroke graph vectorizer**, а **робастной системой анализа почерка в разных режимах исполнения**.

То есть мы усиливаем нашу модель:

```text
HI-CSG → HI-CSG-R
Handwriting-Informed Canonical Stroke Graph
+
Robustness / Variation / Condition / Time / Imitation modeling
```

---

# 1. Что этот источник добавляет к Серегину и Жижиной

Серегин дал нам фундаментальную теорию:

```text
почерк = проявление письменно-двигательного навыка
```

Жижина уточнила методологию:

```text
признак = наблюдаемое проявление свойства навыка
```

Новый учебник добавляет:

```text
один и тот же почерк может проявляться по-разному
в зависимости от объекта, времени, условий, намерения, сходства и объёма материала
```

Это очень важно для модели. Нельзя обучать её так, будто существует один стабильный “почерк автора”. Правильнее:

```text
writer identity
    +
writer state
    +
writing condition
    +
script mode
    +
time period
    +
object type
    =
observed handwriting graph
```

То есть наблюдаемый граф рукописи - это не просто стиль автора. Это результат взаимодействия нескольких факторов.

---

# 2. Новая общая формула модели

Я бы теперь описывал рукопись так:

```text
Observed handwriting image =
    Text content
  + Writer motor program
  + Script system
  + Temporal stage
  + Writing condition
  + Writing object type
  + Intentional / unintentional modification
  + Scanning noise
```

А наша модель должна строить:

```text
image
→ canonical stroke graph
→ handwriting-informed features
→ disentangled representation
→ text + structural explanation + confidence
```

Где disentangled representation - это разложение на факторы:

```json
{
  "text_content": "...",
  "writer_core": {},
  "script_mode": "cursive | print-like | numeric | mixed",
  "temporal_stage": "forming | stable | degrading | unknown",
  "condition_state": "ordinary | unusual_external | unusual_internal | intentional_alteration | unknown",
  "stroke_graph": {},
  "feature_reliability": {},
  "informativeness": {}
}
```

---

# 3. Самое ценное: цифровой почерк как отдельный режим

В книге цифровой почерк рассматривается как самостоятельный объект. Цифровые записи связаны с буквенным письмом общей психофизиологической основой, но отличаются меньшим числом знаков, конструктивной простотой и часто несвязным выполнением. Поэтому они беднее как почерковый материал, но имеют собственные признаки: транскрипцию дат, сумм, веса, нумерации, размещение цифровых записей, интервалы, положение дробей, математических знаков и т. д. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Для нашей модели это прямо полезно. Мы не должны обрабатывать цифры как “просто ещё один класс символов”. Нужно добавить отдельный режим:

```text
Numeric Handwriting Graph Mode
```

Потому что цифры:

```text
имеют мало классов,
часто пишутся раздельно,
часто находятся в бланках,
могут иметь транскрипционные паттерны,
могут быть малоинформативны,
но содержат важные микропризнаки.
```

## Что добавить в модель

Для цифровых записей нужен отдельный блок:

```json
{
  "numeric_graph": {
    "digit_elements": [],
    "digit_strokes": [],
    "transcription_pattern": {},
    "form_context": {},
    "numeric_spacing": {},
    "digit_variant_distribution": {}
  }
}
```

Признаки:

```text
форма цифры,
форма начального штриха,
форма заключительного штриха,
количество движений,
дополнительные штрихи,
точки начала / окончания / пересечения,
размещение относительно строки и графы,
размер дробной части,
наличие точек, запятых, тире, скобок,
наличие слов-спутников: руб., коп., кг, г., № и т.д.
```

Это особенно важно для русских документов: заявления, ведомости, накладные, даты, суммы, номера.

---

# 4. Принцип “признаки в признаках”

В разделе о цифровом почерке есть очень сильная методическая идея: в кратких цифровых записях мало материала, поэтому нужно искать **“признаки в признаках”**, максимально детализировать частные признаки и точно определять участок знака - элемент, штрих, часть элемента, где проявился признак. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Это нужно перенести на всю нашу модель.

Раньше мы говорили:

```text
stroke graph edge = штрих
```

Теперь лучше:

```text
stroke graph edge
    ↓
sub-stroke segment
    ↓
micro-feature
```

То есть у каждого ребра графа должны быть подучастки:

```json
{
  "edge_id": 17,
  "segments": [
    {
      "position": "initial",
      "shape": "arc",
      "width_profile": "...",
      "curvature": 0.41,
      "start_type": "sharp | blunt | dot-like | thin"
    },
    {
      "position": "middle",
      "shape": "straight",
      "waviness": 0.08
    },
    {
      "position": "terminal",
      "shape": "hook",
      "end_type": "thin | blunt | prolonged"
    }
  ]
}
```

Это очень хороший научный вклад: **multi-resolution stroke graph**.

---

# 5. Временная изменчивость почерка

Глава о рукописях, выполненных с разрывом во времени, добавляет важную вещь: почерк не просто “устойчивый”, он проходит жизненные стадии. Для высоковыработанного почерка разрыв может быть в несколько лет и даже десятки лет, а для учащихся или пожилых людей даже 1-2 года могут быть существенными, потому что признаки быстро меняются. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Это полностью меняет нашу идею writer embedding.

Нельзя делать:

```text
writer_id → один неизменный стиль
```

Нужно делать:

```text
writer_id + time_state → handwriting style
```

## Стадии

Из книги следует такая схема:

```text
1. Формирующийся почерк
   школьники, подростки, молодые люди с активной письменной практикой

2. Сформированный почерк
   взрослые, стабильный письменно-двигательный навык

3. Регрессивно изменяющийся почерк
   пожилой возраст, снижение практики, заболевания, ухудшение зрения/моторики
```

## Что добавить в модель

```json
{
  "temporal_state": {
    "stage": "forming | mature | degrading | unknown",
    "development_score": 0.74,
    "degradation_score": 0.12,
    "practice_proxy": 0.68,
    "temporal_uncertainty": 0.31
  }
}
```

Для реализации:

```text
same writer, different years → temporal contrastive learning
```

Модель должна учиться:

```text
сохранять writer-core,
но разрешать изменение отдельных признаков во времени.
```

---

# 6. Writer-core и mutable features

После этой книги я бы явно разделил признаки на два типа:

```text
writer-core features
    признаки, которые сохраняются даже при времени/условиях/маскировке

mutable features
    признаки, которые легко меняются из-за темпа, условий, возраста, намерения
```

Например:

```text
Более устойчивые:
    относительное размещение точек,
    форма некоторых соединений,
    характерные варианты букв,
    некоторые подстрочные/надстрочные элементы,
    топографические привычки,
    относительные пропорции элементов.

Менее устойчивые:
    размер,
    разгон,
    общий наклон,
    степень связности,
    темп-прокси,
    часть пространственной ориентации,
    степень упрощения.
```

Для модели:

```json
{
  "feature": "connection_shape_о_to_next",
  "value": "...",
  "stability_type": "core",
  "condition_sensitivity": 0.18,
  "time_sensitivity": 0.22
}
```

Или:

```json
{
  "feature": "global_slant",
  "value": 71.3,
  "stability_type": "mutable",
  "condition_sensitivity": 0.76,
  "time_sensitivity": 0.44
}
```

Это делает модель намного более научной.

---

# 7. Сходные почерки как hard negatives

Глава о сходных почерках очень важна для машинного обучения. В книге сходными называются почерки разных лиц, у которых совпадают строение, большинство общих признаков и часть частных признаков; выделяются минимальная, средняя и максимальная степень сходства. При исследовании таких почерков эксперт должен особенно внимательно анализировать устойчивые различия, варианты признаков и их количественное соотношение. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

В ML это называется:

```text
hard negative problem
```

То есть для модели обычные “разные авторы” слишком лёгкие. Нужно специально подбирать пары:

```text
different writers, similar handwriting
```

## Что добавить в обучение

```text
Hard-negative contrastive learning
```

Схема:

```text
anchor: рукопись автора A
positive: другая рукопись автора A
easy negative: автор B с сильно другим почерком
hard negative: автор C со сходным почерком
```

Loss:

```text
writer_core(A1) близко к writer_core(A2)
writer_core(A1) далеко от hard_negative(C),
но только по устойчивым детализированным признакам
```

То есть модель должна учиться различать не очевидные вещи вроде размера и наклона, а **тонкие частные признаки**.

---

# 8. Частотность вариантов признака

В сходных почерках иногда различается не сам признак, а **частота появления вариантов**. В книге прямо указывается, что при близком сходстве различий частных признаков может быть мало, и тогда используется разная частота появления одних и тех же вариантов в исследуемой рукописи и образцах. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Это очень сильная идея для нашей модели.

Не просто:

```text
у автора есть вариант буквы “д”
```

А:

```text
у автора вариант д_1 встречается 70%,
д_2 - 20%,
д_3 - 10%
```

Для модели:

```json
{
  "allograph_distribution": {
    "д": {
      "variant_1": 0.71,
      "variant_2": 0.22,
      "variant_3": 0.07
    },
    "т": {
      "variant_1": 0.48,
      "variant_2": 0.52
    }
  }
}
```

Это можно сделать отдельным вкладом:

> **variant-frequency-aware handwriting representation**

---

# 9. Необычные условия письма как отдельный фактор

В книге подробно разбираются необычные условия: непривычная поза, необычный материал, необычный пишущий прибор, недостаточное освещение или темнота, возрастные изменения, стресс, возбуждение, заторможенность, алкогольное/наркотическое опьянение, болезни, непривычное замедление или ускорение письма. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Главная мысль для нас:

```text
необычные условия меняют признаки не хаотично,
а по определённым паттернам
```

Значит, модель должна иметь не просто noise robustness, а **condition-aware robustness**.

## Condition-aware representation

```json
{
  "condition_factors": {
    "external_condition": {
      "unusual_pose": 0.12,
      "unusual_surface": 0.04,
      "low_light": 0.08
    },
    "internal_condition": {
      "coordination_degradation": 0.22,
      "fatigue_like": 0.18,
      "tremor_like": 0.07
    },
    "tempo_condition": {
      "unusually_slow": 0.31,
      "unusually_fast": 0.11
    }
  }
}
```

Важно: мы не должны делать судебно-медицинские выводы вроде “человек был пьян”. Для нашей работы безопаснее и научнее формулировать:

```text
execution condition anomaly
motor regularity degradation
coordination degradation
tempo abnormality
```

---

# 10. Инвариантность и чувствительность

Из этого источника следует очень важный архитектурный принцип:

```text
одни признаки должны быть инвариантны к условиям,
другие должны специально отражать условия
```

Например:

| Признак | Как использовать |
|---|---|
| форма устойчивых соединений | сохранять как writer-core |
| локальная извилистость | condition-sensitive |
| общий размер | mutable |
| размещение точек соединения | чаще writer-core |
| неравномерность наклона | condition-sensitive |
| петли и циклы | partly core, partly condition-sensitive |
| толщина штриха | зависит от прибора/материала |

В архитектуре это можно сделать так:

```text
stroke graph encoder
    ↓
factor disentanglement
    ├── content representation
    ├── writer-core representation
    ├── condition representation
    ├── temporal representation
    └── acquisition-noise representation
```

---

# 11. Непривычная поза, материал, темнота: разные графовые сигнатуры

Из раздела о необычной внешней обстановке можно сделать набор “графовых диагностических сигнатур”.

## Непривычная поза

Характерно:

```text
снижение координации,
неустойчивость размера/разгона,
иногда увеличение связности,
упрощение отдельных движений,
при этом ряд частных признаков сохраняется.
```

Графовая сигнатура:

```json
{
  "pose_like_signature": {
    "coordination_drop": true,
    "connectivity_increase": true,
    "local_shape_core_preserved": true,
    "size_instability": true
  }
}
```

## Необычный материал

Характерно:

```text
снижение темпа,
упрощение,
увеличение размера,
снижение связности,
увеличение нажима/ширины,
деформация из-за поверхности.
```

Графовая сигнатура:

```json
{
  "surface_like_signature": {
    "stroke_width_increase": true,
    "connectivity_decrease": true,
    "size_increase": true,
    "shape_simplification": true,
    "surface_noise": true
  }
}
```

## Темнота / слабое освещение

Характерно:

```text
нарушение пространственной ориентации,
изменение линий строк,
пересечение строк/слов,
но частные признаки часто сохраняются.
```

Графовая сигнатура:

```json
{
  "low_visibility_signature": {
    "layout_disorganization": true,
    "line_crossing": true,
    "baseline_instability": true,
    "letter_core_preserved": true
  }
}
```

Это полезно для нашей модели: если видим плохой layout, но буквы локально стабильны, это не обязательно другой автор или плохая векторизация - это может быть condition factor.

---

# 12. Умышленное изменение почерка

Книга отдельно рассматривает тексты, выполненные намеренно изменённым почерком скорописным способом. Основные виды изменения: наклон, выработанность, общее строение, размер по вертикали, разгон по горизонтали, связность, отдельные частные признаки. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Для нашей задачи это особенно важно, потому что intentional alteration - это adversarial case.

## Как моделировать

```text
Intentional alteration = writer-core + transformation + deautomation artifacts
```

То есть наблюдаемая рукопись:

```text
observed_graph =
    writer_core_graph
  + deliberate_transform
  + deautomation_noise
```

Где deliberate_transform:

```text
изменение наклона,
увеличение/уменьшение размера,
изменение связности,
упрощение/усложнение,
переход к печатным формам.
```

А deautomation artifacts:

```text
остановки,
тупые начала/окончания,
недифференцированный нажим,
неравномерность,
нарушение ритма.
```

В модели:

```json
{
  "intentional_alteration": {
    "probability": 0.34,
    "transforms": {
      "slant_change": 0.71,
      "size_change": 0.42,
      "connectivity_change": 0.38,
      "simplification": 0.55
    },
    "deautomation_artifacts": {
      "stops": 0.24,
      "blunt_terminals": 0.31,
      "rhythm_breaks": 0.27
    }
  }
}
```

---

# 13. Подражание чужому почерку как mixture model

Глава о подражании почерку другого лица особенно полезна. Там подражание определяется как выполнение рукописи с установкой на воспроизведение признаков почерка другого лица; при этом исполнитель одновременно воспроизводит чужие признаки и подавляет свои, что нарушает привычный процесс письма. В рукописи смешиваются признаки обычного почерка исполнителя, имитация чужих признаков и трансформированные признаки самого исполнителя. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Это почти готовая формула для модели:

```text
Imitated handwriting =
    executor_core
  + target_style_imitation
  + deautomation_artifacts
  + failed_imitation_residue
```

В графовой форме:

```json
{
  "imitation_model": {
    "executor_core_features": {},
    "target_like_features": {},
    "deautomation_features": {},
    "residual_inconsistencies": {}
  }
}
```

Очень интересная исследовательская идея:

> обучить модель находить внутреннюю противоречивость графа: часть признаков похожа на один стиль, часть - на другой, а между ними есть признаки деавтоматизации.

Для обычного HTR это не нужно. Для нашей работы - это сильный теоретический плюс.

---

# 14. Печатный почерк и script-mode

В книге есть отдельный блок про тексты, выполненные с подражанием печатному шрифту. Для таких текстов важны не только привычные признаки рукописного письма, но и признаки построения печатных букв: положение элементов, последовательность исполнения, тип шрифта, стилизованность. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Для модели это означает: нельзя считать все рукописи курсивными. Нужен переключатель режима:

```text
script_mode:
    cursive
    print-like
    block letters
    numeric
    mixed
```

Для каждого режима - свои признаки:

```json
{
  "script_mode": "print_like",
  "print_graph_features": {
    "element_alignment": {},
    "stroke_intersections": {},
    "parallelism": {},
    "letter_construction_group": {},
    "execution_sequence_hypotheses": {}
  }
}
```

Это особенно полезно для русских рукописей: люди часто смешивают скоропись, печатные буквы, латиницу, цифры и сокращения.

---

# 15. Непривычная левая рука

В книге указано, что при перемене пишущей руки часть навыков переносится: орфографические навыки, технические представления, зрительные образы письменных знаков и представления о движениях. Но двигательная реализация нарушается, и степень переноса зависит от особенностей нервных процессов конкретного исполнителя. Один из диагностических признаков - разные проявления зеркальности. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Для модели это важная идея: **нужно разделять визуально-графический образ буквы и моторную реализацию**.

То есть:

```text
letter visual schema
≠
motor execution quality
```

В архитектуре:

```text
letter-shape encoder
+
motor-execution encoder
```

Пример:

```json
{
  "letter_schema_preserved": 0.83,
  "motor_execution_degraded": 0.61,
  "mirror_errors": 0.24,
  "hand_switch_like": 0.17
}
```

Опять же: мы не делаем вывод “написано левой рукой”, но можем учитывать режим **motor transfer anomaly**.

---

# 16. Методика исследования сходных почерков → модель “тонких различий”

Для сходных почерков книга рекомендует максимально детализировать признаки, изучать варианты, делать алфавитную разработку проявлений признаков и учитывать частоту вариантов. Это идеально для нашей научной постановки.

В машинной модели это можно реализовать через:

```text
Allograph Variant Inventory
```

То есть для каждого автора/документа:

```json
{
  "letter_inventory": {
    "а": [
      {"variant": "oval_open_top", "frequency": 0.34},
      {"variant": "closed_oval", "frequency": 0.51},
      {"variant": "angular_simplified", "frequency": 0.15}
    ],
    "д": [
      {"variant": "looped_descender", "frequency": 0.78},
      {"variant": "straight_descender", "frequency": 0.22}
    ]
  }
}
```

И отдельно:

```json
{
  "micro_feature_inventory": {
    "start_point_positions": {},
    "crossing_positions": {},
    "connector_shapes": {},
    "terminal_shapes": {}
  }
}
```

Это превращает экспертную “алфавитную разработку” в машинное представление.

---

# 17. Важный принцип: объём материала определяет надёжность

Книга постоянно возвращается к тому, что короткие записи сложнее исследовать, а чем короче спорная запись, тем больше нужно образцов. Для цифрового почерка прямо указано: экспериментальные образцы должны быть повторены несколько раз, а общее правило - чем короче запись, тем сложнее исследование и тем больше образцов нужно. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Это надо встроить как обязательный модуль:

```text
Informativeness Estimator
```

Он оценивает:

```json
{
  "informativeness": {
    "text_length": 0.42,
    "symbol_diversity": 0.38,
    "letter_repetition": 0.61,
    "digit_repetition": 0.14,
    "allograph_coverage": 0.49,
    "stroke_complexity": 0.57,
    "estimated_reliability": 0.52
  }
}
```

Для HTR это будет confidence calibration. Для writer/style анализа - защита от ложных выводов.

---

# 18. Дополнение к HI-CSG: теперь нужна иерархия объектов

После этой книги я бы изменил формат графа. Теперь он должен учитывать не только строки и буквы, но и тип объекта:

```text
document graph
  ├── text block graph
  ├── line graph
  ├── word graph
  ├── character/allograph graph
  ├── digit graph
  ├── punctuation graph
  ├── correction/overwrite graph
  └── stroke primitive graph
```

Особенно важны:

```text
correction graph
```

и

```text
form-field graph
```

Потому что цифровые записи, дописки, исправления, бланки и графы - это типичные реальные документы.

---

# 19. Новый формат HI-CSG-R

Я бы теперь описал нашу модель так:

```json
{
  "document": {
    "layout_graph": {},
    "form_field_graph": {},
    "line_graphs": []
  },
  "stroke_graph": {
    "nodes": [],
    "edges": [],
    "loops": [],
    "sub_stroke_features": []
  },
  "symbol_graph": {
    "letters": [],
    "digits": [],
    "punctuation": [],
    "special_marks": []
  },
  "handwriting_features": {
    "general": {},
    "special": {},
    "private": {},
    "numeric": {},
    "topographic": {}
  },
  "factor_states": {
    "writer_core": {},
    "temporal_stage": {},
    "condition_state": {},
    "script_mode": {},
    "intentional_alteration": {},
    "imitation_mixture": {}
  },
  "reliability": {
    "informativeness": {},
    "feature_confidence": {},
    "graph_quality": {},
    "ocr_confidence": {}
  }
}
```

---

# 20. Как это влияет на архитектуру

Теперь архитектура должна быть не просто image-to-graph-to-text.

Лучше:

```text
Image
  ↓
Visual Encoder
  ↓
Dense Prediction Heads
  ├── skeleton
  ├── stroke width
  ├── tangent field
  ├── endpoint / junction / crossing
  ├── baseline / zones
  ├── digit / letter / punctuation masks
  ├── correction / overwrite candidates
  └── form-field / layout cues
  ↓
Hierarchical Graph Builder
  ↓
HI-CSG-R
  ↓
Factorized Graph Encoder
  ├── content branch
  ├── writer-core branch
  ├── condition branch
  ├── temporal branch
  ├── script-mode branch
  └── informativeness branch
  ↓
Outputs
  ├── recognized text
  ├── vector graph
  ├── structural features
  ├── confidence
  └── condition/variation explanation
```

---

# 21. Новые loss-функции

Добавляем к прежним:

```text
L = L_text
  + L_graph
  + L_vector_render
  + L_keypoints
  + L_topology
  + L_width
  + L_writer_core
  + L_condition
  + L_temporal
  + L_script_mode
  + L_digit_features
  + L_variant_frequency
  + L_informativeness
  + L_hard_negative
```

Особенно важные новые:

## L_hard_negative

Для сходных почерков:

```text
разные авторы со сходными общими признаками должны различаться
по тонким устойчивым частным признакам.
```

## L_temporal

Для разрыва во времени:

```text
один автор в разные периоды должен сохранять writer-core,
но отдельные признаки могут изменяться закономерно.
```

## L_condition

Для необычных условий:

```text
модель должна отделять writer-core от condition artifacts.
```

## L_variant_frequency

Для сходных почерков:

```text
важна не только форма варианта,
но и частота его повторения.
```

## L_digit_features

Для цифровых записей:

```text
цифры, даты, суммы и слова-спутники должны анализироваться
как отдельный подтип графового письма.
```

---

# 22. Новые эксперименты для статьи

## Эксперимент 1. OCR + graph на обычных рукописях

Базовый:

```text
image-only
graph-only
image + graph
image + graph + handwriting-informed features
```

## Эксперимент 2. Сходные почерки

Создать hard-negative пары:

```text
похожие по наклону, размеру, связности, строению
но разных авторов
```

Проверить:

```text
улучшает ли micro-feature graph различение сходных рукописей?
```

## Эксперимент 3. Разрыв во времени

Для авторов с разными годами:

```text
same writer, different time
different writer, similar style
```

Проверить:

```text
может ли модель сохранять writer-core и не путать возрастные/временные изменения с другим автором?
```

## Эксперимент 4. Необычные условия

Аугментировать или собрать данные:

```text
медленно,
быстро,
на плохой поверхности,
с плохой видимостью,
с изменённым наклоном,
с упрощением,
с печатными формами.
```

Проверить:

```text
condition-aware model лучше, чем обычная?
```

## Эксперимент 5. Цифровой почерк

Отдельный набор:

```text
даты,
суммы,
номера,
цифровые записи в бланках,
смешанные записи: “руб.”, “кг”, “г.”, “№”
```

Проверить:

```text
помогает ли numeric graph mode при распознавании и векторизации?
```

## Эксперимент 6. Информативность

Подать:

```text
длинный текст,
короткий текст,
одно слово,
несколько цифр,
разрозненные цифры,
исправленную цифру.
```

Проверить:

```text
умеет ли модель снижать confidence при недостаточном материале?
```

---

# 23. Новая научная гипотеза

После этого источника я бы усилил гипотезу так:

> **Почерк в офлайн-изображении следует моделировать не как статический стиль автора, а как факторизованное графово-векторное проявление письменно-двигательного навыка, зависящее от времени, условий, типа рукописного объекта, режима письма и возможной деавтоматизации.**

Кратко:

```text
handwriting graph = writer-core + condition + time + object-mode + variation
```

---

# 24. Итоговая версия темы

Теперь лучшая формулировка:

> **Почерковедчески-информированная каноническая графово-векторная реконструкция офлайн-рукописей с факторизацией устойчивых, временных, ситуативных и объектно-специфических признаков для русско-английского HTR.**

На английском:

> **Handwriting-Forensics-Informed Canonical Stroke Graph Vectorization with Factorized Writer-Core, Temporal, Condition, and Object-Specific Features for Offline Russian-English HTR.**

Чуть короче:

> **HI-CSG-R: Robust Handwriting-Informed Canonical Stroke Graphs for Offline Russian-English HTR.**

---

# 25. Что меняется в нашей концепции окончательно

Раньше:

```text
из скана строим stroke graph и распознаём текст
```

Теперь:

```text
из скана строим иерархический stroke graph,
из него извлекаем признаки почерка,
разделяем устойчивое ядро автора, содержание, условия письма,
временную стадию, режим письма и информативность,
а затем распознаём текст и оцениваем надёжность.
```

Именно это делает работу не просто инженерной, а действительно научной.

Финальная модель должна отвечать не только на вопрос:

```text
что написано?
```

но и на вопросы:

```text
как это написано?
какие штриховые структуры это образуют?
какие признаки устойчивы?
какие признаки ситуативно искажены?
достаточно ли материала для уверенного вывода?
почему модель уверена или не уверена?
```

Это очень сильная основа для теоретического раздела и для проектирования архитектуры.

## Prompt:
23/05/2026, 21:30:05

окей, теперь тебе необходимо провести полный аудит, что необходимо сделать, нужно продумать возможные архитектурные решения, возможные подходы, также обрати внимание на возможные проблемы, нужно провести полноценную предварительную исследовательскую работу от которой можно будет уже составлять полноценный план реализации

## Response:
23/05/2026, 21:30:06

Сделаю аудит как исследовательский pre-plan: разложу задачу на цели, данные, архитектуры, эксперименты, риски и MVP/полную версию. Сначала доберу из загруженных материалов именно методические требования и ограничения, затем сверю с современными ML-подходами.

> **Searching for technical papers on handwriting and SVG**
>
>
> **Searching for current handwriting recognition datasets**
>
>
> **Planning full audit document structure**
>
> It looks like the user needs a comprehensive audit-based response. The document should cover various sections: executive decisions, research goals, data audits, recommended architecture, analysis methods, risks, and even milestones like MVP and V2. I’ll focus on providing factual statements and avoid any forensic claims on personality or medical topics. I'll also remember to emphasize ethical and legal concerns.
>
> **Clarifying citation format for uploaded files**
>
> Okay, I know I need to cite uploaded files based on methods/problems, and references like Seregin FDK streams. The instructions say I should include line ranges, but the search results don't show those—just markers instead. I'll probably proceed with markers for now, like in past responses, and maybe check files directly for line numbers if needed.
>
> **Deciding how to cite sources effectively**
>
> It seems like I don't need to include direct line citations every time, especially with file markers. Web citations should be included without overemphasizing references. For PDFs, I might avoid direct citing and instead reference the abstract or webpage. When using sources like TrOCR, it's better to cite the arXiv page, not the PDF. I'll focus on keeping the response comprehensive but organized, avoiding excessive bullet points.
>
> **Refining the implementation plan and citations**
>
> I need to include citations within the main text, like for methods, executive summaries, and modeling. The web sources should cite baselines, such as TrOCR, DiffVG, and datasets. It makes sense to propose an implementation plan with clear phases, like data pipeline, neural vectorizer, and experiments. I'll also focus on striking a balance between "minimum viable" vs. "ambitious version".
>
> Thought for 40s

Ниже - полноценный предварительный аудит, от которого уже можно переходить к **плану реализации, структуре статьи и техническому ТЗ**.

Мой итоговый вывод: тему нужно оформлять как **не просто HTR/OCR**, а как исследование нового промежуточного представления:

> **HI-CSG-R: Handwriting-Informed Robust Canonical Stroke Graphs for Offline Russian-English Handwritten Text Recognition**

По-русски:

> **Почерковедчески-информированная робастная каноническая графово-векторная реконструкция штрихов офлайн-рукописей для русско-английского HTR.**

---

# 1. Главная исследовательская постановка

Мы хотим решить не одну, а связанную цепочку задач:

```text
скан / фото рукописи
→ предобработка
→ структурная векторизация
→ canonical stroke graph
→ почерковедческие признаки
→ распознавание текста
→ оценка устойчивости / вариативности / информативности
```

Ключевая новизна:

> Мы не восстанавливаем реальную траекторию пера. Мы строим **канонический граф видимой структуры письма**, который сохраняет форму, топологию, связность, петли, точки соединения, толщину, наклон, вариации и почерковедчески значимые признаки.

Это важно, потому что в офлайн-скане настоящего порядка движения пера чаще всего нет. Попытка жёстко восстановить pen trajectory будет методологически слабее, чем честная реконструкция **видимой структурной организации штрихов**.

---

# 2. Что уже есть в литературе и где наш зазор

Технический обзор показывает, что существующие методы структурной векторизации обычно идут по схеме: бинаризация → скелетизация → граф скелета → выделение штрихов → аппроксимация кривыми; при этом часто изучаются отдельные шаги, а не полный конвейер “пиксель → граф/вектор → текст”. Также подчёркнуты три главных пробела: отсутствие штриховых аннотаций, слабая разработанность end-to-end нейро-векторизаторов и отсутствие единых метрик качества структурного представления. `Исполнительное резюме.pdf`

Современные HTR-модели вроде TrOCR сильны как image-to-text baseline: TrOCR использует предобученный image Transformer и text Transformer и является end-to-end OCR-моделью для печатного, рукописного и scene text recognition. Но такая модель не строит явного штрихового графа и почти не даёт интерпретируемой структуры письма. ([arXiv](https://arxiv.org/abs/2109.10282?utm_source=chatgpt.com))

Работа Chan по offline handwritten mathematical expression recognition ближе к нашей идее: она разбивает скелет бинарного изображения на junctions и segments, затем объединяет segments в strokes и нормализует порядок. Но это сделано для математических выражений и ориентировано на перевод offline в online-like representation, а не на почерковедчески-информированный граф обычного русско-английского текста. ([arXiv](https://arxiv.org/abs/1905.06749?utm_source=chatgpt.com))

Дифференцируемая векторная графика, например DiffVG, даёт техническую возможность связывать raster loss и параметры векторных кривых: дифференцируемый rasterizer позволяет оптимизировать векторное представление по пиксельным loss-функциям. Это важно для нас, потому что штриховых ground truth почти нет. ([MIT CSAIL](https://people.csail.mit.edu/tzumao/diffvg/?utm_source=chatgpt.com))

Новые работы по Bézier stroke extraction для рукописных символов прямо указывают на проблему фрагментации пиксельных методов и потери топологической структуры; Bézier-представление лучше соответствует нашей цели, но существующая работа сфокусирована на символах, а не на строках, словах, кириллице и HTR-конвейере. ([BMVC 2025](https://bmvc2025.bmva.org/proceedings/478/?utm_source=chatgpt.com))

---

# 3. Теоретический фундамент из почерковедения

Из Серегина и Жижиной мы берём не судебную цель, а **научную модель почерка**.

Почерк описывается как проявление письменно-двигательного функционально-динамического комплекса навыков. В Серегине подчёркивается, что судебное почерковедение изучает закономерности почерка, методы его исследования, общие и частные признаки, математические методы, стадии экспертизы, изменённый почерк, подписи, краткие и цифровые записи. `Seregin_Pocherkovedenie_2015.pdf`

Очень важное методологическое основание: почерк - сложная динамичная система, функционирующая под влиянием разных факторов; поэтому формализация и математическое моделирование нужны для объективизации исследования. `Seregin_Pocherkovedenie_2015.pdf` Жижина дополнительно подчёркивает комплексность методов: качественно-описательные, графические, инструментальные, количественные, модельные, компьютерные и комплексные подходы. `Zhizhina_M_V_Pocherkovedenie_2006.docx`

Для нашей модели это значит:

```text
CSG не должен быть просто набором кривых.
CSG должен быть измеримой моделью проявлений письменно-двигательного навыка.
```

---

# 4. Итоговая концепция модели

Я предлагаю называть наше представление:

```text
HI-CSG-R
Handwriting-Informed Canonical Stroke Graph with Robustness Factors
```

То есть:

```text
HI-CSG-R =
    stroke graph
  + vector curves
  + handwriting-informed features
  + stability / variation layer
  + condition / temporal / object-mode factors
  + recognition decoder
```

Наблюдаемая рукопись моделируется так:

```text
Observed handwriting =
    text content
  + writer-core motor program
  + script system
  + writing condition
  + temporal stage
  + object mode
  + acquisition noise
```

Где:

```text
writer-core - устойчивое ядро почерка;
condition - необычные условия, темп, материал, поза, прибор;
temporal stage - формирующийся / зрелый / регрессивно изменяющийся почерк;
object mode - буквенный текст, цифры, смешанные записи, печатный режим;
acquisition noise - качество скана/фото.
```

---

# 5. Что именно должно быть выходом модели

Не только текст.

Минимальный выход:

```json
{
  "text": "распознанная строка",
  "stroke_graph": {
    "nodes": [],
    "edges": [],
    "loops": [],
    "components": []
  },
  "vector_representation": {
    "curves": [],
    "format": "polyline | bezier | svg-like"
  },
  "handwriting_features": {
    "general": {},
    "special_blocks": {},
    "private": {},
    "numeric": {}
  },
  "confidence": {
    "ocr": 0.91,
    "graph_quality": 0.78,
    "informativeness": 0.69
  }
}
```

Полный выход:

```json
{
  "recognized_text": "...",
  "layout_graph": {},
  "line_graphs": [],
  "word_graphs": [],
  "symbol_graphs": [],
  "stroke_graph": {},
  "features": {
    "structural_geometric": {},
    "coordination_rhythm": {},
    "pressure_width_proxy": {},
    "topographic": {},
    "private_letter_features": {},
    "numeric_transcription": {}
  },
  "factorized_states": {
    "writer_core": {},
    "condition_state": {},
    "temporal_state": {},
    "script_mode": {},
    "object_mode": {}
  },
  "quality": {
    "vectorization_quality": {},
    "ocr_confidence": {},
    "feature_reliability": {},
    "informativeness": {}
  }
}
```

---

# 6. Ключевые признаки, которые нужно извлекать

## 6.1. Узлы графа

```text
endpoint
junction
crossing
loop_closure
curvature_extreme
connection_point
baseline_touch
upper_zone_extreme
lower_zone_extreme
start_candidate
end_candidate
```

Важно: `start_candidate` и `end_candidate`, а не `start` и `end`, потому что мы не знаем реальный порядок движения пера.

## 6.2. Рёбра графа

```text
visible stroke segment
connector
loop segment
descender segment
ascender segment
terminal segment
additional stroke
correction / overwrite candidate
```

## 6.3. Геометрические признаки ребра

```text
length
curvature
signed curvature
smoothness
angularity
number of inflection points
Bezier control points
polyline simplification error
local tangent field
```

## 6.4. Топологические признаки

```text
number of components
number of cycles
loop preservation
junction degree distribution
crossing count
broken connections
probable pen lifts
graph connectivity
```

## 6.5. Почерковедческие признаки

```text
наклон
размер
разгон
связность
выработанность
координация
темп-прокси
нажим-прокси
форма движений
направление движений
протяжённость
количество движений
вид соединения
точки начала / окончания / соединения / пересечения
```

Серегин подчёркивает, что общие признаки проявляются в рукописи в целом и характеризуют топографию, степень сформированности письменно-двигательного навыка и структуру движений по траектории. `Seregin_Pocherkovedenie_2015.pdf` Частный признак становится характерным только тогда, когда он не просто отклоняется от прописи, но устойчиво проявляется в рукописи. `Seregin_Pocherkovedenie_2015.pdf`

---

# 7. Особые режимы, которые обязательно учитывать

## 7.1. Цифровой почерк

Особенная часть подчёркивает, что цифровые записи имеют меньше знаков, более простую конструкцию и часто несвязное выполнение; поэтому они беднее как материал, но имеют специфические признаки транскрипции дат, сумм, весов, нумерации, размещения цифр и слов-спутников. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Значит, в модели нужен режим:

```text
object_mode = numeric | alphabetic | mixed
```

И отдельные признаки:

```text
numeric transcription
digit spacing
digit height
digit slant
fraction placement
punctuation in numbers
words-companions: руб., коп., кг, г., №
```

## 7.2. Разрыв во времени

В особенной части указано, что для высоковыработанных почерков разрыв во времени может измеряться годами и десятилетиями, а для учащихся и пожилых даже 1-2 года могут быть существенными, потому что признаки меняются быстрее. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Значит, нельзя учить writer embedding как фиксированный стиль. Нужна факторизация:

```text
writer_core + temporal_stage
```

## 7.3. Сходные почерки

Сходные почерки - это hard negatives. Для них недостаточно общих признаков вроде наклона и размера. Нужно детализировать микропризнаки, варианты букв и частотность вариантов.

## 7.4. Необычные условия

Серегин подчёркивает, что процесс выполнения рукописи часто сопровождается внешними или внутренними сбивающими факторами. `Seregin_Pocherkovedenie_2015.pdf` Для модели это значит: нужна не просто noise augmentation, а **condition-aware representation**.

## 7.5. Малый объём материала

Особенная часть прямо формулирует правило: чем короче спорная запись, тем сложнее исследование и тем больше образцов нужно. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc` Для нашей модели это превращается в модуль:

```text
informativeness estimator
```

---

# 8. Данные: что использовать

## 8.1. Английский

**IAM** - основной английский baseline: 657 writers, 1,539 pages, 5,685 sentences, 13,353 text lines, 115,320 words. ([fki.tic.heia-fr.ch](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database?utm_source=chatgpt.com))

Использование:

```text
line-level HTR
word-level vectorization
writer split
cross-script transfer EN → RU
```

## 8.2. Русский современный

**HWR200** - русский датасет изображений рукописных текстов от 200 авторов, сфотографированных в разных условиях. ([Hugging Face](https://huggingface.co/datasets/AntiplagiatCompany/HWR200?utm_source=chatgpt.com))

Использование:

```text
RU line/page HTR
domain robustness
photo-quality variation
writer-level analysis
```

## 8.3. Русский/казахский кириллический

**HKR** - более 1400 заполненных форм, около 63 000 предложений, более 715 699 символов примерно от 200 авторов; 95% русского и 5% казахского текста. ([arXiv](https://arxiv.org/abs/2007.03579?utm_source=chatgpt.com))

Использование:

```text
large-scale Cyrillic pretraining
line recognition
writer-independent validation
```

## 8.4. Русский исторический

**Digital Peter** - 9694 line images с транскриптами строк из рукописей Петра I. ([ar5iv](https://ar5iv.labs.arxiv.org/html/2103.09354?utm_source=chatgpt.com))

Использование:

```text
historical robustness
domain shift
stress test for graph representation
```

## 8.5. Собственная малая разметка

Нам обязательно нужен небольшой gold subset:

```text
300-500 слов / фрагментов
или
1000-2000 символов / цифр
```

Разметка:

```text
skeleton correction
nodes: endpoint / junction / loop / crossing
stroke grouping
Bezier/polyline approximation
letter/digit label
connection label
```

Без этого будет трудно доказать качество структурной векторизации.

---

# 9. Главная проблема данных

Почти все доступные HTR-датасеты имеют транскрипты строк/слов, но не имеют stroke-level graph annotations. Это уже отмечалось в техническом обзоре: нет стандартных датасетов, где к офлайн-изображению приписана разметка порядка или групп пикселей одного штриха. `Исполнительное резюме.pdf`

Поэтому стратегия должна быть такой:

```text
1. Pseudo-labels from classical skeletonization
2. Synthetic / semi-synthetic stroke data
3. Small manually corrected gold subset
4. Self-supervised raster ↔ vector reconstruction
5. Downstream validation through HTR
```

---

# 10. Возможные архитектуры

## Архитектура A. Classical CSG baseline

Это обязательный первый baseline.

```text
image
→ binarization
→ skeletonization
→ graph extraction
→ pruning
→ stroke grouping
→ Bezier/polyline fitting
→ graph features
→ optional HTR
```

Плюсы:

```text
быстро реализовать;
интерпретируемо;
хороший baseline;
даёт pseudo-labels.
```

Минусы:

```text
чувствительно к шуму;
скелетизация ломается на плохих фото;
трудно на сильной связности;
много эвристик.
```

Это нужно сделать первым.

---

## Архитектура B. Neural dense structural prediction

```text
image
→ CNN / U-Net / Swin encoder
→ dense heads:
    skeleton map
    endpoint heatmap
    junction heatmap
    crossing heatmap
    loop heatmap
    tangent field
    width map
    baseline zones
→ graph builder
→ CSG
```

Плюсы:

```text
устойчивее к шуму;
можно учить на pseudo-labels;
хорошо подходит для keypoint extraction.
```

Минусы:

```text
нужна хорошая генерация pseudo-labels;
dense maps ещё не дают готовый граф;
нужно аккуратно строить graph builder.
```

Это лучший второй этап.

---

## Архитектура C. Differentiable vectorizer

```text
image
→ neural curve proposal
→ Bezier curves
→ differentiable rasterizer
→ raster reconstruction loss
```

Плюсы:

```text
не требует полного stroke ground truth;
получаем настоящий векторный выход;
научно сильная идея.
```

Минусы:

```text
сложная оптимизация;
много локальных минимумов;
трудно масштабировать до строки;
дифференцируемый рендеринг может быть тяжёлым.
```

Использовать не как первую версию, а как продвинутый модуль.

---

## Архитектура D. Graph-only HTR

```text
CSG
→ Graph Neural Network / Graph Transformer
→ sequence pooling
→ CTC / Transformer decoder
→ text
```

Плюсы:

```text
проверяет ценность графа как самостоятельного представления;
хорошо для интерпретируемости.
```

Минусы:

```text
может проиграть raster models;
сложно упорядочивать граф в строку;
графы строк могут быть большими.
```

Использовать как эксперимент, но не как единственную модель.

---

## Архитектура E. Raster + Graph fusion

Это основной рекомендуемый вариант.

```text
image encoder ──────────────┐
                            ├→ fusion → text decoder
graph encoder ──────────────┘
```

Более детально:

```text
Image
  ↓
Visual Encoder: CNN/Swin/ViT
  ↓
Visual features

Image
  ↓
CSG Extractor
  ↓
Graph Encoder: GNN/Graph Transformer
  ↓
Graph features

Visual features + Graph features
  ↓
Cross-attention / gated fusion
  ↓
CTC or Transformer decoder
  ↓
Text
```

Плюсы:

```text
raster branch видит пиксельные детали;
graph branch видит структуру;
можно сравнить image-only, graph-only, image+graph;
научно доказуемо.
```

Минусы:

```text
дороже;
сложнее отлаживать;
есть риск, что graph branch не даст прироста на чистых данных.
```

Это должна быть **основная архитектура статьи**.

---

# 11. Рекомендуемая итоговая архитектура

Я бы строил так:

```text
Input image
  ↓
Preprocessing module
  ↓
Shared visual encoder
  ↓
Dense structural heads
  ├── skeleton probability
  ├── stroke width map
  ├── tangent field
  ├── endpoint heatmap
  ├── junction heatmap
  ├── crossing heatmap
  ├── loop heatmap
  ├── baseline / x-height / zones
  └── symbol-region masks
  ↓
Graph builder
  ↓
HI-CSG-R
  ↓
Graph encoder
  ├── content branch
  ├── writer-core branch
  ├── condition branch
  ├── temporal / variation branch
  └── informativeness branch
  ↓
Raster-graph fusion
  ↓
Outputs:
  ├── recognized text
  ├── vector graph
  ├── structural metrics
  ├── feature stability
  └── confidence
```

---

# 12. Технические модули

## 12.1. Preprocessing

```text
deskew
dewarping
contrast normalization
background removal
binarization
line segmentation
word/patch extraction
noise removal
```

Важно: не уничтожить признаки. Например, нельзя агрессивно нормализовать наклон, потому что наклон - почерковедческий признак.

Правило:

```text
делать две версии изображения:
1. normalized image для OCR;
2. forensic-preserving image для признаков.
```

## 12.2. Skeleton graph extractor

```text
binary image
→ skeleton
→ pixel graph
→ node detection
→ edge tracing
→ pruning
→ smoothing
```

Узлы:

```text
degree = 1 → endpoint
degree = 2 → intermediate
degree ≥ 3 → junction
cycle closure → loop
intersection / crossing → crossing candidate
```

## 12.3. Stroke grouping

Эвристики:

```text
минимальный угол поворота;
плавность кривой;
одинаковая толщина;
локальная непрерывность;
тип соединения;
вероятность lift-up.
```

## 12.4. Curve fitting

```text
polyline simplification: Ramer-Douglas-Peucker
Bezier fitting
B-spline fitting
arc/line primitive fitting
```

## 12.5. Graph encoder

Варианты:

```text
GAT
Graph Transformer
Edge-conditioned GNN
Graphormer-like model
Message passing with edge attributes
```

Атрибуты узлов:

```text
x, y
node type
degree
zone
local width
confidence
```

Атрибуты рёбер:

```text
length
curvature
shape type
width profile
smoothness
tangent histogram
connection type
Bezier parameters
```

## 12.6. Fusion module

Лучшие варианты:

```text
late fusion:
    image logits + graph logits

feature fusion:
    concat / gated fusion

cross-attention:
    text decoder attends to image tokens and graph tokens

dual decoder:
    image decoder + graph decoder + consistency loss
```

Рекомендация: начать с **late fusion**, затем перейти к **cross-attention fusion**.

---

# 13. Loss-функции

Минимальный набор:

```text
L = L_text + L_skeleton + L_keypoint + L_vector_render
```

Расширенный набор:

```text
L =
  L_text
+ L_skeleton
+ L_endpoint
+ L_junction
+ L_loop
+ L_width
+ L_tangent
+ L_graph_topology
+ L_bezier_fit
+ L_vector_render
+ L_feature_stability
+ L_writer_contrastive
+ L_condition
+ L_informativeness
```

## Самые важные

### L_text

```text
CTC loss или seq2seq cross-entropy
```

### L_skeleton

Сравнение predicted skeleton map с pseudo-skeleton.

### L_keypoint

Endpoint / junction / crossing / loop heatmap.

### L_vector_render

Граф рендерится обратно в растр, затем сравнивается с исходным изображением.

### L_topology

Штраф за:

```text
потерянные петли;
лишние компоненты;
разрывы связных букв;
ложные соединения;
искажённые пересечения.
```

### L_stability

Один автор / похожие фрагменты должны иметь близкие устойчивые признаки.

### L_condition

Модель должна отделять устойчивые признаки от ситуативных деформаций.

---

# 14. Метрики

## 14.1. OCR metrics

```text
CER
WER
line accuracy
word accuracy
```

## 14.2. Structural vectorization metrics

```text
Skeleton F1
Endpoint F1
Junction F1
Loop preservation score
Connected component consistency
Graph edit distance
Chamfer distance between skeletons
Fréchet distance between curves
Bezier fitting error
Raster reconstruction IoU
```

Технический обзор уже отмечает, что CER/WER недостаточны, потому что не чувствительны к качеству полученной структуры; нужны графовые и криволинейные метрики. `Исполнительное резюме.pdf`

## 14.3. Handwriting-informed metrics

```text
slant MAE
size MAE
spacing MAE
connectivity error
pressure-proxy consistency
tempo-proxy consistency
feature repeatability
variant-frequency similarity
```

## 14.4. Robustness metrics

```text
CER under blur
CER under noise
CER under low contrast
CER under skeleton breaks
graph quality degradation
confidence calibration
```

## 14.5. Cross-script metrics

```text
train EN → test RU
train RU → test EN
train RU+EN → test historical RU
```

---

# 15. Главные исследовательские гипотезы

## H1. Graph helps HTR under difficult conditions

```text
image + graph > image-only
```

Особенно на:

```text
шумных фото;
исторических рукописях;
малых данных;
кириллице;
сильно связной скорописи.
```

## H2. HI-CSG лучше обычного skeleton graph

```text
handwriting-informed graph > naive skeleton graph
```

По:

```text
loop preservation;
junction correctness;
recognition robustness;
feature stability.
```

## H3. Stable/mutable feature separation improves transfer

```text
writer-core + condition factorization
лучше переносится между авторами, условиями и письменностями.
```

## H4. Special block features improve Cyrillic

```text
оваловые блоки, подстрочные блоки, надстрочные блоки,
connector groups улучшают распознавание русской скорописи.
```

## H5. Informativeness estimator improves confidence

```text
модель должна знать, когда структурной информации недостаточно.
```

---

# 16. Главные проблемы и риски

## Проблема 1. Нет настоящей stroke-разметки

Риск: модель будет учиться на ошибках skeleton pseudo-labels.

Решение:

```text
pseudo-label cleaning;
manual gold subset;
synthetic data with known vector ground truth;
self-supervised raster reconstruction;
evaluate not only pseudo-label agreement, but also HTR effect.
```

## Проблема 2. Скелетизация нестабильна

Особенно на:

```text
плохих фото;
толстых штрихах;
пересечениях;
залитых петлях;
размытых линиях;
старых документах.
```

Решение:

```text
multi-threshold skeleton ensemble;
learned skeleton map;
width-aware skeleton pruning;
topology-preserving postprocessing.
```

## Проблема 3. Давление пера почти не наблюдаемо

Риск: некорректно утверждать, что мы восстановили нажим.

Решение:

```text
говорить pressure proxy;
использовать stroke width + ink density + local contrast;
хранить reliability score.
```

## Проблема 4. Темп письма не наблюдаем напрямую

Решение:

```text
говорить tempo proxy;
использовать связность, упрощение, тупые окончания, ретушь, остановки.
```

## Проблема 5. Графы строк слишком большие

Решение:

```text
patch-level graph;
word-level graph;
hierarchical graph pooling;
subgraph batching;
line graph = sequence of word graphs.
```

## Проблема 6. Graph branch может не улучшить clean OCR

На чистом IAM TrOCR-like baseline может быть сильнее.

Решение:

```text
не позиционировать работу как "мы побили TrOCR на чистом IAM";
позиционировать как:
    structural representation,
    interpretability,
    robustness,
    low-resource transfer,
    Cyrillic analysis.
```

## Проблема 7. Disentanglement сложен

Разделить writer-core, condition, time и content трудно.

Решение:

```text
начать с простого:
    content vs graph features;
затем добавить:
    writer contrastive;
потом:
    condition augmentation;
и только после этого temporal modeling.
```

## Проблема 8. Риск переусложнения

Очень большой проект можно не завершить.

Решение: жёстко разделить на MVP и extended version.

---

# 17. MVP, который уже можно реализовать

Минимальная научно защищаемая версия:

```text
1. Classical CSG extractor
2. Structural metrics
3. Graph feature extractor
4. Raster + graph HTR fusion
5. Experiments on IAM + HWR200/HKR
6. Small manually corrected graph subset
```

MVP-вклад:

```text
canonical stroke graph representation;
pipeline image → graph → vector;
structural metrics;
proof that graph features improve robustness / Cyrillic transfer.
```

Не включать в MVP:

```text
полный temporal modeling;
имитацию чужого почерка;
сложную судебную диагностику;
полный differentiable renderer;
страничный layout на первом этапе.
```

---

# 18. Полная версия работы

Расширенная версия:

```text
HI-CSG-R
+ differentiable vectorization
+ handwriting-informed feature heads
+ stability/variation modeling
+ numeric handwriting mode
+ condition-aware augmentation
+ cross-script transfer
+ temporal robustness
```

Это уже уровень большой магистерской/аспирантской работы или серии статей.

---

# 19. Рекомендуемый план реализации

## Этап 0. Формализация

Сделать документ:

```text
definition of HI-CSG-R
node taxonomy
edge taxonomy
feature taxonomy
metrics taxonomy
dataset protocol
```

Результат:

```text
теоретический раздел + техническое ТЗ
```

---

## Этап 1. Данные и baseline OCR

Собрать:

```text
IAM
HWR200
HKR
Digital Peter, если хватит времени
```

Обучить / запустить baseline:

```text
CRNN + CTC
TrOCR fine-tune или pretrained baseline
```

Результат:

```text
baseline CER/WER по каждому датасету
```

---

## Этап 2. Classical CSG extractor

Реализовать:

```text
binarization
skeletonization
graph extraction
node classification
edge tracing
pruning
Bezier/polyline fitting
JSON/SVG export
```

Результат:

```text
для каждого изображения есть graph.json + visualization.svg
```

---

## Этап 3. Structural metrics

Реализовать:

```text
skeleton F1
junction F1
endpoint F1
loop score
Chamfer distance
graph statistics
render-back IoU
```

Результат:

```text
можно сравнивать алгоритмы векторизации
```

---

## Этап 4. Gold subset

Разметить вручную:

```text
100-200 слов EN
100-200 слов RU
50-100 цифровых/смешанных записей
```

Разметка:

```text
nodes
edges
loops
connection type
symbol label
```

Результат:

```text
маленький, но честный benchmark
```

---

## Этап 5. Neural structural heads

Обучить U-Net/Swin-like модель:

```text
image → skeleton map
image → keypoint maps
image → width map
image → tangent field
```

Результат:

```text
learned graph extractor устойчивее классического
```

---

## Этап 6. Graph encoder

Построить:

```text
node/edge feature matrix
GAT / Graph Transformer
graph pooling along x-axis
CTC decoder
```

Результат:

```text
graph-only HTR baseline
```

---

## Этап 7. Raster + graph fusion

Сравнить:

```text
image-only
graph-only
image + graph
image + graph + handwriting features
```

Результат:

```text
главная экспериментальная таблица
```

---

## Этап 8. Robustness experiments

Проверить:

```text
blur
noise
low contrast
broken strokes
thick strokes
skew
variable lighting
connected cursive
```

Результат:

```text
доказательство пользы графа
```

---

## Этап 9. Cross-script transfer

Проверить:

```text
train EN → RU
train RU → EN
train EN+RU → historical RU
```

Результат:

```text
доказательство language-agnostic stroke features
```

---

## Этап 10. Paper/report

Структура статьи:

```text
1. Introduction
2. Related Work
3. Handwriting-informed theory
4. HI-CSG-R representation
5. Method
6. Datasets and annotation
7. Experiments
8. Metrics
9. Discussion and limitations
10. Conclusion
```

---

# 20. Что должно быть в репозитории

```text
project/
  data/
    raw/
    processed/
    annotations/
  configs/
  src/
    preprocessing/
    skeleton/
    graph_builder/
    vectorization/
    features/
    models/
      image_baseline/
      graph_encoder/
      fusion/
    metrics/
    visualization/
  notebooks/
  experiments/
  outputs/
    graphs/
    svg/
    predictions/
  docs/
    representation_spec.md
    annotation_guidelines.md
    metrics.md
```

---

# 21. Практический стек

```text
Python
PyTorch
OpenCV
scikit-image
networkx
PyTorch Geometric или DGL
shapely / scipy для геометрии
svgwrite / CairoSVG для SVG
HuggingFace Transformers для TrOCR baseline
```

Для дифференцируемой векторизации:

```text
diffvg или собственный differentiable renderer
```

Но это лучше делать после классического и learned extractor.

---

# 22. Что делать в первую очередь

Самый правильный порядок:

```text
1. Зафиксировать HI-CSG-R schema.
2. Реализовать classical graph extractor.
3. Визуализировать 50-100 русских и английских примеров.
4. Посмотреть глазами, какие ошибки систематические.
5. Сформировать gold subset.
6. Сделать baseline OCR.
7. Проверить, помогают ли простые graph features.
8. Только потом строить сложную neural architecture.
```

Не надо начинать сразу с большой нейросети. Сначала нужно понять, **какой граф вообще получается из реальных рукописей**.

---

# 23. Критерии успеха

Работа успешна, если мы покажем хотя бы 3 результата:

## Результат 1

```text
HI-CSG-R даёт более качественную структурную реконструкцию,
чем naive skeleton graph / contour vectorization.
```

## Результат 2

```text
image + graph лучше image-only на сложных условиях:
noise, blur, Cyrillic cursive, historical data, low-resource setting.
```

## Результат 3

```text
почерковедческие признаки дают интерпретируемость:
модель может объяснить связность, петли, наклон, точки соединения,
вариативность и недостаточную информативность.
```

---

# 24. Что не стоит делать

Я бы не делал в первой версии:

```text
полное восстановление pen trajectory;
заявления о настоящем нажиме;
медицинские или психологические выводы;
полную судебную идентификацию автора;
подписи как основной объект;
страничный layout как главный фокус;
очень сложный differentiable vectorizer с нуля.
```

Подписи, диагностику и имитацию можно оставить как future work.

---

# 25. Финальная рекомендация

Самая сильная и реалистичная траектория:

```text
MVP:
    Classical + neural HI-CSG extraction
    + structural metrics
    + raster-graph fusion for HTR
    + RU/EN experiments

Advanced:
    differentiable vector rendering
    + stability/variation layer
    + numeric mode
    + condition-aware robustness
    + cross-script transfer
```

Формулировка проекта:

> **Мы разрабатываем почерковедчески-информированное графово-векторное представление офлайн-рукописи, которое сохраняет структуру видимых штрихов, топологию, связность, петли, точки соединения и вариативность признаков, а затем используем это представление для более интерпретируемого и робастного русско-английского HTR.**

Это достаточно сложно, ново и защищаемо: здесь есть и компьютерное зрение, и графовые модели, и векторизация, и HTR, и почерковедческая теория, и русско-английская специфика.

---

**Sources:**

- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282?utm_source=chatgpt.com)
- [Differentiable Vector Graphics Rasterization for Editing and ...](https://people.csail.mit.edu/tzumao/diffvg/?utm_source=chatgpt.com)
- [Bézier Curve-Based Stroke Extraction for Handwritten ...](https://bmvc2025.bmva.org/proceedings/478/?utm_source=chatgpt.com)
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database?utm_source=chatgpt.com)
- [AntiplagiatCompany/HWR200 · Datasets at ...](https://huggingface.co/datasets/AntiplagiatCompany/HWR200?utm_source=chatgpt.com)
- [Digital Peter: New Dataset, Competition and Handwriting ...](https://ar5iv.labs.arxiv.org/html/2103.09354?utm_source=chatgpt.com)

## Prompt:
24/05/2026, 13:26:05

прочти еще раз всё что мы продумали - все наши рассуждения и проведи еще раз полный анализ данных подходов, что можно достичь, какие проблемы, возможно что-то еще новое можно предложить - тебе нужно реально think out of box но будь осторожен в мыслях - не запутайся в них и проработай каждую мелочь, в итоге жду анализ наших наработок на данный момент, а затем сразу отдельно готовое конечное решение - план

## Response:
24/05/2026, 13:26:06

Я перечитаю это как единый проект, а не как набор идей: отделю то, что реально даст научный вклад, от того, что красиво звучит, но перегрузит реализацию. Затем сформулирую финальную версию решения и пошаговый план.

> **Researching and preparing a comprehensive solution plan**
>
> I need to answer in Russian, and it seems I should use file search to review past insights, along with current knowledge for accuracy. I’ll also check for up-to-date information on topics like machine learning, vectorization, and graph transformers for 2025/2026, while ensuring proper citations. The user asked about our work approach, so I’ll compile a complete, clear solution without going overboard.
>
> **Looking up more info**
>
> I need to get some information from the web right now. Time to search for the latest details or updates related to the task. I’ll narrow down the focus and find what’s most relevant before moving to the next part!
>
> **Searching for 2025 offline handwriting recognition papers**
>
>
> **Considering citations and referencing PDFs**
>
> User didn't explicitly ask for web citations, but it might still be helpful. I could reference a 2026 IEEE survey or IJCAI 2025 for context on graph structures and emerging HTR trends. If I cite PDFs, I need to use the screenshot tool to capture relevant portions.
>
> **Exploring citation requirements for PDFs and surveys**
>
>
> **Exploring official datasets**
>
> It might be a good idea to search for official datasets to help answer the user’s request. These could provide accurate and up-to-date information. I’ll look for reliable sources that could offer relevant datasets to back up my response or clarify any details I might need.
>
> **Searching for official handwriting recognition datasets**
>
>
> **Structuring the analysis for clarity**
>
> I need to provide a thorough answer, with structured sections and concise content as per the user’s request. I'll include key elements like the core vs. satellites distinction, scientific deliverables, risks, and out-of-the-box proposals like factorized hierarchy and graph grammar. I'll also provide a final decision on the architecture and detailed plan with tasks and milestones. Citations will be used for relevant sources.
>
> Thought for 42s

Ниже я делаю **сводный аудит всей нашей текущей концепции** и затем отдельно даю **готовое конечное решение - план реализации**. Главное: я сейчас не расширяю проект бесконечно, а наоборот - отделяю **ядро, которое надо делать**, от **интересных, но опасных ответвлений**.

---

# Часть I. Анализ наших наработок на данный момент

## 1. Что мы на самом деле строим

Изначальная идея была:

```text
скан рукописи → структурная векторизация / графовое штриховое представление → распознавание
```

После изучения технических материалов и почерковедческой литературы она стала сильнее:

```text
скан / фото рукописи
→ видимая штриховая структура
→ канонический граф штрихов
→ почерковедчески-информированные признаки
→ распознавание текста + оценка качества / устойчивости / неопределенности
```

Финально это уже не просто OCR и не просто stroke extraction. Это:

> **почерковедчески-информированная графово-векторная модель офлайн-рукописи.**

Рабочее название:

```text
HI-CSG-R
Handwriting-Informed Robust Canonical Stroke Graph
```

или по-русски:

```text
почерковедчески-информированный робастный канонический граф штрихов
```

---

## 2. Почему тема сильная

Есть явный научный зазор. Современные HTR-модели хорошо распознают текст, но обычно работают как `image → text`, без явной структурной модели письма. Свежие обзоры HTR фиксируют переход от эвристических методов к современным нейросетевым архитектурам и end-to-end document-level подходам, но это не закрывает задачу интерпретируемой графово-векторной структуры рукописи. ([arXiv](https://arxiv.org/html/2502.08417v1?utm_source=chatgpt.com))

С другой стороны, структурная векторизация офлайн-рукописей обычно распадается на отдельные шаги: скелетизация, построение графа, выделение штрихов, аппроксимация кривыми. В твоём исполнительном резюме уже зафиксированы главные пробелы: нет массовых stroke-аннотаций, нет устоявшихся метрик качества штрихов, мало работ по русской кириллице, а графовые модели плохо масштабированы на полноценный рукописный текст. `Исполнительное резюме.pdf`

Наша ниша находится между четырьмя областями:

```text
offline HTR
+ stroke/vector representation
+ graph neural networks
+ handwriting forensics / почерковедение
```

Это хорошая позиция: работа не будет выглядеть как “ещё один fine-tune TrOCR”.

---

## 3. Главный методологический принцип

Самое важное решение, которое мы уже приняли:

> **не восстанавливать реальную траекторию пера.**

Мы строим не `pen trajectory`, а:

```text
canonical visible stroke graph
```

То есть модель восстанавливает **видимую структуру письма**, а не утверждает, что знает реальный порядок движений руки.

Это очень важно, потому что в офлайн-скане порядок пера часто принципиально неоднозначен. В работах по pen trace reconstruction из скелета задача формулируется как реконструкция последовательного trace из статического изображения; такие подходы полезны, но требуют сильного предположения о порядке движения. ([CEUR Workshop Proceedings](https://ceur-ws.org/Vol-2744/paper27.pdf))

Наш путь осторожнее и научно чище:

```text
не "как двигалось перо"
а
"какая структурная штриховая форма наблюдается в изображении"
```

---

## 4. Что даёт почерковедение

Почерковедение дало нам не “юридическую экспертизу”, а теорию признаков.

Серегин описывает письменно-двигательный ФДК как систему, где в рукописи отражаются три потока информации: структурно-геометрическая, координационно-скоростно-ритмическая и нажимная. `Seregin_Pocherkovedenie_2015.pdf`

Жижина уточняет, что почерк основан на письменно-двигательном ФДК, включающем зрительно-двигательный образ и развернутую систему движений, приспособленную для его реализации. `Zhizhina_M_V_Pocherkovedenie_2006.docx`

Отсюда следует наша архитектурная схема:

```text
1. structural-geometric stream
   форма, петли, соединения, наклон, размер, разгон, топология

2. coordination-rhythm stream
   плавность, связность, ритм, темп-прокси, устойчивость движения

3. pressure-width stream
   ширина штриха, плотность, локальные утолщения, pressure proxy
```

Мы не должны называть третий поток “нажимом” в буквальном смысле, потому что в скане он виден только косвенно. Правильнее:

```text
pressure proxy
```

---

## 5. Самый важный теоретический вывод

Почерк не является фиксированным шаблоном. Он имеет:

```text
индивидуальность
динамическую устойчивость
вариационность
избирательную изменчивость
```

Серегин подчёркивает: динамическая устойчивость не означает абсолютной одинаковости; она проявляется через вариационность и разброс признаков. `Seregin_Pocherkovedenie_2015.pdf`

Это меняет саму логику модели.

Плохая модель:

```text
ищет один "правильный" граф буквы
```

Хорошая модель:

```text
ищет семейство допустимых вариантов графа
и оценивает устойчивость каждого признака
```

То есть нам нужен не один canonical graph, а:

```text
canonical graph + variation field
```

---

## 6. Что мы можем реально достичь

### Реалистично достижимо в первой большой работе

Мы можем достичь четырёх сильных результатов.

**Первое:** формально определить новый тип представления:

```text
HI-CSG-R = graph + vector curves + handwriting-informed attributes
```

**Второе:** сделать работающий конвейер:

```text
image → skeleton → graph → strokes → Bézier/polyline → features → HTR
```

Это соответствует уже известному техническому конвейеру структурной векторизации, но мы расширяем его почерковедческими признаками и русско-английской проверкой. `Исполнительное резюме.pdf`

**Третье:** показать, что графовая ветка помогает не обязательно на чистых данных, а в сложных режимах:

```text
шум
размытие
русская скоропись
малые данные
исторические документы
cross-script transfer
```

**Четвёртое:** создать малый benchmark структурной векторизации:

```text
200-500 слов / фрагментов
с ручной или полуавтоматической разметкой:
nodes, edges, loops, junctions, connections
```

Это особенно важно, потому что отсутствие stroke-аннотаций - один из главных пробелов области. `Исполнительное резюме.pdf`

---

## 7. Что достижимо, но не в первой версии

Некоторые идеи сильные, но их нельзя тащить в MVP.

К таким относятся:

```text
полное моделирование подражания чужому почерку
полная диагностика необычных условий
возраст / пол / психологические свойства
подписи как основной объект
страничный layout как главный фокус
полный differentiable renderer с нуля
настоящий pen-order recovery
```

Особенно нужно быть осторожными с диагностикой. Мы можем использовать понятия “condition anomaly”, “execution instability”, “deautomation-like artifacts”, но не должны делать судебные, медицинские или психологические выводы.

---

# 8. Главная архитектурная развилка

У нас есть пять возможных архитектурных линий.

## A. Classical CSG extractor

```text
image
→ binarization
→ skeletonization
→ graph extraction
→ pruning
→ stroke grouping
→ vector fitting
```

Это обязательно.

Плюсы:

```text
быстро реализуется
даёт визуальные результаты
даёт pseudo-labels
интерпретируемо
```

Минусы:

```text
ломается на шуме
чувствительно к бинаризации
трудно на пересечениях и толстых штрихах
много эвристик
```

Вывод: **делать первым, но не считать финальной моделью.**

---

## B. Neural structural extractor

```text
image
→ CNN/Swin/U-Net encoder
→ skeleton map
→ endpoint map
→ junction map
→ loop map
→ tangent field
→ width map
→ graph builder
```

Это лучший второй этап.

Плюсы:

```text
устойчивее классической скелетизации
можно учить на pseudo-labels
можно добавлять confidence
```

Минусы:

```text
нужны хорошие pseudo-labels
graph builder всё равно нужен
модель может копировать ошибки скелетизации
```

Вывод: **это должна быть основа основной модели.**

---

## C. Differentiable vectorizer

```text
image
→ predicted Bézier curves
→ differentiable rasterizer
→ reconstructed image
→ loss
```

DiffVG показывает, что differentiable rasterization может связывать векторные параметры и raster loss, то есть это технически подходящий механизм для обучения векторного представления без полной ручной разметки. ([people.csail.mit.edu](https://people.csail.mit.edu/tzumao/diffvg/?utm_source=chatgpt.com))

Плюсы:

```text
очень красиво научно
настоящий vector output
можно учить через reconstruction
```

Минусы:

```text
сложная оптимизация
трудно масштабировать на строку
много локальных минимумов
может уйти в "красивую реконструкцию", но не в правильный граф
```

Вывод: **не MVP. Делать как advanced module после классического и neural extractor.**

---

## D. Graph-only HTR

```text
CSG
→ Graph Transformer / GAT
→ CTC / decoder
→ text
```

Плюсы:

```text
показывает, насколько граф самодостаточен
хорошо для абляций
```

Минусы:

```text
может проиграть raster models
сложно упорядочивать граф строки
графы могут быть большими
```

Вывод: **делать как эксперимент, но не как главную модель.**

---

## E. Raster + Graph fusion

Это главный вариант.

```text
image encoder ───────┐
                     ├→ fusion → text decoder
graph encoder ───────┘
```

Свежая IJCAI-2025 работа по graph-enhanced cross-modal mutual learning прямо мотивируется тем, что обычные HTR-методы фокусируются на визуальных пиксельных паттернах и недоиспользуют геометрическую структуру символов. ([ijcai.org](https://www.ijcai.org/proceedings/2025/0574.pdf))

Плюсы:

```text
image branch видит пиксельные детали
graph branch видит структуру
можно честно сравнить image-only / graph-only / image+graph
научно доказуемая польза
```

Минусы:

```text
дороже
сложнее отлаживать
граф может не дать прироста на чистом IAM
```

Вывод: **это финальная основная архитектура.**

---

# 9. Важное новое предложение: не один граф, а три графа

Вот out-of-box, но аккуратная идея.

Один HI-CSG может быть перегружен. Лучше хранить три связанных уровня:

```text
1. Raw Skeleton Graph
   что реально извлечено из изображения

2. Clean Canonical Stroke Graph
   очищенная векторная структура видимых штрихов

3. Handwriting Feature Graph
   граф признаков: связность, петли, наклон, разгон, варианты, устойчивость
```

То есть:

```text
image
→ raw graph
→ canonical graph
→ feature graph
→ text / analysis
```

Почему это важно: если граф ошибся, мы сможем понять, где именно ошибка.

```text
ошибка скелетизации?
ошибка группировки штрихов?
ошибка интерпретации признаков?
ошибка OCR?
```

Это сильно повысит отлаживаемость и научную прозрачность.

---

# 10. Важное новое предложение: graph uncertainty

Нужно не просто предсказывать узлы и рёбра, а хранить уверенность.

```json
{
  "edge": 17,
  "curve": "...",
  "confidence": 0.82,
  "topology_confidence": 0.74,
  "stroke_group_confidence": 0.61,
  "feature_reliability": 0.55
}
```

Почему это важно:

```text
скелетизация часто неоднозначна;
петли могут быть залиты;
соединения могут быть ложными;
пересечения могут быть не реальными crossing, а наложением штрихов;
нажим и темп видны только косвенно.
```

Это особенно нужно для научной честности.

---

# 11. Важное новое предложение: stroke graph grammar

Самая перспективная “нестандартная” идея: сделать не просто граф, а **грамматику штриховых блоков**.

Вместо того чтобы каждая буква была отдельным классом, модель должна учить повторяющиеся блоки:

```text
oval block
vertical stem block
upper loop block
lower descender block
connector block
terminal hook block
crossing block
```

Это хорошо ложится на почерковедческие “общие / частные / особенные” признаки. Жижина выделяет общесистемные и частносистемные признаки, включая структурно-геометрические, динамические и пространственно-ориентационные; структурно-геометрические признаки включают форму, направление, размер, разгон, наклон и связность. `Zhizhina_M_V_Pocherkovedenie_2006.docx`

Для кириллицы это особенно ценно:

```text
а / о / ю / ф → oval group
д / у / р / ц / щ → lower-zone group
б / й → upper-zone group
и / н / п / м / ш / щ → repeated stem-connector group
```

Это даст модели переносимость между буквами и языками.

---

# 12. Важное новое предложение: allograph inventory

Нужно хранить не просто признак, а распределение вариантов.

```json
{
  "letter": "д",
  "variants": {
    "looped_descender": 0.72,
    "straight_descender": 0.18,
    "simplified": 0.10
  }
}
```

Почему это важно: особенная часть подчёркивает, что при сходных почерках важно не только наличие признаков, но и частота вариантов, особенно когда различия тонкие. В цифровом почерке также подчёркивается необходимость искать “признаки в признаках” и детализировать признаки в элементах и штрихах. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

Это можно превратить в сильную идею:

```text
variant-frequency-aware stroke graph representation
```

---

# 13. Важное новое предложение: graph tokenizer

Для HTR можно не подавать весь граф в GNN целиком. Можно токенизировать граф.

Например:

```text
edge token:
    shape=arc
    zone=middle
    direction=rightward
    width=medium
    curvature=low
    connection=continuous

block token:
    oval_closed
    lower_loop
    terminal_hook
    vertical_stem
```

Тогда строка превращается в последовательность графовых токенов:

```text
[stem, connector, oval, loop, terminal, ...]
```

И уже это можно подавать в Transformer/CTC.

Это решает проблему масштабируемости: большие графы строк можно сжимать в последовательность структурных токенов.

---

# 14. Важное новое предложение: topology-first evaluation

Обычные метрики типа CER/WER недостаточны. Нужно сделать главную особенность статьи - структурные метрики.

Минимальный набор:

```text
Skeleton F1
Endpoint F1
Junction F1
Loop preservation
Component consistency
Graph edit distance
Chamfer distance
Raster reconstruction IoU
Bézier fitting error
```

Но я бы добавил ещё одну метрику:

```text
Critical Topology Error
```

Она штрафует не все ошибки одинаково, а особенно:

```text
потеря петли
ложное соединение букв
разрыв связного соединения
исчезновение подстрочного элемента
ложный crossing
```

Почему это важно: OCR может всё равно распознать слово правильно, но структурный граф будет плохим. Наша работа должна уметь это показать.

---

# 15. Важное новое предложение: counterfactual graph normalization

Это более рискованная, но интересная идея.

Модель может учиться отделять:

```text
writer-core
от
condition distortion
```

Например, граф содержит:

```text
сильный наклон
дрожание
разрывистость
тупые окончания
```

Модель строит две версии:

```text
observed graph
normalized graph
```

Где normalized graph - не “исправленный почерк”, а структурная гипотеза без ситуативных искажений.

Но это опасная идея для первой версии. Её можно оставить как future work.

---

# 16. Важное новое предложение: dual image policy

Нельзя использовать одну предобработанную картинку для всего.

Для OCR полезно:

```text
deskew
contrast normalization
denoise
line normalization
```

Но для почерковедческих признаков агрессивная нормализация может уничтожить:

```text
наклон
размер
разгон
нажим-прокси
baseline instability
spacing
```

Значит, нужно вести две версии:

```text
OCR-normalized image
Forensic-preserving image
```

Первая идёт в HTR branch. Вторая - в graph/feature branch.

Это простая, но очень важная инженерная идея.

---

# 17. Данные: что реально использовать

Для английского - IAM. Он содержит 657 авторов, 1 539 страниц, 13 353 размеченные строки и 115 320 слов. ([Fki Tic](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database?utm_source=chatgpt.com))

Для русского - HWR200: русские рукописные тексты от 200 авторов, сфотографированные в разных условиях. ([Hugging Face](https://huggingface.co/datasets/AntiplagiatCompany/HWR200?utm_source=chatgpt.com))

Для кириллицы большего масштаба - HKR: около 63 000 предложений, более 715 699 символов, примерно 200 авторов; около 95% русского и 5% казахского текста. ([arXiv](https://arxiv.org/abs/2007.03579?utm_source=chatgpt.com))

Для исторического русского stress-test - Digital Peter: 9 694 изображений строк с текстовыми файлами. ([arXiv](https://arxiv.org/abs/2103.09354?utm_source=chatgpt.com))

Роль каждого датасета:

```text
IAM:
    английский baseline, чистый benchmark

HWR200:
    русский современный, разные условия фото

HKR:
    масштабная кириллица, предобучение / fine-tune

Digital Peter:
    исторический domain shift, robustness test
```

---

# 18. Главные проблемы

## Проблема 1. Нет stroke ground truth

Это главный риск.

Решение:

```text
classical pseudo-labels
+ small manual gold subset
+ synthetic vector data
+ self-supervised raster reconstruction
+ downstream HTR validation
```

---

## Проблема 2. Скелетизация нестабильна

Особенно на:

```text
толстых штрихах
плохом фото
размытых краях
залитых петлях
пересечениях
старой бумаге
```

Решение:

```text
multi-threshold skeleton ensemble
learned skeleton map
width-aware pruning
uncertainty per edge
manual correction subset
```

---

## Проблема 3. Граф может стать слишком большим

Решение:

```text
line → word → symbol/block → stroke hierarchy
graph pooling
graph tokenizer
subgraph batching
```

---

## Проблема 4. Graph branch может не улучшить clean OCR

Это нормально.

Не нужно обещать:

```text
мы побьём TrOCR на чистом IAM
```

Лучше позиционировать так:

```text
мы улучшаем интерпретируемость, устойчивость, переносимость и структурное качество
```

---

## Проблема 5. Почерковедческие признаки не все наблюдаемы

Нельзя говорить:

```text
мы восстановили нажим
мы восстановили темп
мы восстановили состояние человека
```

Нужно говорить:

```text
pressure proxy
tempo proxy
execution irregularity proxy
condition-sensitive features
```

---

## Проблема 6. Проект может расползтись

Поэтому нужно жёстко разделить:

```text
ядро
расширения
future work
```

---

# 19. Что оставить в ядре

В ядре должны быть только те вещи, которые прямо нужны для научного результата.

```text
1. HI-CSG representation
2. Classical graph extractor
3. Neural structural extractor
4. Structural metrics
5. Raster + graph fusion HTR
6. RU/EN experiments
7. Small graph annotation benchmark
8. Robustness tests
```

---

# 20. Что оставить как расширение

```text
1. Differentiable vector rendering
2. Numeric handwriting mode
3. Allograph inventory
4. Variant frequency modeling
5. Graph tokenizer
6. Condition-aware augmentation
```

---

# 21. Что оставить как future work

```text
1. Подражание чужому почерку
2. Подписи
3. Возраст / пол / психологические свойства
4. Судебная идентификация автора
5. Полный temporal modeling
6. Полное восстановление pen order
7. Full-page document graph
```

---

# Часть II. Готовое конечное решение - план

## Финальная тема

```text
HI-CSG-R: Handwriting-Informed Robust Canonical Stroke Graphs
for Offline Russian-English Handwritten Text Recognition
```

Русская формулировка:

> **Почерковедчески-информированная робастная каноническая графово-векторная реконструкция штрихов для офлайн-распознавания русско-английских рукописей.**

---

## Финальная гипотеза

> Если офлайн-рукопись представить не только как изображение, а как почерковедчески-информированный канонический граф видимых штрихов, учитывающий форму, топологию, связность, петли, точки соединения, вариативность и неопределённость признаков, то такое представление улучшит интерпретируемость и устойчивость HTR, особенно на кириллице, шумных данных, малых выборках и cross-script переносе.

---

## Финальная архитектура

```text
Input handwriting image
        ↓
Dual preprocessing
        ├── OCR-normalized image
        └── feature-preserving image
        ↓
Visual encoder
        ↓
Dense structural heads
        ├── skeleton map
        ├── stroke width map
        ├── endpoint heatmap
        ├── junction heatmap
        ├── crossing heatmap
        ├── loop heatmap
        ├── tangent field
        └── baseline / zones
        ↓
Raw Skeleton Graph
        ↓
Canonical Stroke Graph Builder
        ↓
HI-CSG-R
        ↓
Graph encoder / graph tokenizer
        ↓
Raster + graph fusion
        ↓
Outputs
        ├── recognized text
        ├── vector graph
        ├── structural features
        ├── graph confidence
        └── informativeness score
```

---

## Формат HI-CSG-R

```json
{
  "raw_graph": {
    "nodes": [],
    "edges": []
  },
  "canonical_graph": {
    "nodes": [
      {
        "id": 1,
        "type": "endpoint | junction | crossing | loop_closure | curvature_extreme",
        "x": 0.42,
        "y": 0.37,
        "confidence": 0.91
      }
    ],
    "edges": [
      {
        "id": 12,
        "from": 1,
        "to": 2,
        "curve_type": "polyline | bezier",
        "control_points": [],
        "length": 24.1,
        "curvature": 0.37,
        "width_mean": 2.8,
        "width_std": 0.4,
        "smoothness": 0.82,
        "connection_type": "continuous | interval | uncertain",
        "confidence": 0.76
      }
    ]
  },
  "handwriting_features": {
    "general": {
      "slant": {},
      "size": {},
      "expansion": {},
      "connectivity": {},
      "baseline": {}
    },
    "special_blocks": {
      "oval_blocks": {},
      "descender_blocks": {},
      "connector_blocks": {}
    },
    "private": {
      "start_candidates": {},
      "end_candidates": {},
      "junction_positions": {},
      "loop_shapes": {},
      "terminal_shapes": {}
    }
  },
  "quality": {
    "graph_quality": 0.81,
    "topology_confidence": 0.73,
    "informativeness": 0.68
  }
}
```

---

# Этап 0. Спецификация

Сначала нужно написать техническую спецификацию, без кода.

## Нужно зафиксировать

```text
1. Типы узлов
2. Типы рёбер
3. Типы блоков
4. JSON-схему графа
5. Список почерковедческих признаков
6. Список структурных метрик
7. Правила визуализации
8. Правила ручной разметки
```

## Результат этапа

```text
docs/HI_CSG_R_spec.md
docs/annotation_guidelines.md
docs/metrics.md
```

Это фундамент. Без него проект расползётся.

---

# Этап 1. Данные

## Скачать и подготовить

```text
IAM
HWR200
HKR
Digital Peter
```

## Привести к единому виду

```text
image_id
writer_id
language
script
line_image
transcription
dataset_name
split
```

## Результат

```text
data/processed/
    iam/
    hwr200/
    hkr/
    digital_peter/
```

---

# Этап 2. Baseline OCR

Сначала нужен обычный HTR baseline.

## Модели

```text
CRNN + CTC
Transformer/TrOCR-like baseline
```

## Метрики

```text
CER
WER
line accuracy
```

## Зачем

Чтобы потом честно показать:

```text
image-only
vs
graph-only
vs
image+graph
```

---

# Этап 3. Classical CSG extractor

Это первый основной технический модуль.

## Pipeline

```text
image
→ grayscale
→ binarization
→ denoise
→ skeletonization
→ pixel graph
→ node detection
→ edge tracing
→ pruning
→ polyline / Bézier fitting
→ JSON + SVG
```

## Выход

```text
outputs/graphs/sample_001.json
outputs/svg/sample_001.svg
```

## Что проверить глазами

```text
петли
пересечения
разрывы
ложные ветки
точки соединения
подстрочные элементы
русская связность
```

---

# Этап 4. Structural metrics

Нужно сразу сделать метрики, иначе непонятно, улучшается ли модель.

## Метрики

```text
Skeleton F1
Endpoint F1
Junction F1
Loop preservation
Component consistency
Chamfer distance
Graph edit distance
Raster reconstruction IoU
Bézier fitting error
Critical Topology Error
```

## Critical Topology Error

Особо штрафовать:

```text
потерю петли
ложное соединение букв
разрыв связного соединения
исчезновение нижнего/верхнего элемента
ложный crossing
```

---

# Этап 5. Малый gold subset

Это обязательно.

## Объём

```text
100-200 английских слов
100-200 русских слов
50-100 цифровых / смешанных записей
```

## Разметка

```text
nodes
edges
loops
junctions
crossings
connection type
letter/digit region
```

## Зачем

Без gold subset мы не докажем качество графовой реконструкции.

---

# Этап 6. Neural structural extractor

Теперь обучаем нейросеть строить карты.

## Вход

```text
feature-preserving image
```

## Выходы

```text
skeleton map
endpoint heatmap
junction heatmap
crossing heatmap
loop heatmap
width map
tangent field
baseline/zones
```

## Loss

```text
L_skeleton
+ L_endpoint
+ L_junction
+ L_loop
+ L_width
+ L_tangent
```

## Источники supervision

```text
classical pseudo-labels
gold subset
synthetic data
```

---

# Этап 7. Graph builder v2

Использовать не только skeleton, а neural maps.

```text
skeleton map + keypoint maps + width map
→ graph extraction
→ confidence-aware pruning
→ canonical stroke graph
```

Новая фишка:

```text
каждое ребро и узел получают confidence
```

---

# Этап 8. Handwriting feature extractor

Считать признаки из графа.

## Общие признаки

```text
baseline slope
baseline waviness
slant
size
expansion
spacing
connectivity
component count
tempo proxy
pressure proxy
coordination proxy
```

## Особенные блоки

```text
oval block
upper loop block
lower descender block
connector block
terminal hook
```

## Частные признаки

```text
форма элемента
направление
протяжённость
точка соединения
точка пересечения
начальный штрих
заключительный штрих
```

---

# Этап 9. Graph encoder

## Вариант 1 - простой

```text
graph features
→ MLP / LightGBM
→ auxiliary predictions
```

## Вариант 2 - основной

```text
node/edge graph
→ GAT / Graph Transformer
→ graph tokens
```

## Вариант 3 - масштабируемый

```text
graph
→ graph tokenizer
→ sequence of structural tokens
→ Transformer / CTC
```

---

# Этап 10. Raster + graph fusion

Сравнить четыре модели:

```text
1. image-only
2. graph-only
3. image + graph
4. image + graph + handwriting features
```

## Fusion варианты

Начать с простого:

```text
late fusion
```

Потом:

```text
cross-attention fusion
```

---

# Этап 11. Robustness experiments

Проверить на искажениях:

```text
blur
noise
low contrast
broken strokes
thick strokes
uneven lighting
skew
compression artifacts
```

Сравнить:

```text
CER degradation
graph quality degradation
confidence calibration
```

Главный ожидаемый результат:

```text
graph branch меньше деградирует на структурных искажениях,
а image+graph стабильнее image-only.
```

---

# Этап 12. Cross-script experiments

Проверить:

```text
train EN → test RU
train RU → test EN
train EN+RU → test RU
train RU modern → test Digital Peter
```

Цель:

```text
показать, что часть stroke primitives переносится между латиницей и кириллицей.
```

---

# Этап 13. Ablation study

Убрать по очереди:

```text
без graph branch
без handwriting features
без loop features
без width map
без tangent field
без confidence
без special blocks
без topology loss
```

Это покажет, что именно даёт вклад.

---

# Этап 14. Финальные таблицы статьи

Нужны четыре главные таблицы.

## Таблица 1. OCR

```text
model | IAM CER | HWR200 CER | HKR CER | Digital Peter CER
```

## Таблица 2. Structural quality

```text
method | Skeleton F1 | Junction F1 | Loop score | IoU | Critical topology error
```

## Таблица 3. Robustness

```text
model | clean | blur | noise | low contrast | broken strokes
```

## Таблица 4. Ablation

```text
full model
- graph
- special blocks
- topology loss
- width/tangent
- confidence
```

---

# Этап 15. Что должно быть в статье

## Структура

```text
1. Introduction
2. Related Work
3. Handwriting-Informed Theory
4. HI-CSG-R Representation
5. Method
6. Datasets and Annotation
7. Metrics
8. Experiments
9. Discussion
10. Limitations
11. Conclusion
```

---

# Итоговое решение в одном абзаце

Мы строим **HI-CSG-R** - почерковедчески-информированное графово-векторное представление офлайн-рукописи. Оно извлекает из изображения не реальную траекторию пера, а каноническую структуру видимых штрихов: узлы, рёбра, петли, соединения, пересечения, кривые, ширину, наклон, связность и вариативность признаков. Далее это представление используется вместе с raster branch в гибридной HTR-модели. Научный вклад состоит в формальном представлении HI-CSG-R, структурных метриках, малом benchmark-е графовой разметки, русско-английской проверке и доказательстве, что графовая структура повышает интерпретируемость и робастность распознавания.

---

# Самый короткий практический план

```text
1. Зафиксировать HI-CSG-R schema.
2. Подготовить IAM, HWR200, HKR, Digital Peter.
3. Сделать baseline OCR.
4. Реализовать classical skeleton → graph → vector extractor.
5. Сделать визуализацию graph/SVG.
6. Ввести structural metrics.
7. Разметить малый gold subset.
8. Обучить neural structural extractor.
9. Построить graph encoder.
10. Сделать raster + graph fusion.
11. Провести robustness и cross-script эксперименты.
12. Провести ablation.
13. Написать статью.
```

Финальное решение: **идти не в сторону “идеального восстановления пера”, а в сторону интерпретируемого, устойчивого, почерковедчески-информированного графового слоя между изображением и текстом.** Это самая сильная и реалистичная версия проекта.

---

**Sources:**

- [Handwritten Text Recognition: A Survey](https://arxiv.org/html/2502.08417v1?utm_source=chatgpt.com)
- [](https://ceur-ws.org/Vol-2744/paper27.pdf)
- [Differentiable Vector Graphics Rasterization for Editing and ...](https://people.csail.mit.edu/tzumao/diffvg/?utm_source=chatgpt.com)
- [](https://www.ijcai.org/proceedings/2025/0574.pdf)
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database?utm_source=chatgpt.com)
- [AntiplagiatCompany/HWR200 · Datasets at ...](https://huggingface.co/datasets/AntiplagiatCompany/HWR200?utm_source=chatgpt.com)

## Prompt:
24/05/2026, 14:47:20

окей реализуй теперь полный технический план со всеми этапами от нуля до MVP - все детали, всевозможные эксперименты с архитектурой и с реализацией сделай их как можно больше, и пиши как для человека который пока не знаком с нашей работой чтобы можно было открыть любую часть и не погружаясь в контекст других делать её, проработай все что можно максимально подробно, все что ты тут писал надо использовать

## Response:
24/05/2026, 14:47:21

> **Planning next steps for technical response**
>
> It seems like the task might not require new facts but could involve tools or datasets relevant to my task. I do need to make sure I cite anything used, especially if I pull from prior references. Since the user requests specifics, I might run an exact search but probably don't need to pull more from the web at this point.
>
> **Planning technical response and citing sources**
>
> If I use sources from the web, I should definitely look for official datasets and cite them properly. I think referencing previous source IDs might be helpful, if available, but I don't need to go overboard. The user requested a detailed technical plan, so I’ll aim for thoroughness with manageable verbosity.
>
> **Searching for current sources on text recognition and datasets**
>
> [TrOCR: Transformer-Based Optical Character Recognition ...](https://ojs.aaai.org/index.php/AAAI/article/view/26538)
> [Differentiable Vector Graphics Rasterization for Editing and ...](https://people.csail.mit.edu/tzumao/diffvg/)
> [Samples of handwritten text from the IAM database 3](https://www.researchgate.net/figure/Samples-of-handwritten-text-from-the-IAM-database-3_fig2_285582907)
> [HWR200: New open access dataset of handwritten texts ...](https://dialogue-conf.org/media/5925/potyashiniplusetal048.pdf)
> [TrOCR: Transformer-based OCR Model | PDF](https://www.scribd.com/document/538326096/2109-10282v3)
> [Bézier Splatting for Fast and Differentiable Vector Graphics](https://arxiv.org/html/2503.16424v2)
> [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
> [AntiplagiatCompany/HWR200 · Datasets at ...](https://huggingface.co/datasets/AntiplagiatCompany/HWR200)
> [BachiLi/diffvg: Differentiable Vector Graphics Rasterization](https://github.com/BachiLi/diffvg)
> [iam_handwriting_word_database](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)
> [HWR200: New open access dataset of handwritten texts ...](https://www.semanticscholar.org/paper/HWR200%3A-New-open-access-dataset-of-handwritten-in-Potyashin-Kaprielova/661f05f2506892532d1443681c151112dea69612)
> [ML Benchmarks, SotA Results & Code - Papers With Code 2](https://paperswithcode2.com/dataset/10586)
> [TrOCR Transformer-based OCR for Handwritten Text using ...](https://www.youtube.com/watch?v=eoqH3WPMg0E)
> [Handwriting Text Recognition](https://medium.com/analytics-vidhya/handwriting-text-recognition-3712978249da)
> [82446](https://www.ipu.ru/node/82446)
> [TrOCR: Transformer-Based Optical Character Recognition ...](https://www.studocu.vn/vn/document/dai-hoc-quoc-gia-ha-noi/nhap-mon-xu-li-du-lieu-dinh-luong/trocr-transformer-based-optical-character-recognition-model/159467347)
> [Differentiable Vector Graphics Pipeline](https://www.emergentmind.com/topics/differentiable-vector-graphics-diffvg-pipeline)
>
> **Finalizing technical plan with proper citations**
>
> I need to structure the plan clearly and cite relevant sources, especially for claims made from web sources. Since it's a comprehensive technical plan, citations should focus on key points—like file summaries, specific research papers, and important datasets. The plan should be detailed but easy to follow, covering all key phases leading to an MVP.
>
> Thought for 27s

Ниже - **полный технический план от нуля до MVP**. Я пишу его как самостоятельное ТЗ: человек, который не знает нашего контекста, сможет открыть любой раздел и понять, что делать, зачем это нужно, какие входы/выходы, какие эксперименты провести и какие риски проверить.

---

# 0. Краткое описание проекта

## Название проекта

**HI-CSG-R: Handwriting-Informed Robust Canonical Stroke Graphs for Offline Russian-English Handwritten Text Recognition**

По-русски:

**Почерковедчески-информированная робастная каноническая графово-векторная реконструкция штрихов для офлайн-распознавания русско-английских рукописей.**

---

## Главная идея

Обычная HTR/OCR-модель делает:

```text
image → text
```

Мы делаем:

```text
image
→ видимая структура штрихов
→ canonical stroke graph
→ почерковедческие признаки
→ text + graph + confidence + structural explanation
```

Важно: мы **не восстанавливаем реальное движение пера**. Из скана/фото невозможно надёжно узнать настоящий порядок движений. Поэтому мы строим **канонический граф видимых штрихов**: узлы, рёбра, петли, соединения, пересечения, кривые, толщину, наклон, связность, зоны букв и признаки вариативности.

---

## Почему это научно оправдано

В почерковедении почерк рассматривается как результат письменно-двигательного функционально-динамического комплекса. Серегин выделяет три информационных потока, проявляющихся в рукописи: структурно-геометрический, координационно-скоростно-ритмический и нажимной. Это напрямую превращается в три группы признаков нашей модели: геометрия/топология, ритм/координация и pressure-width proxy. `Seregin_Pocherkovedenie_2015.pdf`

Жижина уточняет, что письмо - сложная системная деятельность, в которой есть зрительно-двигательный образ и система движений, а признаки почерка отражают свойства письменно-двигательного навыка. `Zhizhina_M_V_Pocherkovedenie_2006.docx`

Технический обзор показывает, что текущие методы структурной векторизации часто ограничиваются отдельными шагами - скелетизация, граф, сегментация штрихов, аппроксимация кривыми - и в области не хватает end-to-end нейро-структурных моделей, единых метрик и датасетов со stroke-level аннотациями. `Исполнительное резюме.pdf`

---

# 1. Что должно войти в MVP

MVP - это не “самая маленькая демка”. MVP должен быть достаточно сильным, чтобы на его основе можно было написать первую научную работу.

## В MVP входит

```text
1. Формальная спецификация HI-CSG-R.
2. Подготовка датасетов IAM + HWR200/HKR, опционально Digital Peter.
3. Baseline OCR/HTR.
4. Classical pipeline: image → skeleton → graph → vector.
5. Экспорт graph.json и SVG-визуализации.
6. Structural metrics.
7. Небольшой gold subset ручной/полуручной графовой разметки.
8. Neural structural extractor: image → skeleton/keypoints/width/tangent maps.
9. Graph builder v2 на основе neural maps.
10. Handwriting feature extractor.
11. Graph-only baseline.
12. Raster + graph fusion baseline.
13. Robustness experiments.
14. Cross-script experiments RU ↔ EN.
15. Ablation study.
16. Итоговый отчёт/статья.
```

## В MVP не входит

```text
1. Полное восстановление порядка пера.
2. Полная судебная идентификация автора.
3. Медицинские, психологические или юридические выводы.
4. Подписи как главный объект.
5. Полная диагностика состояния человека.
6. Полный differentiable renderer с нуля.
7. Полный page-level document understanding.
8. Полноценное моделирование подражания чужому почерку.
```

Это можно оставить как future work.

---

# 2. Ожидаемый конечный результат MVP

После MVP у нас должны быть:

```text
1. Код, который принимает изображение строки/слова рукописи.
2. Строит структурный граф штрихов.
3. Сохраняет graph.json.
4. Сохраняет SVG/PNG-визуализацию графа поверх изображения.
5. Считает структурные признаки.
6. Распознаёт текст image-only baseline.
7. Распознаёт текст graph-only baseline.
8. Распознаёт текст image+graph fusion.
9. Сравнивает модели по CER/WER.
10. Сравнивает качество графа по структурным метрикам.
11. Показывает, где графовая ветка помогает: шум, blur, кириллица, малые данные, исторический домен.
```

---

# 3. Общая архитектура MVP

```text
Input handwriting image
        ↓
Dual preprocessing
        ├── OCR-normalized image
        └── feature-preserving image
        ↓
Classical / Neural structural extractor
        ↓
Raw Skeleton Graph
        ↓
Canonical Stroke Graph Builder
        ↓
HI-CSG-R graph
        ↓
Handwriting feature extractor
        ↓
Graph encoder / graph tokenizer
        ↓
Raster + graph fusion
        ↓
Outputs:
        ├── recognized text
        ├── graph.json
        ├── graph.svg / overlay.png
        ├── structural metrics
        ├── graph confidence
        └── informativeness score
```

---

# 4. Репозиторий проекта

Структура репозитория должна быть такой:

```text
hi-csg-r/
  README.md
  pyproject.toml
  requirements.txt

  configs/
    datasets/
      iam.yaml
      hwr200.yaml
      hkr.yaml
      digital_peter.yaml
    preprocessing/
      default.yaml
      preserve_features.yaml
      ocr_normalized.yaml
    graph/
      skeleton_baseline.yaml
      graph_builder.yaml
      pruning.yaml
      bezier.yaml
    models/
      crnn_ctc.yaml
      trocr_baseline.yaml
      neural_structural_extractor.yaml
      graph_encoder.yaml
      fusion.yaml
    experiments/
      exp_001_classical_graph.yaml
      exp_002_baseline_ocr.yaml
      exp_003_graph_only.yaml
      exp_004_fusion.yaml
      exp_005_robustness.yaml
      exp_006_cross_script.yaml

  data/
    raw/
    processed/
    splits/
    annotations/
    pseudo_labels/
    gold_subset/

  docs/
    00_project_overview.md
    01_hi_csg_r_schema.md
    02_annotation_guidelines.md
    03_metrics.md
    04_experiment_protocol.md
    05_risk_register.md

  src/
    datasets/
      registry.py
      iam.py
      hwr200.py
      hkr.py
      digital_peter.py

    preprocessing/
      deskew.py
      binarization.py
      denoise.py
      normalize.py
      line_crop.py
      augmentations.py

    skeleton/
      thinning.py
      medial_axis.py
      skeleton_utils.py

    graph/
      pixel_graph.py
      node_detection.py
      edge_tracing.py
      pruning.py
      stroke_grouping.py
      curve_fitting.py
      graph_schema.py
      graph_io.py

    features/
      geometry.py
      topology.py
      handwriting_general.py
      handwriting_private.py
      special_blocks.py
      numeric_features.py
      quality.py
      informativeness.py

    models/
      image_baselines/
        crnn_ctc.py
        trocr_wrapper.py
      structural_extractor/
        unet_heads.py
        losses.py
      graph_models/
        graph_dataset.py
        gnn.py
        graph_transformer.py
        graph_tokenizer.py
      fusion/
        late_fusion.py
        gated_fusion.py
        cross_attention_fusion.py

    metrics/
      ocr.py
      skeleton_metrics.py
      graph_metrics.py
      curve_metrics.py
      topology_metrics.py
      calibration.py

    visualization/
      draw_graph.py
      draw_overlay.py
      export_svg.py
      error_report.py

    training/
      train_ocr.py
      train_structural_extractor.py
      train_graph_model.py
      train_fusion.py

    evaluation/
      eval_ocr.py
      eval_graph.py
      eval_robustness.py
      eval_cross_script.py
      eval_ablation.py

  notebooks/
    01_data_preview.ipynb
    02_graph_debug.ipynb
    03_feature_debug.ipynb
    04_model_errors.ipynb

  outputs/
    graphs/
    svg/
    overlays/
    predictions/
    metrics/
    reports/
```

---

# 5. Технический стек

## Язык и основные библиотеки

```text
Python 3.10+
PyTorch
torchvision
OpenCV
scikit-image
NumPy
SciPy
NetworkX
PyTorch Geometric или DGL
HuggingFace Transformers
Pillow
Shapely
svgwrite
matplotlib
pandas
tqdm
Hydra или OmegaConf
Weights & Biases / MLflow
```

## Почему так

OpenCV и scikit-image нужны для предобработки, бинаризации и скелетизации. NetworkX удобен для первой версии graph pipeline. PyTorch Geometric или DGL нужны для GNN. HuggingFace Transformers нужен для TrOCR baseline; TrOCR является end-to-end OCR-подходом на основе image Transformer и text Transformer. ([AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/26538?utm_source=chatgpt.com))

DiffVG или аналогичный differentiable rasterizer не нужен в MVP, но его стоит держать как advanced module, потому что DiffVG связывает vector graphics и raster loss через дифференцируемую растеризацию. ([people.csail.mit.edu](https://people.csail.mit.edu/tzumao/diffvg/?utm_source=chatgpt.com))

---

# 6. Датасеты

## 6.1. IAM

Назначение:

```text
английский baseline;
line-level HTR;
проверка на классическом benchmark;
writer-independent split.
```

IAM содержит 657 авторов, 1 539 страниц, 5 685 предложений, 13 353 строки и 115 320 слов. ([fki.tic.heia-fr.ch](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database?utm_source=chatgpt.com))

---

## 6.2. HWR200

Назначение:

```text
современный русский;
фото в разных условиях;
проверка устойчивости к реальным условиям съёмки;
writer-level эксперименты.
```

HWR200 - русские рукописные изображения от 200 авторов с разными почерками и условиями фотографирования. ([Hugging Face](https://huggingface.co/datasets/AntiplagiatCompany/HWR200?utm_source=chatgpt.com))

---

## 6.3. HKR

Назначение:

```text
масштабная кириллица;
предобучение;
проверка RU/KZ Cyrillic generalization;
line-level recognition.
```

HKR включает около 95% русского и 5% казахского текста и используется для offline handwriting recognition. ([GitHub](https://github.com/abdoelsayed2016/HKR_Dataset?utm_source=chatgpt.com))

---

## 6.4. Digital Peter

Назначение:

```text
исторический русский stress-test;
domain shift;
сложные формы, нестандартная орфография, старые документы.
```

В MVP Digital Peter можно использовать не сразу, а на этапе robustness/cross-domain.

---

# 7. Единый формат данных

Каждый пример в processed dataset должен быть приведён к такому виду:

```json
{
  "sample_id": "iam_000001",
  "dataset": "IAM",
  "language": "en",
  "script": "latin",
  "writer_id": "writer_0001",
  "image_path": "data/processed/iam/images/iam_000001.png",
  "transcription": "recognized ground truth text",
  "level": "line | word | page | patch",
  "bbox": null,
  "metadata": {
    "source_split": "train",
    "scan_type": "scan | photo | unknown",
    "instrument": "unknown",
    "paper": "unknown",
    "condition": "ordinary | unknown"
  }
}
```

Для русских данных:

```json
{
  "sample_id": "hwr200_000001",
  "dataset": "HWR200",
  "language": "ru",
  "script": "cyrillic",
  "writer_id": "writer_0042",
  "image_path": "...",
  "transcription": "...",
  "level": "line | page | crop",
  "metadata": {
    "scan_type": "photo",
    "environment": "varied",
    "condition": "unknown"
  }
}
```

---

# 8. Этап 0 - спецификация HI-CSG-R

## Цель

Создать формальную спецификацию графа, чтобы все последующие модули строили один и тот же тип объекта.

## Что нужно сделать

Создать файл:

```text
docs/01_hi_csg_r_schema.md
```

В нём описать:

```text
1. Что такое HI-CSG-R.
2. Какие бывают уровни графа.
3. Какие бывают узлы.
4. Какие бывают рёбра.
5. Какие признаки хранятся в узлах.
6. Какие признаки хранятся в рёбрах.
7. Какие признаки хранятся на уровне слова/строки.
8. Как хранить confidence.
9. Как хранить неопределённость.
10. Как экспортировать в JSON.
11. Как визуализировать.
```

---

## 8.1. Уровни графа

MVP должен поддерживать три уровня:

```text
1. Raw Skeleton Graph
2. Canonical Stroke Graph
3. Handwriting Feature Graph
```

### Raw Skeleton Graph

Это прямой граф из skeleton pixels.

```text
узлы = endpoint / junction / crossing_candidate
рёбра = пиксельные пути между узлами
```

### Canonical Stroke Graph

Это очищенный граф.

```text
узлы = устойчивые структурные точки
рёбра = сглаженные stroke segments
кривые = polyline / Bézier
```

### Handwriting Feature Graph

Это уже интерпретация.

```text
наклон
размер
связность
петли
подстрочные элементы
верхние элементы
соединения
вариативность
информативность
```

---

## 8.2. Типы узлов

```text
endpoint
junction
crossing
crossing_candidate
loop_closure
curvature_extreme
connection_point
baseline_touch
upper_zone_extreme
lower_zone_extreme
start_candidate
end_candidate
noise_node
uncertain_node
```

Важно: `start_candidate` и `end_candidate`, а не `start` и `end`, потому что мы не восстанавливаем реальный порядок письма.

---

## 8.3. Типы рёбер

```text
stroke_segment
connector
loop_segment
ascender_segment
descender_segment
terminal_segment
initial_segment
crossing_segment
correction_candidate
noise_branch
uncertain_edge
```

---

## 8.4. Атрибуты узла

```json
{
  "id": 12,
  "type": "junction",
  "x": 143.2,
  "y": 57.8,
  "x_norm": 0.42,
  "y_norm": 0.37,
  "degree": 3,
  "zone": "middle | upper | lower | unknown",
  "confidence": 0.91,
  "source": "classical | neural | manual"
}
```

---

## 8.5. Атрибуты ребра

```json
{
  "id": 41,
  "from": 12,
  "to": 18,
  "type": "stroke_segment",
  "points": [[143, 58], [144, 59], [146, 60]],
  "curve_type": "polyline | bezier",
  "control_points": [],
  "length": 31.5,
  "curvature_mean": 0.21,
  "curvature_std": 0.09,
  "smoothness": 0.87,
  "angularity": 0.13,
  "width_mean": 2.6,
  "width_std": 0.4,
  "tangent_histogram": [],
  "connection_type": "continuous | interval | uncertain",
  "confidence": 0.78
}
```

---

## 8.6. Атрибуты уровня строки

```json
{
  "line_features": {
    "baseline_slope": 0.04,
    "baseline_waviness": 0.12,
    "mean_slant": 72.4,
    "slant_variance": 5.1,
    "mean_height": 32.1,
    "height_variance": 4.7,
    "expansion_ratio": 0.82,
    "connectivity_score": 0.71,
    "component_count": 13,
    "loop_count": 8,
    "junction_count": 42,
    "endpoint_count": 33,
    "informativeness": 0.68
  }
}
```

---

# 9. Этап 1 - подготовка данных

## Цель

Сделать единый data pipeline для всех датасетов.

## Что сделать

```text
1. Скачать датасеты.
2. Привести изображения к единому формату.
3. Привести транскрипты к единому формату.
4. Сделать train/val/test splits.
5. Зафиксировать writer-independent split, где возможно.
6. Сохранить metadata.
7. Сделать preview-ноутбук.
```

---

## 9.1. Нормализация транскриптов

Для каждого датасета нужно хранить две версии текста.

```json
{
  "raw_transcription": "Original Text!",
  "normalized_transcription": "original text"
}
```

Для русского:

```text
ё/е - отдельно проверить;
старую орфографию Digital Peter не нормализовать без отдельного режима;
пунктуацию хранить, но иметь режим без пунктуации;
цифры сохранять;
латиницу и кириллицу не смешивать автоматически.
```

---

## 9.2. Splits

Минимальные splits:

```text
train
val
test
```

Желательно:

```text
writer_independent_train
writer_independent_val
writer_independent_test
```

Для cross-script:

```text
en_train
ru_train
mixed_train
ru_test
en_test
historical_ru_test
```

---

## 9.3. Проверки качества данных

Скрипт:

```text
src/datasets/validate_dataset.py
```

Проверяет:

```text
битые изображения;
пустые транскрипты;
слишком маленькие изображения;
слишком большие изображения;
несоответствие языка;
дубликаты;
нечитаемые символы;
нет writer_id;
слишком длинные строки.
```

---

# 10. Этап 2 - dual preprocessing

## Цель

Сделать две версии изображения:

```text
1. OCR-normalized image
2. Feature-preserving image
```

Это важно: OCR любит нормализацию, но почерковедческие признаки нельзя уничтожать. Например, агрессивный deskew может уничтожить наклон, а агрессивная бинаризация - признаки толщины штриха.

---

## 10.1. OCR-normalized image

Используется для baseline OCR.

Операции:

```text
grayscale
denoise
contrast normalization
deskew
height normalization
padding
optional binarization
```

Выход:

```text
data/processed/.../ocr_images/
```

---

## 10.2. Feature-preserving image

Используется для graph/feature branch.

Операции:

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

Выход:

```text
data/processed/.../feature_images/
```

---

## 10.3. Эксперименты с preprocessing

Проверить варианты:

```text
P0: no preprocessing
P1: Otsu binarization
P2: Sauvola binarization
P3: adaptive Gaussian threshold
P4: background removal + Sauvola
P5: CLAHE + Sauvola
P6: deskew + Sauvola
P7: no deskew + Sauvola
P8: denoise strong
P9: denoise weak
```

Метрики:

```text
skeleton quality
number of components
number of noise branches
OCR CER
graph reconstruction IoU
manual visual score
```

---

# 11. Этап 3 - baseline OCR/HTR

## Цель

Получить image-only baseline, чтобы потом сравнить graph-only и image+graph.

---

## 11.1. Модель A: CRNN + CTC

Архитектура:

```text
CNN backbone
→ sequence features по ширине
→ BiLSTM или Transformer encoder
→ CTC classifier
```

Плюсы:

```text
простая;
быстрая;
понятная;
хорошая baseline-модель.
```

Минусы:

```text
может проиграть современным Transformer-моделям.
```

---

## 11.2. Модель B: TrOCR baseline

Архитектура:

```text
Vision Transformer encoder
+ text Transformer decoder
```

TrOCR является end-to-end OCR-моделью, использующей Transformer для visual understanding и text generation. ([Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/trocr?utm_source=chatgpt.com))

Плюсы:

```text
сильный современный baseline;
можно fine-tune;
легко сравнить с литературой.
```

Минусы:

```text
тяжелее;
может не иметь хорошей поддержки русского без fine-tune;
не даёт структурного графа.
```

---

## 11.3. Baseline experiments

```text
OCR-1: CRNN+CTC on IAM
OCR-2: CRNN+CTC on HWR200
OCR-3: CRNN+CTC on HKR
OCR-4: TrOCR fine-tune IAM
OCR-5: TrOCR fine-tune HWR200/HKR
OCR-6: train EN → test RU
OCR-7: train RU → test EN
OCR-8: train RU+EN → test RU/EN
OCR-9: test on noisy/blurred data
OCR-10: test on Digital Peter if ready
```

Метрики:

```text
CER
WER
line accuracy
inference time
model size
confidence calibration
```

---

# 12. Этап 4 - classical CSG extractor

## Цель

Сделать первый рабочий pipeline:

```text
image → skeleton → graph → stroke segments → curves → graph.json/svg
```

Технический обзор описывает этот типовой конвейер: скан → предобработка → скелет → граф → сегментация штрихов → аппроксимация кривыми → модель распознавания. `Исполнительное резюме.pdf`

---

## 12.1. Binarization

Вход:

```text
feature-preserving image
```

Варианты:

```text
Otsu
Sauvola
Niblack
adaptive Gaussian
adaptive mean
```

Выход:

```text
binary image
```

---

## 12.2. Skeletonization

Варианты:

```text
Zhang-Suen
Guo-Hall
scikit-image skeletonize
medial axis
```

Выход:

```text
one-pixel skeleton
```

Эксперименты:

```text
SKE-1: Zhang-Suen + Otsu
SKE-2: Zhang-Suen + Sauvola
SKE-3: Guo-Hall + Sauvola
SKE-4: medial axis + Sauvola
SKE-5: skeleton ensemble из нескольких бинаризаций
```

Метрики:

```text
branch count
endpoint count
junction count
noise branch count
manual visual score
skeleton IoU against gold subset
```

---

## 12.3. Pixel graph construction

Правило:

```text
каждый skeleton pixel = потенциальный графовый пиксель
```

Соседство:

```text
8-neighborhood
```

Классификация пикселей:

```text
degree = 0 → isolated noise
degree = 1 → endpoint
degree = 2 → intermediate
degree >= 3 → junction
```

---

## 12.4. Edge tracing

Алгоритм:

```text
1. Найти все endpoint/junction pixels.
2. От каждого такого узла идти по skeleton pixels.
3. Идти до следующего endpoint/junction.
4. Все промежуточные pixels сохранить как edge path.
5. Если цикл без endpoint/junction - создать loop node или virtual node.
```

Выход:

```json
{
  "nodes": [],
  "edges": [
    {
      "from": 1,
      "to": 2,
      "points": [...]
    }
  ]
}
```

---

## 12.5. Pruning

Удаляем:

```text
очень короткие ветки;
изолированные шумовые компоненты;
spurs;
ложные ветки на утолщениях;
ветки меньше порога длины относительно stroke width.
```

Эксперименты:

```text
PRUNE-1: fixed threshold 3 px
PRUNE-2: fixed threshold 5 px
PRUNE-3: threshold = 0.5 * local stroke width
PRUNE-4: threshold = 1.0 * local stroke width
PRUNE-5: no pruning
```

Метрики:

```text
noise branch reduction
lost true endpoints
loop preservation
manual visual score
```

---

## 12.6. Stroke width estimation

Нужно получить approximate stroke width.

Метод:

```text
distance transform от binary mask
width at skeleton pixel = 2 * distance_to_boundary
```

Для каждого edge:

```text
width_mean
width_std
width_min
width_max
width_profile
```

Это будет использоваться как pressure-width proxy, но не как настоящий нажим.

---

## 12.7. Curve fitting

Варианты:

```text
polyline raw
Ramer-Douglas-Peucker simplified polyline
cubic Bézier fitting
B-spline fitting
line/arc primitive fitting
```

Эксперименты:

```text
CURVE-1: raw polyline
CURVE-2: RDP epsilon=1
CURVE-3: RDP epsilon=2
CURVE-4: cubic Bézier per edge
CURVE-5: split edge by curvature extrema + Bézier per segment
CURVE-6: B-spline fitting
```

Метрики:

```text
curve reconstruction error
number of control points
compression ratio
raster reconstruction IoU
Fréchet distance
```

---

# 13. Этап 5 - visual debugging

## Цель

Без визуализации графовый pipeline невозможно отлаживать.

Нужно сделать:

```text
overlay original + skeleton
overlay original + nodes
overlay original + edges
overlay original + loops
overlay original + stroke width heatmap
SVG export
side-by-side error report
```

---

## 13.1. Цветовая схема визуализации

```text
endpoints - зелёные
junctions - красные
crossings - фиолетовые
loop closures - синие
edges - жёлтые
noise branches - серые
low-confidence edges - пунктир
```

---

## 13.2. Error report

Для каждого изображения автоматически сохранять:

```text
original.png
binary.png
skeleton.png
nodes_edges.png
vector.svg
report.json
```

`report.json`:

```json
{
  "sample_id": "...",
  "component_count": 12,
  "endpoint_count": 31,
  "junction_count": 44,
  "loop_count": 8,
  "short_branch_count": 17,
  "graph_quality_score": 0.68,
  "warnings": [
    "too_many_short_branches",
    "possible_lost_loop",
    "large_component_touching_border"
  ]
}
```

---

# 14. Этап 6 - structural metrics

## Цель

Оценивать не только OCR, но и качество графа.

Технический обзор прямо отмечает, что CER/WER не чувствительны к качеству структурного представления, поэтому нужны графовые и криволинейные метрики. `Исполнительное резюме.pdf`

---

## 14.1. Метрики без gold subset

Эти метрики можно считать сразу:

```text
component count
endpoint count
junction count
loop count
average edge length
short branch ratio
graph density
compression ratio
raster reconstruction IoU
skeleton-to-render Chamfer distance
```

---

## 14.2. Метрики с gold subset

```text
Skeleton F1
Endpoint Precision/Recall/F1
Junction Precision/Recall/F1
Crossing Precision/Recall/F1
Loop preservation score
Edge matching F1
Graph edit distance
Curve Fréchet distance
Critical Topology Error
```

---

## 14.3. Critical Topology Error

Это важная авторская метрика.

Штрафовать:

```text
потеря петли в о/а/д/р/б;
ложное соединение двух букв;
разрыв реального соединения;
исчезновение подстрочного элемента;
исчезновение надстрочного элемента;
ложное пересечение;
лишняя компонента внутри буквы;
слияние соседних слов.
```

Формула:

```text
CTE = weighted_sum(critical_errors) / number_of_relevant_structures
```

Пример весов:

```text
lost_loop = 3
false_letter_connection = 4
broken_cursive_connection = 3
lost_descender = 3
false_crossing = 2
noise_branch = 1
```

---

# 15. Этап 7 - gold subset

## Цель

Создать маленький, но честный набор ручной/полуручной разметки графов.

---

## 15.1. Объём

Минимум:

```text
100 английских слов
100 русских слов
50 цифровых/смешанных записей
```

Лучше:

```text
200 английских слов
200 русских слов
100 цифровых/смешанных записей
```

---

## 15.2. Что размечать

```text
1. skeleton correction
2. endpoints
3. junctions
4. crossings
5. loops
6. edges
7. continuous/interval connection
8. letter/digit region, если возможно
9. uncertain zones
```

---

## 15.3. Формат аннотации

```json
{
  "sample_id": "ru_0001",
  "annotator": "manual_v1",
  "nodes": [
    {
      "id": 1,
      "type": "endpoint",
      "x": 123,
      "y": 45
    }
  ],
  "edges": [
    {
      "id": 1,
      "from": 1,
      "to": 2,
      "points": [],
      "type": "stroke_segment"
    }
  ],
  "loops": [
    {
      "id": 1,
      "edge_ids": [4, 5, 6],
      "label": "letter_o_like"
    }
  ],
  "uncertain_regions": []
}
```

---

## 15.4. Инструмент разметки

Минимально можно сделать:

```text
napari plugin
Label Studio custom interface
CVAT polygon/polyline
собственный matplotlib GUI
```

Для MVP проще:

```text
CVAT / Label Studio + экспорт JSON
```

---

# 16. Этап 8 - synthetic data

## Цель

Создать данные, где известен векторный ground truth.

Проблема: реальные датасеты почти всегда имеют только изображение и транскрипт, но не stroke-level разметку. В техническом обзоре это обозначено как один из главных пробелов. `Исполнительное резюме.pdf`

---

## 16.1. Типы синтетики

### Synthetic A: vector fonts

```text
рукописные/курсивные шрифты
→ render to raster
→ known vector outlines
```

Минус: это контур, а не настоящий stroke graph.

### Synthetic B: procedural strokes

```text
задать буквы как набор Bézier strokes
→ случайно варьировать параметры
→ рендерить
→ known stroke graph
```

Плюс: настоящий ground truth.

### Synthetic C: online handwriting to offline

```text
online trajectories
→ render as image
→ known stroke order and vector path
```

Плюс: есть траектория. Минус: может быть не RU.

### Synthetic D: graph augmentations

```text
взять реальный extracted graph
→ деформировать граф
→ рендерить обратно
→ known pseudo-ground truth
```

---

## 16.2. Аугментации графа

```text
slant change
size change
expansion change
baseline curve
stroke width variation
broken connection
loop deformation
connector simplification
noise branch insertion
local tremor
terminal elongation
```

---

# 17. Этап 9 - neural structural extractor

## Цель

Научить модель предсказывать структурные карты из изображения.

---

## 17.1. Вход

```text
feature-preserving image
```

---

## 17.2. Выходы модели

```text
skeleton probability map
endpoint heatmap
junction heatmap
crossing heatmap
loop heatmap
stroke width map
tangent field
baseline map
upper/middle/lower zone map
```

---

## 17.3. Архитектуры

### Model S1: U-Net

```text
encoder-decoder
skip connections
multi-head output
```

Плюсы:

```text
простая;
хорошо для dense prediction;
быстро отлаживается.
```

### Model S2: U-Net + ResNet encoder

```text
ResNet encoder
U-Net decoder
multi-head output
```

### Model S3: HRNet-like

Плюсы:

```text
хорошо держит spatial resolution.
```

### Model S4: Swin-Unet / SegFormer-like

Плюсы:

```text
лучше глобальный контекст;
полезно для строк и длинных слов.
```

Минусы:

```text
сложнее;
тяжелее.
```

---

## 17.4. Loss-функции

```text
L_skeleton = BCE + Dice
L_endpoint = focal loss / heatmap MSE
L_junction = focal loss / heatmap MSE
L_crossing = focal loss / heatmap MSE
L_loop = focal loss / heatmap MSE
L_width = L1 / SmoothL1
L_tangent = cosine loss
L_baseline = L1 / BCE
```

Итого:

```text
L_struct =
  λ1 L_skeleton
+ λ2 L_endpoint
+ λ3 L_junction
+ λ4 L_crossing
+ λ5 L_loop
+ λ6 L_width
+ λ7 L_tangent
+ λ8 L_baseline
```

---

## 17.5. Эксперименты

```text
NS-1: U-Net только skeleton
NS-2: U-Net skeleton + endpoints
NS-3: U-Net skeleton + endpoints + junctions
NS-4: U-Net full heads
NS-5: ResNet-U-Net full heads
NS-6: HRNet-like full heads
NS-7: Swin/Semantic Transformer full heads
NS-8: train on pseudo-labels only
NS-9: train on pseudo-labels + gold subset
NS-10: train on synthetic + pseudo + gold
```

Метрики:

```text
Skeleton F1
Endpoint F1
Junction F1
Loop F1
Width MAE
Tangent angular error
Graph reconstruction after builder
```

---

# 18. Этап 10 - graph builder v2

## Цель

Строить граф не только из classical skeleton, но из neural maps.

---

## Вход

```text
skeleton probability map
endpoint heatmap
junction heatmap
crossing heatmap
loop heatmap
width map
tangent field
```

---

## Алгоритм

```text
1. Threshold skeleton map.
2. Thin predicted skeleton if needed.
3. Extract candidate keypoints from heatmaps.
4. Snap keypoints to skeleton.
5. Trace edges between keypoints.
6. Use tangent field to resolve ambiguous branches.
7. Use width map for pruning.
8. Assign confidence to nodes/edges.
9. Fit curves.
10. Export HI-CSG-R.
```

---

## Confidence

Для узла:

```text
node_confidence =
  heatmap_score
  × skeleton_support
  × local_topology_consistency
```

Для ребра:

```text
edge_confidence =
  mean_skeleton_probability
  × tangent_consistency
  × width_consistency
  × length_prior
```

---

# 19. Этап 11 - handwriting feature extractor

## Цель

Из графа вычислить признаки, вдохновлённые почерковедением.

Жижина делит общие частносистемные признаки на структурно-геометрические, динамические и пространственно-ориентационные; структурно-геометрические признаки включают форму, направление, размер, разгон, наклон и связность. `Zhizhina_M_V_Pocherkovedenie_2006.docx`

---

## 19.1. Общие признаки

```text
baseline slope
baseline waviness
line direction
mean slant
slant variance
mean character height
height variance
expansion ratio
word spacing
component count
connectivity score
loop count
junction density
endpoint density
```

---

## 19.2. Координационно-ритмические proxy

```text
stroke smoothness
angularity
waviness
local tremor proxy
regularity of slant
regularity of height
regularity of spacing
terminal bluntness
retouch candidate count
broken stroke ratio
```

---

## 19.3. Pressure-width proxy

```text
mean stroke width
stroke width variance
local thickening count
width profile asymmetry
ink density mean
ink density variance
pressure_proxy_reliability
```

Всегда писать `pressure_proxy`, не `pressure`.

---

## 19.4. Особенные блоки

```text
oval block
upper loop block
lower descender block
connector block
terminal hook
vertical stem group
arc connector
```

Для кириллицы:

```text
о/а/ю/ф → oval-like group
д/у/р/ц/щ → lower-zone group
б/й → upper-zone group
и/н/п/м/ш/щ → repeated stem-connector group
```

---

## 19.5. Частные признаки

```text
start_candidate position
end_candidate position
connection point position
crossing position
loop closure position
initial stroke shape
terminal stroke shape
element curvature
element direction hypothesis
element length
element width
```

---

# 20. Этап 12 - numeric handwriting mode

## Почему нужен

Особенная часть отдельно рассматривает цифровой почерк. Цифровые записи имеют ограниченное число знаков, часто несвязное выполнение, но обладают специфическими признаками транскрипции дат, сумм, веса, нумерации и слов-спутников. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

В MVP numeric mode можно сделать как feature module, не как отдельную большую модель.

---

## Что реализовать

```text
detect digit-heavy samples
extract digit graph features
extract numeric spacing
extract punctuation in numbers
extract companion words if OCR available
```

---

## Numeric features

```json
{
  "numeric_features": {
    "digit_height": {},
    "digit_slant": {},
    "digit_spacing": {},
    "fraction_position": {},
    "separator_type": "dot | comma | dash | slash | none",
    "companion_words": ["руб", "кг", "г"],
    "transcription_type": "numeric | mixed | unknown"
  }
}
```

---

# 21. Этап 13 - graph-only recognition

## Цель

Проверить, может ли граф сам распознавать текст.

---

## Вариант G1: handcrafted features + classical ML

```text
graph features
→ sequence aggregation
→ LightGBM / MLP
→ character/word classification или auxiliary task
```

Не главный OCR, но хороший sanity check.

---

## Вариант G2: GNN

```text
nodes + edges
→ GCN/GAT/Edge-conditioned GNN
→ graph embeddings
→ CTC decoder
```

---

## Вариант G3: Graph Transformer

```text
graph tokens
+ node/edge positional encodings
→ Graph Transformer
→ sequence pooling along x-axis
→ CTC
```

---

## Вариант G4: Graph tokenizer

```text
CSG
→ structural tokens
→ Transformer encoder
→ CTC/decoder
```

Пример токенов:

```text
oval_closed
vertical_stem
lower_loop
upper_loop
arc_connector
terminal_hook
junction_3
crossing
```

---

## Эксперименты

```text
G-1: handcrafted graph stats → character-level auxiliary
G-2: GAT graph-only → CTC
G-3: Graph Transformer → CTC
G-4: graph tokenizer → Transformer → CTC
G-5: graph-only EN
G-6: graph-only RU
G-7: graph-only RU+EN
```

---

# 22. Этап 14 - raster + graph fusion

## Цель

Главная MVP-модель.

---

## Вариант F1: late fusion

```text
image_model logits
graph_model logits
→ weighted average
```

Плюсы:

```text
легко реализовать;
легко отлаживать;
понятная абляция.
```

---

## Вариант F2: feature concatenation

```text
image_features
graph_features
→ concat
→ decoder
```

---

## Вариант F3: gated fusion

```text
gate = sigmoid(MLP([image_features, graph_features, graph_quality]))
fused = gate * graph_features + (1 - gate) * image_features
```

Это важно: если граф плохой, модель должна меньше ему доверять.

---

## Вариант F4: cross-attention fusion

```text
text decoder queries
attend to image tokens
attend to graph tokens
combine
```

---

## Вариант F5: mutual learning

```text
image branch predicts text
graph branch predicts text
fusion predicts text
branches distill each other
```

---

## Эксперименты

```text
FUS-1: image-only CRNN
FUS-2: image-only TrOCR
FUS-3: graph-only GAT
FUS-4: graph-only graph tokenizer
FUS-5: late fusion
FUS-6: concat fusion
FUS-7: gated fusion
FUS-8: cross-attention fusion
FUS-9: fusion with graph_quality gate
FUS-10: fusion under noisy graph
```

---

# 23. Этап 15 - robustness experiments

## Цель

Показать, что графовая структура помогает не обязательно на чистых данных, а в сложных условиях.

---

## Искажения изображения

```text
Gaussian blur
motion blur
Gaussian noise
salt-and-pepper noise
low contrast
uneven lighting
JPEG artifacts
small rotation
perspective distortion
background texture
```

---

## Структурные искажения

```text
broken strokes
thickened strokes
eroded strokes
dilated strokes
partial occlusion
lost loops
extra noise branches
false connections
```

---

## Почерковедчески мотивированные искажения

```text
slant variation
size variation
expansion variation
baseline waviness
reduced connectivity
increased connectivity
terminal elongation
tremor-like perturbation
local angularity
```

---

## Метрики

```text
CER clean
CER distorted
WER clean
WER distorted
graph quality clean
graph quality distorted
confidence calibration
performance drop
```

---

# 24. Этап 16 - cross-script experiments

## Цель

Проверить, какие признаки переносятся между латиницей и кириллицей.

---

## Эксперименты

```text
CS-1: train EN IAM → test RU HWR200
CS-2: train RU HWR200/HKR → test EN IAM
CS-3: train EN+RU → test RU
CS-4: train EN+RU → test EN
CS-5: train modern RU → test Digital Peter
CS-6: pretrain graph extractor EN → fine-tune RU
CS-7: pretrain graph extractor RU → fine-tune EN
```

---

## Что сравнивать

```text
image-only transfer
graph-only transfer
image+graph transfer
graph feature stability
special blocks usefulness
```

---

# 25. Этап 17 - ablation study

## Цель

Понять, какие части реально дают вклад.

---

## Абляции

```text
ABL-1: full model
ABL-2: without graph branch
ABL-3: without handwriting features
ABL-4: without special blocks
ABL-5: without width map
ABL-6: without tangent field
ABL-7: without loop head
ABL-8: without junction head
ABL-9: without confidence gate
ABL-10: without topology loss
ABL-11: without gold subset fine-tuning
ABL-12: without synthetic data
ABL-13: with classical graph only
ABL-14: with neural graph only
```

---

# 26. Этап 18 - informativeness estimator

## Цель

Модель должна понимать, когда материала недостаточно.

Особенная часть подчёркивает правило: чем короче спорная запись, тем сложнее исследование и тем больше образцов нужно. `Pocherk_expertiza_Uchebnik_Osobennaya_chast_2007.doc`

---

## Признаки информативности

```text
text length
number of connected components
number of letters/digits
number of loops
number of junctions
symbol diversity
repeated letters
allograph coverage
stroke graph density
graph confidence
```

---

## Выход

```json
{
  "informativeness": {
    "score": 0.68,
    "reason": [
      "short_text",
      "few_repeated_letters",
      "low_loop_count"
    ]
  }
}
```

---

# 27. Этап 19 - итоговые таблицы MVP

## Таблица 1. OCR baseline

```text
model | dataset | CER | WER | line accuracy
```

## Таблица 2. Graph quality

```text
method | Skeleton F1 | Endpoint F1 | Junction F1 | Loop score | CTE
```

## Таблица 3. Vector quality

```text
method | Chamfer | Fréchet | Bezier error | compression ratio | IoU
```

## Таблица 4. Fusion

```text
model | IAM CER | HWR200 CER | HKR CER | noisy CER | graph quality
```

## Таблица 5. Robustness

```text
model | clean | blur | noise | low contrast | broken strokes | thick strokes
```

## Таблица 6. Cross-script

```text
train | test | image-only CER | graph-only CER | fusion CER
```

## Таблица 7. Ablation

```text
variant | CER | WER | Loop score | CTE | notes
```

---

# 28. Полный список экспериментов MVP

## Группа A. Preprocessing

```text
A1 Otsu vs Sauvola
A2 deskew vs no deskew
A3 strong denoise vs weak denoise
A4 CLAHE vs no CLAHE
A5 feature-preserving vs OCR-normalized
```

## Группа B. Skeleton

```text
B1 Zhang-Suen
B2 Guo-Hall
B3 medial axis
B4 skeleton ensemble
B5 learned skeleton map
```

## Группа C. Graph extraction

```text
C1 degree-based node detection
C2 heatmap-assisted node detection
C3 fixed pruning
C4 width-aware pruning
C5 tangent-aware edge tracing
```

## Группа D. Curve fitting

```text
D1 raw polyline
D2 simplified polyline
D3 cubic Bézier
D4 curvature-split Bézier
D5 B-spline
```

## Группа E. Structural extractor

```text
E1 U-Net skeleton only
E2 U-Net multi-head
E3 ResNet-U-Net
E4 HRNet-like
E5 Swin/SegFormer-like
```

## Группа F. Graph recognition

```text
F1 handcrafted graph features
F2 GAT
F3 Edge-conditioned GNN
F4 Graph Transformer
F5 graph tokenizer
```

## Группа G. Fusion

```text
G1 late fusion
G2 concat fusion
G3 gated fusion
G4 cross-attention
G5 graph-quality-aware fusion
```

## Группа H. Robustness

```text
H1 blur
H2 noise
H3 low contrast
H4 broken strokes
H5 thick strokes
H6 perspective distortion
H7 baseline waviness
H8 reduced connectivity
```

## Группа I. Cross-script

```text
I1 EN→RU
I2 RU→EN
I3 EN+RU→RU
I4 modern RU→historical RU
```

## Группа J. Feature ablations

```text
J1 no loop features
J2 no junction features
J3 no width features
J4 no tangent features
J5 no special blocks
J6 no confidence
J7 no topology loss
```

---

# 29. Риски и решения

## Риск 1. Classical skeleton плохой

Решение:

```text
multi-threshold ensemble
learned skeleton map
manual gold subset
confidence-aware graph
```

---

## Риск 2. Graph branch не улучшает CER

Решение:

```text
оценивать не только CER;
показывать structural metrics;
показывать robustness;
показывать interpretability;
показывать cross-script transfer.
```

---

## Риск 3. Нет stroke ground truth

Решение:

```text
pseudo-labels
gold subset
synthetic graph data
raster reconstruction
downstream HTR validation
```

---

## Риск 4. Слишком большой граф

Решение:

```text
word-level graphs
patch-level graphs
graph pooling
graph tokenizer
edge simplification
```

---

## Риск 5. Нажим и темп нельзя восстановить

Решение:

```text
использовать только pressure_proxy и tempo_proxy;
хранить reliability;
не делать сильных выводов.
```

---

## Риск 6. Русская скоропись слишком сложна

Решение:

```text
начать со слов/строк средней сложности;
добавить special block features;
использовать HWR200/HKR;
сделать ручной gold subset именно для кириллицы.
```

---

# 30. Критерии готовности MVP

MVP готов, если выполнено:

```text
1. Есть рабочий data pipeline минимум для IAM и HWR200/HKR.
2. Есть OCR baseline.
3. Есть classical CSG extractor.
4. Есть graph.json и SVG/overlay visualization.
5. Есть structural metrics.
6. Есть gold subset.
7. Есть neural structural extractor.
8. Есть graph-only модель.
9. Есть fusion модель.
10. Есть таблицы экспериментов.
11. Есть robustness evaluation.
12. Есть cross-script evaluation.
13. Есть ablation.
14. Есть отчёт с выводами.
```

---

# 31. Порядок реализации по неделям

## Неделя 1-2. Спецификация и данные

```text
docs/HI_CSG_R_schema.md
dataset loaders
dataset validation
preview notebooks
train/val/test splits
```

---

## Неделя 3-4. OCR baseline

```text
CRNN+CTC
TrOCR baseline/fine-tune
CER/WER scripts
baseline tables
```

---

## Неделя 5-7. Classical graph extractor

```text
binarization
skeletonization
pixel graph
node detection
edge tracing
pruning
curve fitting
graph.json
SVG export
```

---

## Неделя 8. Structural metrics

```text
Skeleton F1
Endpoint F1
Junction F1
Loop score
Chamfer
IoU
Critical Topology Error
```

---

## Неделя 9-10. Gold subset

```text
annotation protocol
manual annotation
annotation validation
inter-annotator check if possible
```

---

## Неделя 11-13. Neural structural extractor

```text
U-Net baseline
multi-head outputs
pseudo-label training
gold subset fine-tuning
metrics comparison
```

---

## Неделя 14-15. Feature extractor

```text
general features
private features
special blocks
numeric features
informativeness
```

---

## Неделя 16-18. Graph models

```text
graph dataset
GAT baseline
Graph Transformer/tokenizer baseline
graph-only recognition
```

---

## Неделя 19-21. Fusion

```text
late fusion
gated fusion
cross-attention optional
image+graph experiments
```

---

## Неделя 22-24. Robustness/cross-script/ablation

```text
noise experiments
blur experiments
broken stroke experiments
EN→RU
RU→EN
ablation tables
```

---

## Неделя 25-26. Отчёт/статья

```text
method section
dataset section
metrics section
experiments
limitations
future work
```

---

# 32. Минимальный порядок, если времени мало

Если времени сильно меньше, делать так:

```text
1. IAM + HWR200.
2. CRNN baseline.
3. Classical skeleton graph extractor.
4. SVG visualization.
5. Structural metrics.
6. Gold subset 100 EN + 100 RU.
7. Graph features + simple MLP/LightGBM.
8. Late fusion image logits + graph features.
9. Robustness blur/noise.
10. Final report.
```

Это будет слабее, но всё равно даст научно защищаемый MVP.

---

# 33. Финальная формулировка MVP-вклада

В первой версии работы мы заявляем не “лучшую OCR-модель в мире”, а:

```text
1. Новый формат HI-CSG-R для представления офлайн-рукописей.
2. Реализацию pipeline image → graph → vector → features.
3. Метрики структурного качества графа.
4. Небольшой benchmark графовой разметки RU/EN.
5. Гибридную image+graph HTR-модель.
6. Экспериментальное доказательство, что графовая структура полезна для интерпретируемости, робастности и cross-script transfer.
```

---

# 34. Главная техническая стратегия

Не начинать с большой сложной нейросети.

Правильный порядок:

```text
1. Сначала построить граф классически.
2. Посмотреть глазами 100-200 примеров.
3. Понять типовые ошибки.
4. Сделать метрики.
5. Сделать gold subset.
6. Только потом обучать neural structural extractor.
7. Только потом graph encoder.
8. Только потом fusion.
```

Это снизит риск, что модель станет “чёрным ящиком”, который вроде обучается, но непонятно, что именно делает.

---

# 35. Итог

Финальный MVP должен выглядеть так:

```text
Input:
    рукописная строка или слово на русском/английском

Output:
    распознанный текст
    canonical stroke graph
    SVG-векторизация
    структурные признаки
    graph confidence
    informativeness score
```

Главное направление:

```text
не pen trajectory recovery,
а handwriting-informed visible stroke graph reconstruction.
```

Это и есть наиболее сильная, осторожная и реализуемая версия нашей темы. Она использует все наши предыдущие идеи: скелетизацию, графы, векторизацию, GNN, raster+graph fusion, почерковедческие признаки, вариативность, цифровой режим, информативность, структурные метрики и русско-английскую специфику.

---

**Sources:**

- [TrOCR: Transformer-Based Optical Character Recognition ...](https://ojs.aaai.org/index.php/AAAI/article/view/26538?utm_source=chatgpt.com)
- [Differentiable Vector Graphics Rasterization for Editing and ...](https://people.csail.mit.edu/tzumao/diffvg/?utm_source=chatgpt.com)
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database?utm_source=chatgpt.com)
- [AntiplagiatCompany/HWR200 · Datasets at ...](https://huggingface.co/datasets/AntiplagiatCompany/HWR200?utm_source=chatgpt.com)
- [Handwritten Kazakh and Russian (HKR) database for text ...](https://github.com/abdoelsayed2016/HKR_Dataset?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)