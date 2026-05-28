# Этап 1. Data audit and preprocessing

---

## 1. Цель этапа

Цель Этапа 1 — создать воспроизводимый data pipeline, который приводит разные рукописные датасеты к единому внутреннему формату, не теряя важные для исследования свойства изображений и разметки.

На этом этапе нельзя сразу начинать обучение OCR/HTR-модели. Сначала нужно проверить данные, понять их уровни разметки, качество изображений, транскрипции, наличие writer_id, размеры, дубликаты, сложные случаи, лицензии, структуру файлов и пригодность к построению графа.

Главный результат этапа:

```text
data/raw/...           # исходные данные, не публикуются в репозитории
data/processed/...     # унифицированная структура метаданных и производных файлов
data/splits/...        # train/val/test splits
data/reports/...       # аудит датасетов
notebooks/01_data_preview.ipynb
```

---

## 2. Роль этапа в общей логике проекта

Этап 1 обеспечивает все последующие части проекта.

```text
Stage 1 data audit/preprocessing
        ↓
Stage 2 HI-CSG-R schema implementation
        ↓
Stage 3 image-only HTR baseline
        ↓
Stage 4 classical graph extractor
        ↓
Stage 5 gold subset
        ↓
Stage 6 structural metrics
        ↓
Stage 7 graph features and fusion
        ↓
Stage 8 robustness and ablation
```

Если Этап 1 выполнен плохо, дальше возникнут методологические ошибки:

```text
1. нельзя будет корректно сравнивать CER/WER;
2. невозможно будет воспроизвести splits;
3. graph extractor будет получать неоднородные изображения;
4. HTR baseline будет обучаться на плохо нормализованных транскрипциях;
5. gold subset будет нерепрезентативным;
6. robustness experiments будут смешивать реальные и искусственные искажения без контроля.
```

---

## 3. Основной принцип

Для каждого датасета нужно хранить не только изображения и текст, но и полную исследовательскую информацию:

```text
что это за датасет;
какой язык;
какой script;
какой уровень разметки;
есть ли writer_id;
является ли изображение словом, строкой, страницей или символом;
какая версия транскрипции используется;
какое качество изображения;
какой preprocessing применялся;
в какой split попал sample;
можно ли использовать sample для HTR, graph extraction, gold subset или robustness.
```

---

## 4. Датасеты этапа

### 4.1. Core datasets

В Этапе 1 обязательно готовятся:

```text
1. IAM
2. Cyrillic Handwriting Dataset
3. HWR200
4. HKR
```

### 4.2. Optional datasets

В Этапе 1 можно сделать только metadata-level audit, без полной обработки:

```text
1. School Notebooks
2. Digital Peter
3. Russian Handwriting OCR
4. Cyrillic-MNIST
```

### 4.3. Роли датасетов

| Датасет                      | Роль в проекте                                      |                 Уровень | Статус   |
| ---------------------------- | --------------------------------------------------- | ----------------------: | -------- |
| IAM                          | английский baseline, line-level HTR, EN gold subset |               line/word | Core     |
| Cyrillic Handwriting Dataset | основной русский word/phrase старт                  |             word/phrase | Core     |
| HWR200                       | реальные условия съёмки, robustness, RU hard cases  |      page/sentence/word | Core     |
| HKR                          | крупный RU line-level корпус                        |          line/page/form | Core-B   |
| School Notebooks             | будущий page-level extension                        | page/word/line polygons | Optional |
| Digital Peter                | historical domain shift                             |        historical lines | Optional |
| Russian Handwriting OCR      | дополнительный image-to-text корпус                 |              image/text | Optional |
| Cyrillic-MNIST               | toy/sanity symbols                                  |               character | Optional |

---

## 4.4. Фактический raw inventory после первичной проверки

По фактическому осмотру `data/raw` структура данных выглядит так:

| Dataset folder                          | Фактическое содержимое                                                    | Решение для pipeline                                                                       |
| --------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `data/raw/cyrillic-handwriting-dataset` | 73 830 PNG + `train.tsv` / `test.tsv`                                     | первый конвертер; самый низкий риск                                                        |
| `data/raw/iam`                          | 8 архивов `.tgz`: `words`, `lines`, `xml`, `ascii`, `sentences`, `forms*` | сначала распаковать в `data/interim/iam`, затем писать converter                           |
| `data/raw/hwr200`                       | Git/HF-структура + 11 `.zip`, включая `annotations.zip` и image shards    | сначала распаковать annotations и один shard для pilot                                     |
| `data/raw/hkr`                          | 2 `.rar` архива + PDF                                                     | нужен extractor для `.rar`; после распаковки повторить inventory                           |
| `data/raw/school_notebooks`             | HF/Git-структура + COCO json + `images.zip`                               | optional metadata audit; не Core на первом проходе                                         |
| `data/raw/peter`                        | HF/Git-структура + COCO json + `images.zip`                               | optional/future historical audit                                                           |
| `data/raw/cyrillic_mnist`               | 28 304 PNG + 4 CSV                                                        | официальный auxiliary symbol dataset; использовать только toy/sanity, не как HTR benchmark |

Практический вывод: сначала реализуется converter для `cyrillic-handwriting-dataset`, потому что он уже распакован, имеет готовые PNG и TSV-разметку. Фактический TSV-формат простой: `filename<TAB>transcription`, без заголовка. Примеры: `aa1.png<TAB>Молдова`, `test0.png<TAB>ибо`. IAM, HWR200 и HKR требуют предварительного этапа распаковки в `data/interim`.

### 4.5. Новое правило для архивов

Raw archives не распаковываются внутрь `data/raw`, чтобы сохранить исходную структуру. Все распаковки идут в `data/interim`:

```text
data/raw/iam/*.tgz                  → data/interim/iam/
data/raw/hwr200/*.zip               → data/interim/hwr200/
data/raw/hkr/*.rar                  → data/interim/hkr/
data/raw/peter/images.zip           → data/interim/peter/
data/raw/school_notebooks/images.zip → data/interim/school_notebooks/
```

После распаковки для каждого датасета повторяется inventory, и только затем пишется converter.

### 4.6. Приоритет реализации после фактического inventory

```text
1. Cyrillic Handwriting Dataset converter
2. общие metadata/validation/splits modules
3. IAM archive extraction + IAM converter
4. HWR200 annotations/images pilot extraction + HWR200 converter
5. HKR rar extraction + HKR converter
6. optional Peter / School Notebooks COCO audit
7. optional COMNIST/Cyrillic-MNIST-like converter for toy checks
```

---

## 5. Главные deliverables этапа

К концу Этапа 1 должны быть готовы:

```text
1. data_audit_report.md
2. dataset_stats.csv
3. dataset_issues.csv
4. unified_metadata.jsonl для каждого core dataset
5. splits.json для каждого core dataset
6. alphabet_report.json
7. preprocessing_preview_report.md
8. notebooks/01_data_preview.ipynb
9. scripts для валидации датасетов
10. OCR-normalized и feature-preserving версии изображений для pilot subset
```

---

## 6. Структура директорий

```text
hi-csg-r/
  data/
    raw/
      iam/
      cyrillic_handwriting/
      hwr200/
      hkr/
      school_notebooks/          # optional
      digital_peter/             # optional
      russian_handwriting_ocr/   # optional
      cyrillic_mnist/            # optional

    interim/
      iam/
      cyrillic_handwriting/
      hwr200/
      hkr/

    processed/
      iam/
        images/
        ocr_images/
        feature_images/
        metadata.jsonl
      cyrillic_handwriting/
        images/
        ocr_images/
        feature_images/
        metadata.jsonl
      hwr200/
        images/
        crops/
        ocr_images/
        feature_images/
        metadata.jsonl
      hkr/
        images/
        lines/
        ocr_images/
        feature_images/
        metadata.jsonl

    splits/
      iam_splits.json
      cyrillic_handwriting_splits.json
      hwr200_splits.json
      hkr_splits.json

    reports/
      data_audit_report.md
      dataset_stats.csv
      dataset_issues.csv
      alphabet_report.json
      preprocessing_preview_report.md

  notebooks/
    01_data_preview.ipynb

  src/
    datasets/
      __init__.py
      registry.py
      base.py
      iam.py
      cyrillic_handwriting.py
      hwr200.py
      hkr.py
      validate_dataset.py
      splits.py
      metadata.py

    preprocessing/
      __init__.py
      ocr_preprocess.py
      feature_preprocess.py
      binarization.py
      normalization.py
      crop_utils.py
      quality_checks.py
      preview.py
```

---

## 7. Единый формат metadata

Каждый sample должен быть приведён к единому внутреннему формату. Основной формат — JSONL: одна строка = один sample.

### 7.1. Базовая схема sample

```json
{
  "sample_id": "iam_line_000001",
  "dataset": "iam",
  "source_id": "a01-000u-00",
  "language": "en",
  "script": "latin",
  "level": "line",
  "writer_id": "writer_0001",
  "image_path": "data/processed/iam/images/iam_line_000001.png",
  "ocr_image_path": "data/processed/iam/ocr_images/iam_line_000001.png",
  "feature_image_path": "data/processed/iam/feature_images/iam_line_000001.png",
  "raw_transcription": "Original text.",
  "normalized_transcription": "original text.",
  "transcription_modes": {
    "raw": "Original text.",
    "lower": "original text.",
    "no_punct": "original text",
    "ctc_default": "original text."
  },
  "bbox": null,
  "polygon": null,
  "split": null,
  "metadata": {
    "source_split": null,
    "scan_type": "scan",
    "acquisition_condition": "unknown",
    "page_id": null,
    "line_id": null,
    "word_id": null,
    "quality_flags": [],
    "usable_for_htr": true,
    "usable_for_graph": true,
    "usable_for_gold_subset": true,
    "usable_for_robustness": true
  }
}
```

---

### 7.2. Обязательные поля

```text
sample_id
dataset
language
script
level
image_path
raw_transcription
normalized_transcription
metadata.quality_flags
metadata.usable_for_htr
metadata.usable_for_graph
metadata.usable_for_gold_subset
metadata.usable_for_robustness
```

---

### 7.3. Желательные поля

```text
writer_id
source_id
ocr_image_path
feature_image_path
bbox
polygon
page_id
line_id
word_id
scan_type
acquisition_condition
source_split
```

---

### 7.4. Поле level

Допустимые значения:

```text
character
word
phrase
line
sentence
page
crop
unknown
```

Важно: нельзя смешивать word-level и line-level результаты в одной таблице без явного указания `level`.

---

## 8. Правила sample_id

Формат:

```text
{dataset}_{level}_{running_id}
```

Примеры:

```text
iam_line_000001
cyr_word_000001
hwr200_page_000001
hwr200_crop_000001
hkr_line_000001
```

Требования:

```text
1. sample_id должен быть стабилен между запусками pipeline;
2. sample_id не должен зависеть от абсолютного пути на диске;
3. sample_id должен позволять восстановить dataset и level;
4. source_id хранится отдельно, чтобы не терять оригинальный идентификатор.
```

---

## 9. Нормализация транскрипций

### 9.1. Общий принцип

Никогда не перезаписывать оригинальную транскрипцию. Всегда хранить:

```text
raw_transcription
normalized_transcription
transcription_modes
```

---

### 9.2. Базовые режимы

```json
{
  "raw": "Исходный Текст!",
  "lower": "исходный текст!",
  "no_punct": "исходный текст",
  "ctc_default": "исходный текст!",
  "ctc_no_punct": "исходный текст"
}
```

---

### 9.3. Русский текст

Правила:

```text
1. сохранять raw_transcription без изменений;
2. normalized_transcription по умолчанию приводит к NFC unicode;
3. ё не заменять на е в raw;
4. создать отдельный режим ru_yo_to_e;
5. пунктуацию не удалять глобально;
6. создать отдельный режим no_punct;
7. цифры сохранять;
8. латиницу внутри русского текста не удалять автоматически;
9. неизвестные символы логировать в alphabet_report.
```

---

### 9.4. Английский текст

Правила:

```text
1. сохранять raw_transcription;
2. lower-case режим создать отдельно;
3. punctuation режимы разделить;
4. не удалять апострофы без отдельного режима;
5. сохранять цифры.
```

---

### 9.5. Исторический текст

Для Digital Peter:

```text
1. не нормализовать дореформенную орфографию в Core;
2. хранить original transcription;
3. создать отдельный historical_raw mode;
4. не смешивать с modern Russian HTR без отдельного эксперимента.
```

---

## 10. Alphabet report

Нужно собрать все символы по каждому датасету и каждому режиму транскрипции.

### 10.1. Выходной файл

```text
data/reports/alphabet_report.json
```

### 10.2. Содержимое

```json
{
  "iam": {
    "raw_chars": ["A", "B", "a", "b", "."],
    "ctc_default_chars": ["a", "b", "."],
    "num_samples": 10000,
    "unknown_or_rare_chars": []
  },
  "cyrillic_handwriting": {
    "raw_chars": ["А", "Б", "а", "б", "ё", "№"],
    "ctc_default_chars": ["а", "б", "ё", "№"],
    "num_samples": 73830,
    "unknown_or_rare_chars": ["…"]
  }
}
```

### 10.3. Зачем это нужно

Alphabet report нужен для:

```text
1. построения CTC alphabet;
2. поиска мусорных символов;
3. контроля русско-английского смешения;
4. проверки numeric/mixed samples;
5. описания данных в научной работе.
```

---

## 11. Data validation

### 11.1. Проверки изображений

Каждый image sample проверяется на:

```text
1. файл существует;
2. файл открывается;
3. формат поддерживается;
4. width > min_width;
5. height > min_height;
6. aspect ratio в допустимом диапазоне;
7. изображение не полностью белое;
8. изображение не полностью чёрное;
9. есть достаточно foreground pixels после soft threshold;
10. нет очевидной битой кодировки;
11. нет экстремально большого размера;
12. цветовой режим приводим к grayscale/RGB.
```

---

### 11.2. Проверки транскрипций

```text
1. raw_transcription не пустая;
2. normalized_transcription не пустая;
3. длина текста не равна 0;
4. длина текста не превышает max_length для данного level;
5. символы входят в known alphabet или логируются;
6. нет подозрительных control characters;
7. нет html/xml мусора;
8. нет битой unicode-кодировки;
9. язык примерно соответствует dataset/script;
10. не удалены цифры или пунктуация в raw mode.
```

---

### 11.3. Проверки metadata

```text
1. sample_id уникален;
2. image_path уникален или duplicate отмечен;
3. writer_id заполнен, если доступен;
4. level задан;
5. split пока null до генерации splits;
6. bbox/polygon валиден, если есть;
7. page_id/line_id/word_id согласованы, если есть;
8. acquisition_condition задан для HWR200;
9. source dataset указан.
```

---

### 11.4. Проверки дубликатов

Два уровня:

```text
1. exact duplicate by file hash;
2. near duplicate by perceptual hash.
```

Минимально:

```text
sha256 image hash
```

Желательно:

```text
perceptual hash / average hash
```

---

## 12. Quality flags

Каждому sample можно присваивать список flags.

### 12.1. Общие flags

```text
empty_transcription
missing_image
broken_image
too_small
too_large
extreme_aspect_ratio
low_contrast
mostly_blank
mostly_black
very_noisy
blurred
text_touches_border
multiple_lines_in_word_sample
unknown_characters
mixed_script
missing_writer_id
duplicate_exact
duplicate_near
bad_polygon
bad_bbox
```

---

### 12.2. HWR200-specific flags

```text
poor_light
good_light
scan
page_level_only
needs_cropping
segmentation_available
segmentation_missing
```

---

### 12.3. HKR-specific flags

```text
form_background
grid_lines_present
kazakh_chars_present
line_crop_needed
word_segmentation_available
```

---

### 12.4. Как использовать flags

```text
usable_for_htr = false, если нет транскрипции или изображение битое;
usable_for_graph = false, если foreground почти отсутствует или изображение слишком маленькое;
usable_for_gold_subset = false, если невозможно визуально интерпретировать структуру;
usable_for_robustness = true, если sample качественный baseline или имеет контролируемую acquisition condition.
```

---

## 13. Dataset-specific converters

### 13.1. BaseDatasetConverter

Все датасеты должны реализовать общий интерфейс.

```python
class BaseDatasetConverter:
    dataset_name: str

    def scan_raw(self) -> list:
        """Find raw files and annotations."""

    def parse_annotations(self) -> list[dict]:
        """Read original annotations."""

    def build_metadata(self) -> list[dict]:
        """Convert original annotations to unified metadata."""

    def validate(self) -> dict:
        """Run dataset-level validation."""

    def export_metadata(self, output_path: str) -> None:
        """Write metadata.jsonl."""
```

---

### 13.2. IAM converter

Задачи:

```text
1. прочитать IAM annotation files;
2. связать line/word images с транскрипциями;
3. извлечь writer_id, если доступен;
4. сохранить source_id;
5. привести к unified metadata;
6. сформировать line-level split;
7. optional: word-level metadata для gold subset.
```

Особенности:

```text
основной уровень — line;
word-level можно использовать для gold subset и graph debugging;
writer-independent split желателен.
```

---

### 13.3. Cyrillic Handwriting Dataset converter

Задачи:

```text
1. прочитать список изображений и транскрипций;
2. определить level: word или phrase;
3. проверить длину текста;
4. проверить кириллицу/цифры/пунктуацию;
5. создать sample_id;
6. создать train/val/test split;
7. сохранить metadata.jsonl.
```

Особенности:

```text
writer_id может отсутствовать;
если writer_id отсутствует, явно фиксировать missing_writer_id;
использовать как primary RU word/phrase dataset;
очень полезен для gold subset.
```

---

### 13.4. HWR200 converter

Задачи:

```text
1. прочитать JSON-аннотации;
2. извлечь page-level metadata;
3. извлечь sentence/word segmentation, если доступна;
4. сохранить acquisition_condition: scan / good_light / poor_light;
5. сохранить writer_id;
6. создать page-level metadata;
7. создать crop-level metadata для слов/предложений;
8. связать разные съёмки одного текста, если доступно;
9. создать robustness subsets.
```

Особенности:

```text
HWR200 важен не только для HTR, но и для real-condition robustness.
Нужно аккуратно не смешивать scan/good_light/poor_light в train/test без контроля.
```

---

### 13.5. HKR converter

Задачи:

```text
1. прочитать line-level транскрипции;
2. сохранить writer_id, если доступен;
3. отметить наличие казахских символов;
4. определить line crops;
5. создать metadata;
6. сделать writer-independent split, если возможно;
7. отметить form/grid background flags.
```

Особенности:

```text
HKR может содержать 95% русский и 5% казахский текст.
Для Core RU можно использовать только samples без дополнительных казахских символов.
Для extended Cyrillic можно оставить все.
```

---

## 14. Splits

### 14.1. Типы splits

Для каждого core dataset нужно создать:

```text
train
val
test
```

Где возможно:

```text
writer_independent_train
writer_independent_val
writer_independent_test
```

Для HWR200 дополнительно:

```text
robustness_scan
robustness_good_light
robustness_poor_light
```

---

### 14.2. Базовые пропорции

```text
train: 80%
val: 10%
test: 10%
```

Если dataset маленький или используется как external test:

```text
train: 70%
val: 10%
test: 20%
```

---

### 14.3. Writer-independent split

Правило:

```text
один writer_id не должен попадать одновременно в train, val и test.
```

Если writer_id отсутствует:

```text
1. использовать random split;
2. явно пометить split_type = random_no_writer_id;
3. не делать выводов о writer-independent generalization.
```

---

### 14.4. HWR200 robustness split

Для HWR200 важно сохранить acquisition condition.

Рекомендуемая схема:

```text
1. train на scan + good_light subset, если делаем HWR200 training;
2. test отдельно на scan;
3. test отдельно на good_light;
4. test отдельно на poor_light;
5. если есть одинаковый текст в разных условиях — создать paired robustness set.
```

---

### 14.5. split metadata

Файл:

```text
data/splits/{dataset}_splits.json
```

Формат:

```json
{
  "dataset": "cyrillic_handwriting",
  "split_version": "v1",
  "split_type": "random_no_writer_id",
  "seed": 42,
  "train": ["cyr_phrase_000001", "cyr_phrase_000002"],
  "val": ["cyr_phrase_010001"],
  "test": ["cyr_phrase_020001"],
  "notes": "writer_id unavailable"
}
```

---

## 15. Preprocessing streams

### 15.1. OCR-normalized preprocessing

Назначение:

```text
создать вход для image-only HTR baseline.
```

Pipeline:

```text
load image
→ convert to grayscale
→ remove alpha if exists
→ optional denoise
→ contrast normalization
→ optional deskew
→ resize to fixed height
→ preserve aspect ratio
→ pad width to batch-compatible size
→ save PNG
```

Default parameters:

```text
height = 64 px for CRNN baseline
max_width = configurable
padding_value = white
keep_aspect_ratio = true
```

Что можно делать в OCR stream:

```text
deskew;
contrast normalization;
stronger denoise;
height normalization.
```

---

### 15.2. Feature-preserving preprocessing

Назначение:

```text
создать вход для graph extraction.
```

Pipeline:

```text
load image
→ convert to grayscale
→ light background normalization
→ weak denoise only if needed
→ no aggressive deskew by default
→ preserve original scale or controlled scale
→ save grayscale feature image
→ optionally save soft binary preview
```

Default parameters:

```text
no aggressive deskew
no strong blur
preserve stroke width
preserve local intensity
save at original or near-original resolution
```

Что нельзя делать без отдельного режима:

```text
aggressive deskew;
strong morphological opening;
strong denoise;
hard resizing that changes stroke width too much;
thresholding that destroys thin strokes.
```

---

### 15.3. Почему нужны два потока

OCR-normalized image оптимизируется для распознавания текста. Feature-preserving image оптимизируется для сохранения структуры. Это разные цели.

Пример конфликта:

```text
OCR может выиграть от deskew,
но graph branch потеряет информацию о естественном наклоне и baseline geometry.
```

Поэтому каждый sample должен иметь:

```text
image_path
ocr_image_path
feature_image_path
```

---

## 16. Pilot preprocessing experiments

Перед полной обработкой всех данных нужно сделать pilot на небольшом subset.

### 16.1. Pilot subset

```text
100 IAM samples
100 Cyrillic Handwriting samples
100 HWR200 samples
100 HKR samples
```

Если времени мало:

```text
50 samples на датасет
```

---

### 16.2. Варианты preprocessing

```text
P0 original grayscale
P1 Otsu binarization preview
P2 Sauvola binarization preview
P3 adaptive Gaussian threshold preview
P4 feature-preserving + Sauvola
P5 OCR-normalized + deskew + resize
```

---

### 16.3. Метрики pilot

```text
foreground ratio
connected component count
estimated stroke width mean/std
skeleton pixel count
endpoint count
junction count
short branch ratio
visual quality score
warnings count
```

Цель pilot — выбрать default preprocessing для graph extraction, а не доказать финальную гипотезу.

---

## 17. Data audit report

Файл:

```text
data/reports/data_audit_report.md
```

Структура:

```text
1. Summary
2. Dataset availability
3. Dataset formats
4. Sample counts by dataset and level
5. Image size distributions
6. Text length distributions
7. Alphabet analysis
8. Writer metadata analysis
9. Quality flags summary
10. Duplicate analysis
11. Split strategy
12. Preprocessing pilot results
13. Risks and decisions
14. Final dataset inclusion table
```

---

## 18. Dataset stats

Файл:

```text
data/reports/dataset_stats.csv
```

Колонки:

```text
dataset
level
num_samples
num_writers
num_train
num_val
num_test
avg_width
avg_height
median_width
median_height
avg_text_len
median_text_len
min_text_len
max_text_len
num_empty_transcriptions
num_missing_images
num_bad_images
num_duplicates_exact
num_duplicates_near
num_low_contrast
num_blurred
num_mixed_script
num_unknown_chars
```

---

## 19. Dataset issues

Файл:

```text
data/reports/dataset_issues.csv
```

Колонки:

```text
sample_id
dataset
level
issue_type
severity
message
image_path
transcription
suggested_action
```

Severity:

```text
info
warning
error
critical
```

Suggested actions:

```text
keep
exclude_from_htr
exclude_from_graph
exclude_from_gold
manual_review
fix_metadata
```

---

## 20. Preview notebook

Файл:

```text
notebooks/01_data_preview.ipynb
```

Обязательные блоки:

```text
1. Load unified metadata.
2. Show random samples per dataset.
3. Show samples by quality flags.
4. Show longest and shortest transcriptions.
5. Show extreme aspect ratios.
6. Show OCR-normalized vs feature-preserving images.
7. Show preliminary binarization previews.
8. Show writer distribution where available.
9. Show alphabet distribution.
10. Select candidate gold subset samples.
```

---

## 21. Gold subset candidate selection на Этапе 1

На Этапе 1 ещё не размечаем gold subset полностью, но уже формируем candidate pool.

### 21.1. Candidate pool size

```text
IAM: 300 candidates
Cyrillic Handwriting: 400 candidates
HWR200: 300 candidates
HKR: 300 candidates
numeric/mixed: 150 candidates
hard/ambiguous: 150 candidates
```

Из этого на следующем этапе выбирается финальный gold subset.

---

### 21.2. Candidate selection criteria

```text
1. разные уровни сложности;
2. наличие петель;
3. наличие junctions;
4. наличие длинных соединений;
5. разные размеры текста;
6. разные acquisition conditions;
7. low/high contrast;
8. numeric and mixed content;
9. ambiguous connections;
10. representative clean samples.
```

---

## 22. Команды CLI

Нужно спроектировать команды так, чтобы pipeline был воспроизводимым.

### 22.1. Convert dataset

```bash
python -m src.datasets.convert --dataset iam --raw_dir data/raw/iam --out_dir data/processed/iam
python -m src.datasets.convert --dataset cyrillic_handwriting --raw_dir data/raw/cyrillic_handwriting --out_dir data/processed/cyrillic_handwriting
python -m src.datasets.convert --dataset hwr200 --raw_dir data/raw/hwr200 --out_dir data/processed/hwr200
python -m src.datasets.convert --dataset hkr --raw_dir data/raw/hkr --out_dir data/processed/hkr
```

### 22.2. Validate dataset

```bash
python -m src.datasets.validate_dataset --metadata data/processed/iam/metadata.jsonl --report_dir data/reports/iam
```

### 22.3. Create splits

```bash
python -m src.datasets.splits --metadata data/processed/iam/metadata.jsonl --out data/splits/iam_splits.json --split_type writer_independent --seed 42
```

### 22.4. Preprocess images

```bash
python -m src.preprocessing.ocr_preprocess --metadata data/processed/iam/metadata.jsonl --out_dir data/processed/iam/ocr_images
python -m src.preprocessing.feature_preprocess --metadata data/processed/iam/metadata.jsonl --out_dir data/processed/iam/feature_images
```

### 22.5. Generate report

```bash
python -m src.datasets.audit_report --metadata_dir data/processed --out data/reports/data_audit_report.md
```

---

## 23. Конфиги

### 23.1. Dataset config example

```yaml
name: iam
raw_dir: data/raw/iam
processed_dir: data/processed/iam
metadata_path: data/processed/iam/metadata.jsonl
splits_path: data/splits/iam_splits.json
language: en
script: latin
primary_level: line
use_writer_split: true
text_normalization:
  default_mode: ctc_default
  lowercase: true
  keep_punctuation: true
  keep_digits: true
preprocessing:
  ocr:
    height: 64
    deskew: true
    normalize_contrast: true
  feature:
    preserve_scale: true
    deskew: false
    light_denoise: true
```

---

### 23.2. HWR200 config example

```yaml
name: hwr200
raw_dir: data/raw/hwr200
processed_dir: data/processed/hwr200
metadata_path: data/processed/hwr200/metadata.jsonl
splits_path: data/splits/hwr200_splits.json
language: ru
script: cyrillic
primary_level: page
derived_levels:
  - sentence
  - word
use_writer_split: true
acquisition_conditions:
  - scan
  - good_light
  - poor_light
robustness:
  create_condition_subsets: true
  create_paired_sets: true
text_normalization:
  default_mode: ctc_default
  lowercase: true
  yo_policy: keep
  keep_punctuation: true
  keep_digits: true
```

---

## 24. Acceptance criteria

Этап 1 считается завершённым, если:

```text
1. Все core datasets имеют metadata.jsonl.
2. Для каждого sample есть sample_id, dataset, level, image_path, raw_transcription, normalized_transcription.
3. Для каждого core dataset создан dataset_stats.csv summary.
4. Ошибочные samples вынесены в dataset_issues.csv.
5. Создан alphabet_report.json.
6. Созданы train/val/test splits.
7. Где возможно, создан writer-independent split.
8. Для HWR200 созданы condition subsets: scan/good_light/poor_light.
9. Реализованы OCR-normalized и feature-preserving preprocessing streams.
10. Pilot preprocessing report показывает выбранные default параметры.
11. Создан preview notebook.
12. Сформирован candidate pool для gold subset.
13. Репозиторий не содержит raw/processed dataset files, которые нельзя публиковать.
14. Код можно опубликовать без датасетов.
```

---

## 25. Что не входит в Этап 1

```text
1. Обучение CRNN/CTC.
2. Полная скелетизация всех данных.
3. Построение HI-CSG-R для всех samples.
4. Ручная gold annotation.
5. Structural metrics.
6. Graph-only recognition.
7. Fusion.
8. Robustness experiments.
```

Этап 1 только готовит данные и проверяет, что дальнейшая работа имеет надёжную основу.

---

## 26. Ближайший порядок реализации

### Шаг 1. Создать структуру проекта

```text
configs/datasets/
src/datasets/
src/preprocessing/
data/raw/
data/processed/
data/splits/
data/reports/
notebooks/
```

### Шаг 2. Реализовать BaseDatasetConverter

Файлы:

```text
src/datasets/base.py
src/datasets/metadata.py
src/datasets/registry.py
```

### Шаг 3. Реализовать converter для самого простого RU датасета

Первым делать:

```text
Cyrillic Handwriting Dataset
```

Причина:

```text
готовые word/phrase crops, низкий риск, быстро даст первые данные для graph pipeline.
```

### Шаг 4. Реализовать IAM converter

Причина:

```text
нужен EN baseline и line-level HTR.
```

### Шаг 5. Реализовать validation

```text
image checks
text checks
metadata checks
duplicate checks
alphabet report
```

### Шаг 6. Реализовать splits

```text
random split
writer-independent split
condition split for HWR200
```

### Шаг 7. Реализовать preprocessing streams

```text
ocr_preprocess
feature_preprocess
preview comparison
```

### Шаг 8. Реализовать HWR200 converter

Причина:

```text
после базового pipeline можно подключать более сложный page/crop dataset.
```

### Шаг 9. Реализовать HKR converter

Причина:

```text
крупный line-level RU корпус после стабилизации конвертеров.
```

### Шаг 10. Сформировать отчёты

```text
data_audit_report.md
dataset_stats.csv
dataset_issues.csv
alphabet_report.json
preprocessing_preview_report.md
```

---

## 27. Вопросы, которые нужно решить перед кодом

Эти вопросы не блокируют составление плана, но блокируют точную реализацию scripts.

1. **Где физически будут лежать датасеты?**
   Предлагаемый default:

   ```text
   data/raw/{dataset_name}/
   ```

2. **Какая ОС и среда разработки?**
   Нужно знать для путей, multiprocessing и установки OpenCV/scikit-image.

3. **Будем ли сразу использовать DVC/Git LFS?**
   Моя рекомендация: raw/processed images не хранить в Git; DVC можно подключить позже, но `.gitignore` нужен сразу.

4. **Какой первый датасет скачан локально?**
   Я бы начал с Cyrillic Handwriting Dataset, затем IAM, затем HWR200 subset, затем HKR.

5. **Нужен ли Docker/conda environment с самого начала?**
   Моя рекомендация: сначала `pyproject.toml` + `requirements.txt`, Docker позже.

---

## 28. Решение по умолчанию

Если не уточнять дополнительные параметры, принимаем defaults:

```text
project_root = hi-csg-r/
raw_data_dir = data/raw/
processed_data_dir = data/processed/
splits_dir = data/splits/
reports_dir = data/reports/
random_seed = 42
ocr_image_height = 64
feature_preprocessing_deskew = false
ocr_preprocessing_deskew = true
primary_ru_dataset_first = Cyrillic Handwriting Dataset
primary_en_dataset_second = IAM
robustness_dataset_third = HWR200 subset
line_ru_dataset_fourth = HKR
```

---

## 29. Статус реализации: Cyrillic Handwriting Dataset

Первый converter для `cyrillic-handwriting-dataset` успешно отработал.

Фактический результат:

```text
metadata.jsonl:            73 830 samples
metadata.validated.jsonl:  73 830 samples
metadata.final.jsonl:      73 830 samples
```

Распределение по уровню:

```text
word:    66 060
phrase:   7 770
```

Первичный split на основе source train/test был заменён, потому что обнаружены exact-duplicate leakage между train/val/test.

Финальный clean split:

```text
train:    65 033
val:       7 232
test:      1 563
excluded:      2
```

Проверка leakage:

```text
hash leakage groups: 0
```

Full preprocessing выполнен:

```text
records:               73 830
missing ocr:                0
missing feature:            0
ocr files exist:       73 830
feature files exist:   73 830
```

Сохраняющиеся flags:

```text
duplicate_exact: 993
mixed_script: 4
empty_normalized_transcription: 2
empty_raw_transcription: 2
```

Интерпретация:

```text
1. Датасет пригоден для дальнейших этапов.
2. Empty samples исключены из финального split.
3. Exact duplicates остаются размеченными flags, но больше не создают split leakage.
4. Mixed-script samples остаются в metadata как manual-review/mixed cases.
5. OCR и feature preprocessing успешно созданы для всех samples.
```

Финальный файл для дальнейших этапов:

```text
data/processed/cyrillic_handwriting/metadata.preprocessed.jsonl
```

---

## 29.1. Preprocessing pilot status: Cyrillic Handwriting Dataset

Pilot preprocessing на 1000 samples успешно отработал.

Результат:

```text
records:          1000
missing ocr:      0
missing feature:  0
pilot flags:      0
```

Issue counts после validation:

```text
duplicate_exact: 993
mixed_script: 4
empty_normalized_transcription: 2
empty_raw_transcription: 2
```

Manual review mixed/empty cases:

```text
mixed_script:
  cyr_word_037736: ИhМ
  cyr_phrase_045319: компоненты x, y
  cyr_word_047800: R-эффективна
  cyr_word_062149: cержантам

empty_transcription:
  cyr_word_063653
  cyr_word_064128
```

Duplicate leakage check:

```text
duplicate groups:              895
duplicate samples:            1888
cross-split duplicate groups:   184
cross-split duplicate samples:  393
```

Интерпретация:

```text
1. empty_transcription samples исключаются из HTR, gold subset и финальных splits.
2. mixed_script samples не удаляются автоматически; они остаются как valid mixed/numeric или manual-review cases.
3. duplicate_exact нельзя игнорировать: обнаружены cross-split duplicates, значит текущий split потенциально даёт leakage.
4. Необходимо заменить source_train_test split на hash-group-aware split.
5. Exact duplicates должны попадать только в один split.
6. Если duplicate group содержит source test sample, вся группа отправляется в test.
7. Если duplicate group содержит только source train samples, группа целиком отправляется либо в train, либо в val.
8. Если duplicate group имеет разные транскрипции для одного и того же изображения, это annotation conflict и он должен попасть в duplicate_groups_report.csv.
```

Решение по OCR stream:

```text
OCR-P0: grayscale + alpha composite only — самый близкий к оригиналу, но слишком сырой для OCR.
OCR-P1: grayscale + autocontrast + resize — лучший баланс для OCR baseline.
OCR-P2: grayscale + autocontrast + median denoise + resize — местами сглаживает тонкие элементы.

Default OCR stream для Cyrillic Handwriting Dataset: OCR-P1.
Default feature stream: grayscale после alpha composite, без autocontrast, denoise и resize.
```

---

## 30. Финальное резюме этапа

Этап 1 должен превратить набор разнородных рукописных датасетов в контролируемую исследовательскую основу. Главный результат — не красивые изображения и не первая модель, а воспроизводимый слой данных: единые metadata, validation, splits, preprocessing streams, alphabet report, quality flags и preview notebook. Только после этого можно корректно переходить к image-only baseline и classical graph extractor.
