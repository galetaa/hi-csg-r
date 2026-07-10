# Источники данных и восстановление

Машиночитаемый источник истины —
[`research/datasets.yaml`](../research/datasets.yaml). На текущем устройстве нет
ни raw, ни interim/processed изображений. Таблица ниже описывает способ их
легального восстановления, а не утверждает, что они уже скачаны.

| Dataset | Revision/version | Доступ | Ожидаемый локальный root |
|---|---|---|---|
| Cyrillic Handwriting | Kaggle v4 | Kaggle account/API | `data/raw/cyrillic-handwriting-dataset` |
| HKR Words / Forms | commit `2e7a86a…` | ручная заявка авторам | `data/raw/hkr` |
| School Notebooks RU | HF `a10cd261…` | публичный | `data/raw/school_notebooks` |
| HWR200 | HF `366a882…` | публичный | `data/raw/hwr200` |
| IAM 3.0 | version 3.0 | регистрация, non-commercial research | `data/raw/iam` |

## Публичные источники

HWR200 и School Notebooks можно получить с Hugging Face, обязательно закрепляя
revision из registry. Для больших файлов следует использовать официальный
`huggingface-cli`/Git LFS и проверить, что checkout соответствует указанному
commit. Cyrillic Handwriting опубликован в Kaggle; воспроизведение использует
version 4 и требует настроенной Kaggle authentication.

## Ограниченные источники

IAM требует регистрации и разрешает использование для некоммерческого
исследования. HKR требует заполнить application form и получить ссылку от
авторов; лицензия запрещает свободное перераспространение. Поэтому проект не
содержит downloader, архивы или зеркала этих наборов.

## После загрузки

1. Сохранить архивы в ожидаемом `data/raw/...` без коммита в Git.
2. Записать URL, revision, размер и SHA-256 каждого архива.
3. Выполнить исторические converters из отдельного worktree milestone tag.
4. Сравнить counts с `data/reports/*/dataset_stats.json` и summary reports.
5. Проверить split leakage и page/writer independence.
6. Сохранить hashes итоговых manifests в provenance нового запуска.

`data/reports/raw_inventory.json` — исторический снимок прежнего устройства. Он
полезен для ожидаемых имён архивов, но не заменяет checksum и текущий аудит.
