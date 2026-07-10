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

## Автоматизатор

Все действия собраны в одном сценарии
[`scripts/download_datasets.py`](../scripts/download_datasets.py). Без
`--execute` команды `download` и `extract` работают как dry-run.

Подготовка окружения и просмотр плана:

```bash
uv sync --group data-download
uv run python -m scripts.download_datasets plan
```

Скачивание публичных наборов с закреплёнными revisions:

```bash
# Около 45.5 GB; revision HWR200 закреплён в research/datasets.yaml.
uv run python -m scripts.download_datasets download hwr200 --execute

# Около 3.1 GB.
uv run python -m scripts.download_datasets download school_notebooks --execute

# Kaggle version 4; CLI может потребовать настроенную authentication.
uv run python -m scripts.download_datasets download cyrillic_handwriting --execute
```

Для возобновления или обновления уже непустого каталога нужно явно передать
`--force`. Перед этим рекомендуется сохранить его checksum.

Распаковка ZIP/TAR/TGZ в ожидаемые `data/interim`:

```bash
uv run python -m scripts.download_datasets extract hwr200 school_notebooks --execute
uv run python -m scripts.download_datasets extract iam --execute
```

RAR-архивы HKR автоматически не распаковываются: для них нужен отдельно
установленный совместимый распаковщик и соблюдение условий лицензии.

Проверка вручную полученных IAM/HKR архивов:

```bash
uv run python -m scripts.download_datasets manual-check iam hkr_words hkr_forms
```

Фиксация локальных SHA-256 после загрузки:

```bash
uv run python -m scripts.download_datasets checksum \
  cyrillic_handwriting school_notebooks hwr200 iam hkr_words
```

Результаты записываются в игнорируемый Git каталог `data/local_provenance/`.
Файл содержит source revision, размеры и SHA-256 всех файлов, кроме служебного
HF cache. Не запускайте checksum одновременно со скачиванием.

Автоматизатор не запускает дорогое обучение и не изменяет frozen evidence.
После подготовки interim-данных необходимо использовать converters и проверки
split из соответствующего milestone worktree.
