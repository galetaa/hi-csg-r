# Архитектура завершённого проекта

Проект разделён на три слоя.

## 1. Каноническое научное ядро

- `src/datasets` — metadata, converters, нормализация и split.
- `src/preprocessing` — обработка изображений и School foreground.
- `src/graph` — бинаризация, скелетизация и HI-CSG-R.
- `src/htr` — CRNN-CTC, decoding и метрики.
- `src/pipeline` — проверка завершённого исследования и provenance.

`src/pipeline` намеренно мал. Он не скрывает прошлые эксперименты и не запускает
дорогое переобучение автоматически; его задача — проверить, что финальная
рукопись, claims, evidence и milestone tags согласованы.

## 2. Декларативное состояние исследования

- `research/claims.yaml` — что доказано и с какими ограничениями.
- `research/milestones.yaml` — какие коммиты заморожены как этапы.
- `research/pipeline.yaml` — роли стадий и статус их реализации.
- `research/datasets.yaml` — происхождение, revision, лицензии и локальная доступность данных.
- `research/evidence.yaml` — проверяемые числа v11 и известные evidence gaps.
- `research/artifact_inventory.json` — SHA-256 канонических артефактов.

## 3. Исторический слой

- `archive/legacy_tools` — одноразовые runners, builders, browsers и report generators.
- `outputs` — замороженные результаты.
- `chats` — история формирования решений.

Legacy-слой не входит в canonical lint/test scope. Это архив полной реализации,
а не библиотечный интерфейс.

## Канонические команды

```bash
uv run python -m src.pipeline status
uv run python -m src.pipeline verify
uv run python -m src.pipeline reproduce-lite
uv run python -m src.pipeline audit-data
uv run python -m src.pipeline reproduce-full
uv run python -m src.pipeline regenerate-tables
uv run python -m src.pipeline validate-manifest MANIFEST.jsonl
uv run python -m src.pipeline snapshot --out provenance.json
uv run pytest
```
