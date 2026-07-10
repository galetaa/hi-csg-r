# Воспроизводимость

Воспроизводимость разделена на три разных утверждения. Это важно для честной
интерпретации завершённой научной работы.

1. **Evidence reproduction** — проверить неизменность рукописи, манифестов,
   predictions и summaries, пересчитать ключевые агрегаты. Работает без GPU и
   исходных изображений.
2. **Evaluation reproduction** — заново оценить сохранённые веса на тех же
   изображениях. Сейчас заблокировано: model weights и изображения не находятся
   в репозитории.
3. **Training reproduction** — повторить подготовку данных и обучение на seed
   42/43/44. Рецепт и историческое окружение сохранены, но данные необходимо
   восстановить по условиям их владельцев.

## Уровень 1: проверка evidence без датасетов

```bash
uv sync
uv run python -m src.pipeline verify
uv run python -m src.pipeline reproduce-lite
uv run python -m src.pipeline regenerate-tables
uv run pytest
```

`reproduce-lite` выполняет числовые проверки из
[`research/evidence.yaml`](../research/evidence.yaml) и сверяет SHA-256 84
канонических артефактов из
[`research/artifact_inventory.json`](../research/artifact_inventory.json).
Проверяются, среди прочего:

- средние CER baseline и +10k и их независимый пересчёт по трём seed;
- размер и acceptance структурной проверки H1;
- ROC-AUC трёх вариантов selective prediction;
- versioned page-disjoint manifests, predictions, summaries, configs и histories;
- SHA-256 финальной рукописи v11.

`regenerate-tables` строит четыре CSV в `reproducibility/generated/` напрямую
из JSON evidence. Эти файлы являются производными и могут быть регенерированы в
любой момент.

Предупреждения (`WARN`) не скрываются. Зафиксированы single-seed характер H4,
отсутствие отдельного точного машинного агрегата для части округлённых
page-disjoint строк v11 и неполный комплект canonical prediction files для
основной six-run пары baseline/+10k. Это не превращается автоматически в ошибку
всей работы, но такие строки не объявляются перепроверенными от исходной пары
`target/prediction`.

## Уровень 2/3: готовность полного воспроизведения

```bash
uv run python -m src.pipeline audit-data
uv run python -m src.pipeline reproduce-full
```

На 10 июля 2026 года ожидаемый результат на этом устройстве — `BLOCKED` и код
возврата `2`: отсутствуют три набора, необходимые для финального обучения
(Cyrillic, HKR Words и School Notebooks). Веса также отсутствуют: это блокирует
повторную evaluation, но не обучение с нуля после восстановления данных.
Наличие старого `data/reports/raw_inventory.json` означает, что данные были
доступны в историческом запуске, но не доказывает их наличие сейчас.

После восстановления данных `audit-data` показывает найденные root directories.
Перед обучением дополнительно нужны checksum исходных архивов; их следует
записать командой `snapshot` в локальный, не обязательно публикуемый provenance.

## Восстановление исторического кода

Экспериментальные сценарии сохранены без удаления в `archive/legacy_tools`, а
их первоначальное расположение — в milestone tag. Для точного исторического
layout безопаснее создать отдельный worktree:

```bash
git worktree add ../hi-csg-r-publication-v3 milestone/publication-v3-final
cd ../hi-csg-r-publication-v3
uv sync
```

Не изменяйте milestone tag. Новую попытку следует вести в отдельной ветке и
сохранять рядом config, history, predictions, summary и provenance snapshot.

## Окружение и детерминизм

- Текущее CPU-safe окружение закреплено `uv.lock` и Python 3.11.
- Фактический поздний Linux snapshot: Python 3.11.12, Torch 2.9.0+cu128,
  NumPy 2.2.6; полный список — `outputs/htr_publication_v3/pip_freeze.txt`.
- Исторический snapshot был сделан при dirty working tree. Это явно отмечено в
  [`research/environment_profiles.yaml`](../research/environment_profiles.yaml).
- Seed policy фиксирует Python, NumPy и Torch RNG. CUDA может не давать битовой
  идентичности между версиями драйвера/устройствами; поэтому научный критерий —
  повторение протокола по seed и отчёт распределения, а не побитовое совпадение весов.

## Smoke-test без научных данных

`reproducibility/smoke/` содержит искусственное PGM-изображение и manifest.
Тест проверяет image I/O, manifest, skeletonization и graph construction. Он не
использует и не подменяет научные датасеты.

## Критерии приёмки нового запуска

- dataset revision, лицензия, checksum архивов и split hashes записаны;
- train/validation/test leakage checks пройдены до обучения;
- seed и параметры декодирования сохранены, test не использован для настройки;
- сохранены config, environment, history, predictions и summary;
- таблицы построены из machine-readable evidence, а не перепечатаны вручную;
- расхождения с v11 оформлены новой записью, исторические outputs не перезаписаны.
