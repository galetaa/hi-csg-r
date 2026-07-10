# Машинно-проверяемая цепочка evidence

Цепочка устроена так:

```text
v11 / research/claims.yaml
  → research/evidence.yaml (число → JSON selector → допуск)
  → frozen summaries и predictions
  → research/artifact_inventory.json (SHA-256)
  → reproduce-lite и regenerated CSV
```

Основные связи:

| Вывод | Источник | Проверка |
|---|---|---|
| H2, CER по трём seed | `seed_confirmation_summary.json` | исходные строки и независимое среднее |
| H2 по доменам | `domainwise_seed_confirmation.json` | regenerated domain table |
| H1, n=200 | `annotation_summary.json` | n, rates, acceptance |
| H3, risk AUC | `selective_summary.json` | feature/confidence/confidence+graph |
| H3, operating point | `operating_points.json` | coverage и risk на фиксированном режиме |
| H4 | graph-fusion summary/table | checksum, exploratory boundary |
| page-disjoint controls | manifests, predictions, summaries | 68 файлов внутри inventory |

Inventory генерируется только явной командой:

```bash
uv run python -m src.pipeline build-inventory
```

Эту команду нельзя запускать для «исправления» неожиданного checksum mismatch.
Сначала нужно установить причину изменения и оформить новый научный snapshot.
После намеренного добавления нового immutable evidence обновляются одновременно
`research/artifacts.yaml`, inventory, claims/evidence и документация.
