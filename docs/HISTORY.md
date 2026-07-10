# Замороженная история исследования

Полный машинный реестр расположен в
[`research/milestones.yaml`](../research/milestones.yaml). История линейная,
содержит 71 исходный коммит и сохраняется без squash/rebase.

| Tag | Commit | Научный этап |
|---|---|---|
| `milestone/problem-definition-v1` | `d5ad781` | Постановка проблемы и первоначальных гипотез |
| `milestone/data-graph-pilot-v1` | `44af7c3` | Аудит данных и классический graph pilot |
| `milestone/htr-baselines-v1` | `e73b7f1` | Стабильные CRNN-CTC baseline |
| `milestone/first-evidence-freeze-v1` | `3fc7974` | Первый замороженный evidence package |
| `milestone/final-evidence-v1` | `35fcb36` | Финальный экспериментальный пакет v1 |
| `milestone/publication-v3-final` | `d9f8bde` | Same-size/page-disjoint controls и итоговая хронология |

## Политика сохранения

- Теги annotated и указывают на неизменяемые исторические состояния.
- Старые отчёты сохраняют исходные формулировки, даже если поздние контроли их уточнили.
- `chats/` сохраняет первичную историю решений.
- `archive/legacy_tools/` сохраняет все старые экспериментальные сценарии.
- Старую команду воспроизводят на milestone tag, где существовал её исходный путь.
- Новое научное толкование добавляется в `research/claims.yaml`, а не переписывает прошлое.

