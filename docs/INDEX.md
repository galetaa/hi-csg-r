# Единая документация HI-CSG-R

Этот индекс отделяет актуальное состояние завершённой работы от исторических
планов и промежуточных отчётов.

## Актуальные документы

1. [Финальная рукопись v11](../article/HI_CSG_R_v11.docx).
2. [Реестр научных утверждений](../research/claims.yaml).
3. [Человекочитаемая матрица утверждений](CLAIMS.md).
4. [Архитектура и канонический pipeline](ARCHITECTURE.md).
5. [Замороженная история и milestones](HISTORY.md).
6. [Воспроизводимость](REPRODUCIBILITY.md).
7. [Датасеты и их восстановление](DATASETS.md).
8. [Цепочка evidence и checksum](EVIDENCE.md).

## Исторические источники

- [`00_research_problem.md`](00_research_problem.md) — первоначальная постановка.
- [`01_data_audit_and_preprocessing.md`](01_data_audit_and_preprocessing.md) — аудит данных.
- [`02_stage2_checkpoint_report.md`](02_stage2_checkpoint_report.md) — graph pilot.
- [`final_experimental_protocol_v1.md`](final_experimental_protocol_v1.md) — поздний протокол до v11.
- [`research_execution_chronology_v1.md`](research_execution_chronology_v1.md) — подробная реконструкция работы.
- [`../chats/`](../chats/) — первичная история решений и выполнения.

Исторические документы не переписываются под текущий вывод. При расхождении
приоритет имеют v11 и `research/claims.yaml`.

## Evidence-пакеты

- [`../outputs/final_result_package_v1/`](../outputs/final_result_package_v1/) — thesis tables и проверки provenance.
- [`../outputs/htr_publication_v3/`](../outputs/htr_publication_v3/) — same-size, leakage и page-disjoint controls.
- [`../outputs/iter2_structural_gold_v1/`](../outputs/iter2_structural_gold_v1/) — структурная диагностика.
