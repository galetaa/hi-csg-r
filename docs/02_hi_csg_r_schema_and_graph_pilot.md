# Этап 2. HI-CSG-R schema and classical graph extraction pilot

**Проект:** HI-CSG-R  
**Статус:** начало graph pipeline  
**Назначение:** формально определить графовое представление HI-CSG-R v1, создать pilot subset и подготовить classical graph extraction pipeline.

---

## 1. Цель этапа

Цель Этапа 2 — перейти от подготовленных изображений к воспроизводимому графовому представлению видимой штриховой структуры рукописи.

Pipeline этапа:

```text
feature_image
→ binary mask
→ skeleton
→ raw pixel graph
→ canonical stroke graph
→ graph.json
→ SVG/PNG overlay
→ diagnostics.json