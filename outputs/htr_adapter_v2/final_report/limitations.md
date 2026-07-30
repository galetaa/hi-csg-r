# Ограничения HI-CSG-R Late Correction v2

- V1 validation уже использовалась исследовательски; V2 применяет новый group-aware train/dev/holdout split.
- Structural risk attenuation является фиксированным proxy, а не вероятностью корректности графа.
- Zero-graph является dependency control, но не fair image-only baseline.
- Научный вывод запрещено делать по smoke, train loss, gate variability или одному development run.
- Если holdout gate возвращает STOP, canonical test остается закрытым.
- Результат относится к существующему extractor, 20 x-aligned признакам и зафиксированным русскоязычным доменам.
