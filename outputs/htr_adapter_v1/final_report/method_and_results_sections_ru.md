# Фрагменты для текста работы

## 2.X. Локальное выравнивание HI-CSG-R с временной осью CRNN-CTC

HI-CSG-R преобразуется в 20-мерную последовательность локальных признаков,
выровненную с временными шагами CRNN-CTC. Temporal adapter и quality-aware
residual gate добавляют структурное представление перед существующим BiLSTM.
Вспомогательная graph CTC objective используется только при обучении.

## 3.X. Сравнение image-only и локально структурно усиленной CRNN-CTC

На seed 42 сравнивались matched image-only fine-tuning (M0-FT), полный
x-aligned HI-CSG-R adapter (M3) и matched shuffled-graph control. Использовались
фиксированный blank penalty -0.4 и выбор checkpoint только по validation
micro-CER. Переход к M2 и дополнительным seeds был разрешён только при успешном
прохождении заранее зафиксированного validation gate.

## 4.X. Локальное слияние HI-CSG-R с CRNN-CTC

M0-FT получил validation CER 0.079537, а M3 — 0.082196, что соответствует
ухудшению на 3.342% relative. M3 ухудшил CER во всех трёх основных validation
доменах. Правильный граф был немного лучше matched shuffled graph, а gate и
градиенты не коллапсировали, однако основной критерий превосходства над M0-FT и
доменный критерий не были выполнены.

H4 остаётся поисковой; локальное структурное слияние не продемонстрировало
устойчивого превосходства над matched image-only fine-tuning. Согласно frozen
stopping rules M2, seeds 43/44 и test evaluation не запускались.
