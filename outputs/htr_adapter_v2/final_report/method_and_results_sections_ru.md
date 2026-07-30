# Фрагменты метода и результатов для научной работы

## 2.X. Late correction HI-CSG-R

После отрицательного результата раннего x-aligned fusion v1 визуальная CRNN-CTC в v2 полностью замораживалась. Нормализованные локальные признаки HI-CSG-R маскировались после стандартизации, агрегировались masked pooling с окнами 1, 5 и 9 и преобразовывались в 128-мерное представление. Остаточная поправка добавлялась не к hidden sequence, а к baseline CTC logits.

Вклад графа ограничивался произведением non-empty mask, визуальной неопределенности, learned gate и bounded alpha. Для V2-2 дополнительно применялось фиксированное structural-risk attenuation. CNN, BiLSTM и baseline classifier не получали градиентов.

## 4.X. Результаты

H4-v2 не подтверждена. Оба разрешенных development-варианта не прошли заранее установленный dev gate: снижение CER было меньше 1%, а correct graph не превзошел matched shuffle. Согласно frozen protocol p10, holdout, final seeds и test не запускались. Отрицательный вывод v1 сохранен отдельно.

Результат v1 сохраняется отдельно: раннее слияние было технически работоспособно и sample-specific, но уступило matched image-only fine-tuning.
