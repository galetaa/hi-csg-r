# HTR baseline report — mini10k_cyrillic_v1

## Setup

```text
decode blank penalty selected on validation: -0.9
model: height-preserving CRNN + BiLSTM + CTC
dataset: Cyrillic Handwriting
subset: mini10k
input: OCR-preprocessed images
target: transcription_modes.ctc_default
blank penalty: scheduled during training
```

## Metrics

| split | n | CER | WER | exact | pred_len | empty | blank |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 10000 | 0.0005 | 0.0022 | 0.9975 | 7.41 | 0.000 | 0.822 |
| val | 2000 | 0.1602 | 0.5506 | 0.4390 | 7.33 | 0.000 | 0.828 |
| test | 1563 | 0.2556 | 0.7890 | 0.1708 | 9.28 | 0.000 | 0.840 |

## Interpretation

The mini10k run confirms that the CRNN-CTC pipeline is viable. The model strongly overfits the 10k training subset, but validation performance is already usable for a first image-only baseline. This run should be treated as a development baseline, not the final full-dataset result.

## Example predictions

| target | pred | CER |
|---|---|---:|
| `деформация` | `дефоормация` | 0.100 |
| `линейное` | `лиейное` | 0.125 |
| `используем` | `исполрусм` | 0.300 |
| `дисперсия` | `испереия` | 0.222 |
| `принцип` | `прикцин` | 0.286 |
| `порядок` | `поодок` | 0.286 |
| `равномерно` | `равломерно` | 0.100 |
| `осуществляется` | `оусоствяется` | 0.286 |
| `через` | `через` | 0.000 |
| `имеем` | `имсем` | 0.200 |
| `линейных` | `линейных` | 0.000 |
| `метод` | `мепод` | 0.200 |
| `моделей` | `моделей` | 0.000 |
| `тогда` | `пояда` | 0.400 |
| `сверху` | `свереу` | 0.167 |
| `ерез` | `еряз` | 0.250 |
| `линейная` | `лнчейная` | 0.250 |
| `вероятность` | `версхонеть` | 0.455 |
| `схема` | `схема` | 0.000 |
| `ибо` | `ибо` | 0.000 |
| `осталось` | `осталось` | 0.000 |
| `поле` | `поге` | 0.250 |
| `оптическое` | `оптическое` | 0.000 |
| `1 класса` | `1 киасса` | 0.125 |
| `г. ульяновск` | `2 лояновск` | 0.333 |
| `на место` | `но место` | 0.125 |
| `паспорт` | `потслоря` | 0.571 |
| `назначение` | `назндчения` | 0.200 |
| `отправление` | `отпровление` | 0.091 |
| `было только` | `босло только` | 0.182 |