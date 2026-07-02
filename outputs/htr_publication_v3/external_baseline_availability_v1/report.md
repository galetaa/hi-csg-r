# External Baseline Availability v1

## Python Packages

| package | available | version/error |
|---|---:|---|
| `easyocr` | False | ModuleNotFoundError |
| `pytesseract` | False | ModuleNotFoundError |
| `tesserocr` | False | ModuleNotFoundError |
| `kraken` | False | ModuleNotFoundError |
| `keras_ocr` | False | ModuleNotFoundError |
| `doctr` | False | ModuleNotFoundError |
| `paddleocr` | False | ModuleNotFoundError |
| `transformers` | True | 4.57.1 |
| `torch` | True | 2.9.0+cu128 |
| `cv2` | True | 4.12.0 |

## CLI Tools

| tool | available | path | version |
|---|---:|---|---|
| `tesseract` | False |  |  |
| `kraken` | False |  |  |
| `calamari-predict` | False |  |  |
| `paddleocr` | False |  |  |

## Cached HuggingFace Models

- `bert-base-uncased`
- `microsoft/trocr-base-handwritten`

## Existing Baseline Results

| baseline | n | CER | WER | exact | interpretation |
|---|---:|---:|---:|---:|---|
| `external_trocr_zero_shot_full` | 5563 | 1.2985 | 1.4753 | 0.0040 | external |
| `external_trocr_finetuned_tri10k_base_test` | 5563 | 1.2657 | 1.0343 | 0.0043 | external |
| `mixed_cyrillic_natural_full_v1` | 5563 | 0.0822 | 0.3350 | 0.6245 | internal CRNN positioning baseline |
| `mixed_cyrillic_balanced50k_v1` | 5563 | 0.0979 | 0.3853 | 0.5774 | internal CRNN positioning baseline |

## Publication Interpretation

- external baseline available locally: True
- competitive external Russian/Cyrillic baseline available locally: False
- prepared EasyOCR wrapper: `tools/evaluate_easyocr_baseline_v1.py`
- EasyOCR command after install: `python tools/evaluate_easyocr_baseline_v1.py --manifest data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1/test.jsonl --out_dir outputs/htr_publication_v3/external_easyocr_page_disjoint_test_v1`
- boundary: Only TrOCR-base-handwritten is cached locally as an external HTR/OCR model. The completed external TrOCR zero-shot and decoder-only adaptation baselines are weak. No EasyOCR, Tesseract, Kraken, PaddleOCR, docTR, or Calamari runtime is available locally.

To close the gap:
- Install/evaluate a suitable external Cyrillic/Russian OCR/HTR system on the same test protocol.
- For EasyOCR specifically, run the prepared wrapper after installing the package and model weights.
- Or download a relevant HuggingFace/Russian OCR checkpoint and run it on the fixed manifests.
- If no such model is available, report TrOCR as a negative/limited external baseline and avoid SOTA claims.
