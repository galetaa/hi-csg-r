# Publication V3 Remaining Addendum v1

## Strong Internal Baselines

The only cached external HuggingFace OCR/HTR model found is TrOCR-base-handwritten. The additional strong baselines are internal CRNN baselines trained on larger in-domain data; they are useful for positioning but are not external SOTA.

| baseline | n | CER | WER | exact | checkpoint epoch | status |
|---|---:|---:|---:|---:|---:|---|
| `mixed_cyrillic_natural_full_v1` | 5563 | 0.0822 | 0.3350 | 0.6245 | 46 | complete |
| `mixed_cyrillic_balanced50k_v1` | 5563 | 0.0979 | 0.3853 | 0.5774 | 47 | complete |

Cached HuggingFace models: `['bert-base-uncased', 'microsoft/trocr-base-handwritten']`.

External baseline availability:
- report: `outputs/htr_publication_v3/external_baseline_availability_v1/report.md`
- competitive external Russian/Cyrillic baseline available locally: False
- prepared EasyOCR wrapper: `tools/evaluate_easyocr_baseline_v1.py`
- EasyOCR command after install: `python tools/evaluate_easyocr_baseline_v1.py --manifest data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1/test.jsonl --out_dir outputs/htr_publication_v3/external_easyocr_page_disjoint_test_v1`
- boundary: Only TrOCR-base-handwritten is cached locally as an external HTR/OCR model. The completed external TrOCR zero-shot and decoder-only adaptation baselines are weak. No EasyOCR, Tesseract, Kraken, PaddleOCR, docTR, or Calamari runtime is available locally.

## Page-Disjoint HKR+School Split

- base manifest root: `data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1`
- line manifest root: `data/experiments/htr_publication_v3/page_disjoint_hkr_school_plus_lines_10k_v1`
- base train/val/test n: 20000/4000/4000
- line train n: 28014 (line samples selected: 8014)
- train-vs-test page overlap: []
- cyrillic limitation: cyrillic_handwriting is excluded because it has no page_id/source_image_file/writer_id metadata.
- full retrain command: `python -u tools/run_page_disjoint_hkr_school_v1.py --seeds 42 43 44 --epochs 80 --num_workers 4`
- control retrain command: `python -u tools/run_page_disjoint_hkr_school_v1.py --variants page_random_crops_8k_control page_school_words_8k_control --seeds 42 43 44 --epochs 80 --num_workers 4`
- control comparison command: `python tools/build_page_disjoint_control_comparisons_v1.py`

Page-disjoint same-size control manifests:

| control | train n | added n | train-vs-test page overlap | ready |
|---|---:|---:|---|---:|
| `page_random_crops_8k_control` | 28014 | 8014 | [] | True |
| `page_school_words_8k_control` | 28014 | 8014 | [] | True |

| variant | seed | last epoch | best exists | eval returncode | status |
|---|---:|---:|---|---:|---|
| `page_random_crops_8k_control` | 42 | 80 | True | 0 | complete |
| `page_random_crops_8k_control` | 43 | 80 | True | 0 | complete |
| `page_random_crops_8k_control` | 44 | 80 | True | 0 | complete |
| `page_school_words_8k_control` | 42 | 80 | True | 0 | complete |
| `page_school_words_8k_control` | 43 | 80 | True | 0 | complete |
| `page_school_words_8k_control` | 44 | 80 | True | 0 | complete |

Page-disjoint fixed-penalty evaluation:

| variant | seed | n | CER | WER | exact | checkpoint epoch |
|---|---:|---:|---:|---:|---:|---:|
| `page_base` | 42 | 4000 | 0.1429 | 0.4617 | 0.4135 | 78 |
| `page_base` | 43 | 4000 | 0.1620 | 0.5113 | 0.3573 | 79 |
| `page_base` | 44 | 4000 | 0.1400 | 0.4562 | 0.4130 | 64 |
| `page_line_10k` | 42 | 4000 | 0.1330 | 0.4365 | 0.4373 | 53 |
| `page_line_10k` | 43 | 4000 | 0.1217 | 0.4122 | 0.4640 | 70 |
| `page_line_10k` | 44 | 4000 | 0.1265 | 0.4193 | 0.4552 | 69 |
| `page_random_crops_8k_control` | 42 | 4000 | 0.1253 | 0.4214 | 0.4510 | 75 |
| `page_random_crops_8k_control` | 43 | 4000 | 0.1332 | 0.4403 | 0.4325 | 50 |
| `page_random_crops_8k_control` | 44 | 4000 | 0.1367 | 0.4467 | 0.4253 | 44 |
| `page_school_words_8k_control` | 42 | 4000 | 0.1211 | 0.4126 | 0.4585 | 69 |
| `page_school_words_8k_control` | 43 | 4000 | 0.1217 | 0.4117 | 0.4627 | 62 |
| `page_school_words_8k_control` | 44 | 4000 | 0.1202 | 0.4086 | 0.4600 | 71 |

Page-disjoint aggregate:

| variant | completed seeds | mean CER | std CER | mean WER | mean exact |
|---|---|---:|---:|---:|---:|
| `page_base` | [42, 43, 44] | 0.1483 | 0.0119 | 0.4764 | 0.3946 |
| `page_line_10k` | [42, 43, 44] | 0.1271 | 0.0057 | 0.4227 | 0.4522 |
| `page_random_crops_8k_control` | [42, 43, 44] | 0.1317 | 0.0058 | 0.4362 | 0.4363 |
| `page_school_words_8k_control` | [42, 43, 44] | 0.1210 | 0.0007 | 0.4110 | 0.4604 |

Mean `page_line_10k - page_base` delta: CER -0.0212, WER -0.0537, exact 0.0576.

Mean `page_line_10k - control` deltas:

| control | delta CER | delta WER | delta exact |
|---|---:|---:|---:|
| `page_random_crops_8k_control` | -0.0047 | -0.0135 | 0.0159 |
| `page_school_words_8k_control` | 0.0061 | 0.0117 | -0.0083 |

Page-disjoint paired line-vs-base comparison:

| seed | n | delta CER | 95% CI | School delta CER | School 95% CI | delta WER | delta exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4000 | -0.0099 | [-0.0147, -0.0051] | -0.0180 | [-0.0263, -0.0098] | -0.0253 | 0.0238 |
| 43 | 4000 | -0.0403 | [-0.0450, -0.0357] | -0.0396 | [-0.0482, -0.0316] | -0.0990 | 0.1068 |
| 44 | 4000 | -0.0135 | [-0.0181, -0.0089] | -0.0177 | [-0.0255, -0.0097] | -0.0369 | 0.0423 |

Page-disjoint paired line-vs-control comparison:

| comparison | seed | n | delta CER | 95% CI | School delta CER | School 95% CI | delta WER | delta exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `line_vs_random_crops_control` | 42 | 4000 | 0.0077 | [0.0031, 0.0126] | 0.0066 | [-0.0017, 0.0149] | 0.0150 | -0.0137 |
| `line_vs_random_crops_control` | 43 | 4000 | -0.0115 | [-0.0159, -0.0070] | -0.0196 | [-0.0273, -0.0119] | -0.0280 | 0.0315 |
| `line_vs_random_crops_control` | 44 | 4000 | -0.0102 | [-0.0149, -0.0054] | -0.0168 | [-0.0253, -0.0088] | -0.0274 | 0.0300 |
| `line_vs_school_words_control` | 42 | 4000 | 0.0119 | [0.0073, 0.0168] | 0.0168 | [0.0084, 0.0252] | 0.0238 | -0.0212 |
| `line_vs_school_words_control` | 43 | 4000 | 0.0001 | [-0.0045, 0.0046] | 0.0011 | [-0.0073, 0.0094] | 0.0005 | 0.0013 |
| `line_vs_school_words_control` | 44 | 4000 | 0.0063 | [0.0019, 0.0106] | 0.0081 | [0.0004, 0.0158] | 0.0107 | -0.0048 |

## Annotation Reliability

Full report: `outputs/htr_publication_v3/annotation_reliability_addendum_v1/report.md`.
Repeated annotation overlap n: 40.

| field | agreement | kappa | weighted kappa |
|---|---:|---:|---:|
| `audit_usable` | 0.900 | 0.286 | n/a |
| `ink_visible_ok` | 0.875 | -0.064 | n/a |
| `skeleton_follows_ink` | 0.925 | 0.625 | n/a |
| `missed_visible_stroke` | 1.000 | 1.000 | n/a |
| `spurious_stroke` | 0.971 | 0.653 | n/a |
| `endpoint_error` | 0.946 | 0.471 | n/a |
| `junction_error` | 0.811 | 0.000 | n/a |
| `loop_error` | 1.000 | 1.000 | n/a |
| `critical_topology_error` | 0.900 | 0.444 | n/a |
| `graph_quality_0_3` | 0.800 | 0.350 | 0.612 |

Interpretation: A formal inter-annotator agreement claim is not supported until a genuinely independent second annotator fills the blind package and the scoring report shows adequate agreement.

Independent annotation package:
- package ready: True
- browser: `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_browser.html`
- expected filled CSV: `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_second_filled.csv`
- minimally complete rows: 0
- formal IAA ready: False

## Remaining Boundary

- Added: page-disjoint HKR+School manifests with zero train/val/test page overlap
- Added: page-disjoint line augmentation restricted to train pages
- Added: page-disjoint same-size random-crop and School-word control manifests
- Added: completed 3-seed page-disjoint base-vs-line retraining
- Added: completed 3-seed page-disjoint same-size controls
- Added: annotation repeated-consistency and line-quality Wilson intervals
- Added: blind second-annotation package for formal IAA
- Added: strong data-rich internal CRNN baselines on the same tri10k test
- Still not fully solved: formal independent inter-annotator agreement
- Still not fully solved: competitive external Russian/Cyrillic HTR baseline beyond cached TrOCR
- Claim boundary: The strict page-disjoint base, line, and same-size controls are complete. A unique natural-line-context claim is allowed only if the paired line-vs-control deltas support it.
