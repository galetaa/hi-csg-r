# Publication V3 Status Report

## Plan

Full hardening checklist: `docs/publication_hardening_plan_v1.md`.

## Normalized Fixed-Penalty Evaluation

All rows below use fixed test-time `blank_logit_penalty=-0.4`.

| variant | seed | CER | WER | exact | checkpoint epoch | status |
|---|---:|---:|---:|---:|---:|---|
| `tri10k_base` | 42 | 0.1461 | 0.5155 | 0.4383 | 72 | complete |
| `tri10k_base` | 43 | 0.1494 | 0.5147 | 0.4354 | 74 | complete |
| `tri10k_base` | 44 | 0.1629 | 0.5575 | 0.3960 | 68 | complete |
| `line_context_10k` | 42 | 0.1353 | 0.4929 | 0.4629 | 73 | complete |
| `line_context_10k` | 43 | 0.1372 | 0.4935 | 0.4584 | 76 | complete |
| `line_context_10k` | 44 | 0.1348 | 0.4828 | 0.4710 | 79 | complete |
| `random_crops_10k_control` | 42 | 0.1339 | 0.4890 | 0.4661 | 75 | complete |
| `random_crops_10k_control` | 43 | 0.1372 | 0.4880 | 0.4636 | 62 | complete |
| `random_crops_10k_control` | 44 | 0.1365 | 0.4913 | 0.4634 | 60 | complete |
| `school_words_10k_control` | 42 | 0.1354 | 0.4864 | 0.4665 | 77 | complete |
| `school_words_10k_control` | 43 | 0.1365 | 0.4888 | 0.4634 | 75 | complete |
| `school_words_10k_control` | 44 | 0.1378 | 0.4854 | 0.4683 | 80 | complete |

## Aggregate By Variant

| variant | completed seeds | mean CER | std CER | mean WER | mean exact |
|---|---|---:|---:|---:|---:|
| `tri10k_base` | [42, 43, 44] | 0.1528 | 0.0089 | 0.5293 | 0.4232 |
| `line_context_10k` | [42, 43, 44] | 0.1358 | 0.0013 | 0.4898 | 0.4641 |
| `random_crops_10k_control` | [42, 43, 44] | 0.1358 | 0.0017 | 0.4894 | 0.4644 |
| `school_words_10k_control` | [42, 43, 44] | 0.1366 | 0.0012 | 0.4869 | 0.4661 |

## Normalized Line-Context Effect

Mean base CER: 0.1528. Mean line-context CER: 0.1358. Mean delta CER: -0.0170.

| seed | delta CER | 95% CI | School delta CER | School 95% CI |
|---:|---:|---:|---:|---:|
| 42 | -0.0108 | [-0.0147, -0.0067] | -0.0190 | [-0.0267, -0.0114] |
| 43 | -0.0122 | [-0.0163, -0.0081] | -0.0272 | [-0.0351, -0.0195] |
| 44 | -0.0281 | [-0.0323, -0.0239] | -0.0341 | [-0.0425, -0.0260] |

## External TrOCR Baseline

- model: `microsoft/trocr-base-handwritten`
- protocol: pretrained zero-shot generation baseline
- n: 5563
- CER: 1.2985
- WER: 1.4753
- exact: 0.0040

Interpretation: this is an external pretrained zero-shot reference, not a competitive fine-tuned Russian HTR baseline. The result is weak and cannot satisfy a strong-publication baseline requirement by itself.

## Fine-Tuned TrOCR Baseline

- model: `outputs/htr_publication_v3/trocr_finetuned_tri10k_base/best`
- protocol: decoder-only TrOCR adaptation; encoder frozen due 6GB GPU memory limit
- n: 5563
- CER: 1.2657
- WER: 1.0343
- exact: 0.0043

Interpretation: this external baseline is complete but weak. It does not outperform the CRNN controls and should be reported as a negative/limited external-baseline result.

## Full Same-Size Control Status

| control variant | seed | status |
|---|---:|---|
| `random_crops_10k_control` | 42 | complete |
| `random_crops_10k_control` | 43 | complete |
| `random_crops_10k_control` | 44 | complete |
| `school_words_10k_control` | 42 | complete |
| `school_words_10k_control` | 43 | complete |
| `school_words_10k_control` | 44 | complete |

## Validity Addendum

Full addendum report: `outputs/htr_publication_v3/validity_addendum_v1/report.md`.

Claim boundary:
- No exact train-test sample/image/file duplication was detected by automated audits.
- At least one high-risk perceptual near-duplicate candidate was detected; it is too small to explain aggregate metrics, but it should be removed or isolated in a strict publication split.
- Page/source-image overlap exists for at least one variant; page-disjoint stress rows must be cited alongside all-test metrics.
- The controlled result remains nuanced: line-context beats the base model, but same-size controls are comparable; the defensible claim is an augmentation/data-volume effect, not a proven unique line-context mechanism.
- Writer-disjoint validation remains unresolved unless reliable writer_id metadata is added.

Metadata leakage audit, train vs test:

| variant | sample_id overlap | image_path overlap | page overlap | line overlap | text overlap |
|---|---:|---:|---:|---:|---:|
| `tri10k_base` | 0 | 0 | 14 | 95 | 857 |
| `line_context_10k` | 0 | 0 | 14 | 95 | 858 |
| `random_crops_10k_control` | 0 | 0 | 14 | 100 | 959 |
| `school_words_10k_control` | 0 | 0 | 14 | 95 | 989 |

Visual duplicate audit:

| variant | SHA1 overlap | dHash candidates | train paths hashed | test paths hashed |
|---|---:|---:|---:|---:|
| `tri10k_base` | 0 | 1 | 30000 | 5563 |
| `line_context_10k` | 0 | 1 | 39998 | 5563 |
| `random_crops_10k_control` | 0 | 1 | 40000 | 5563 |
| `school_words_10k_control` | 0 | 1 | 40000 | 5563 |

High-risk dHash near-duplicate candidates:

| train sample | test sample | train text | test text | variants |
|---|---|---|---|---|
| `cyr_phrase_054903` | `cyr_phrase_073129` | долларов сша | ) долларов сша | `tri10k_base`, `line_context_10k`, `random_crops_10k_control`, `school_words_10k_control` |

Fixed-penalty dose response:

| run | line train n | CER | WER | exact | delta CER vs base |
|---|---:|---:|---:|---:|---:|
| `baseline_0_lines` | 0 | 0.1454 | 0.5127 | 0.4411 | 0.0000 |
| `plus_2k_lines` | 1998 | 0.1447 | 0.5104 | 0.4436 | -0.0007 |
| `plus_5k_lines` | 4999 | 0.1360 | 0.4902 | 0.4609 | -0.0095 |
| `plus_10k_lines` | 9998 | 0.1352 | 0.4923 | 0.4636 | -0.0103 |

- Best fixed-penalty dose row is plus_10k_lines with CER=0.1352.
- Largest incremental CER decrease is plus_2k_lines -> plus_5k_lines (delta -0.0087).
- The 5k->10k increment is small, consistent with a plateau rather than a linear data-scaling effect.

## Remaining Addendum

Full addendum report: `outputs/htr_publication_v3/remaining_addendum_v1/report.md`.

Strong internal baselines on the same tri10k test:

| baseline | n | CER | WER | exact | checkpoint epoch | status |
|---|---:|---:|---:|---:|---:|---|
| `mixed_cyrillic_natural_full_v1` | 5563 | 0.0822 | 0.3350 | 0.6245 | 46 | complete |
| `mixed_cyrillic_balanced50k_v1` | 5563 | 0.0979 | 0.3853 | 0.5774 | 47 | complete |

Baseline interpretation: The only cached external HuggingFace OCR/HTR model found is TrOCR-base-handwritten. The additional strong baselines are internal CRNN baselines trained on larger in-domain data; they are useful for positioning but are not external SOTA.
Cached HuggingFace models: `['bert-base-uncased', 'microsoft/trocr-base-handwritten']`.

Page-disjoint HKR+School status:
- manifest ready: True
- control manifest ready: True
- 3-seed base-vs-line retrain complete: True
- 3-seed same-size controls complete: True
- full strict page-disjoint package complete: True
- run status: `outputs/htr_publication_v3/page_disjoint_hkr_school_v1/run_status.json`
- full command: `python -u tools/run_page_disjoint_hkr_school_v1.py --seeds 42 43 44 --epochs 80 --num_workers 4`
- control command: `python -u tools/run_page_disjoint_hkr_school_v1.py --variants page_random_crops_8k_control page_school_words_8k_control --seeds 42 43 44 --epochs 80 --num_workers 4`
- control comparison command: `python tools/build_page_disjoint_control_comparisons_v1.py`

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

Annotation reliability:
- report: `outputs/htr_publication_v3/annotation_reliability_addendum_v1/report.md`
- repeated annotation overlap n: 40
- independent package ready: True
- independent browser: `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_browser.html`
- expected filled CSV: `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_second_filled.csv`
- independent minimally complete rows: 0
- formal IAA ready: False
- weak fields: `['audit_usable', 'ink_visible_ok', 'endpoint_error', 'junction_error', 'critical_topology_error', 'graph_quality_0_3']`
- claim boundary: The strict page-disjoint base, line, and same-size controls are complete. A unique natural-line-context claim is allowed only if the paired line-vs-control deltas support it.

## Publication Assessment

Completed now:
- fixed-penalty normalized 3-seed base-vs-line evaluation
- paired CI for normalized base-vs-line comparisons
- completed 3-seed from-scratch same-size random-crop and School-word controls
- external pretrained TrOCR zero-shot baseline on the full test split
- fine-tuned/decode-adapted external TrOCR baseline on the full test split
- automated metadata leakage, visual duplicate, group-stress, domain, error, and fixed dose-response addendum
- completed strict 3-seed HKR+School page-disjoint base-vs-line retraining
- prepared strict page-disjoint same-size control manifests
- completed strict 3-seed HKR+School page-disjoint same-size controls
- annotation repeated-consistency addendum and Wilson intervals for line-quality checks
- blind second-annotation package for formal IAA
- strong data-rich internal CRNN baselines on the same tri10k test

Still missing:
- formal independent inter-annotator agreement
- competitive external Russian/Cyrillic HTR baseline beyond cached TrOCR

Verdict: full same-size v3 controls, validity addenda, and strict 3-seed page-disjoint HKR+School retraining are complete; journal-level readiness is still mainly blocked by formal independent IAA, lack of a competitive external Russian/Cyrillic HTR baseline.
