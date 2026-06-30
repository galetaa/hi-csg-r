# Publication V3 Remaining Addendum v1

## Strong Internal Baselines

The only cached external HuggingFace OCR/HTR model found is TrOCR-base-handwritten. The additional strong baselines are internal CRNN baselines trained on larger in-domain data; they are useful for positioning but are not external SOTA.

| baseline | n | CER | WER | exact | checkpoint epoch | status |
|---|---:|---:|---:|---:|---:|---|
| `mixed_cyrillic_natural_full_v1` | 5563 | 0.0822 | 0.3350 | 0.6245 | 46 | complete |
| `mixed_cyrillic_balanced50k_v1` | 5563 | 0.0979 | 0.3853 | 0.5774 | 47 | complete |

Cached HuggingFace models: `['bert-base-uncased', 'microsoft/trocr-base-handwritten']`.

## Page-Disjoint HKR+School Split

- base manifest root: `data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1`
- line manifest root: `data/experiments/htr_publication_v3/page_disjoint_hkr_school_plus_lines_10k_v1`
- base train/val/test n: 20000/4000/4000
- line train n: 28014 (line samples selected: 8014)
- train-vs-test page overlap: []
- cyrillic limitation: cyrillic_handwriting is excluded because it has no page_id/source_image_file/writer_id metadata.
- full retrain command: `python -u tools/run_page_disjoint_hkr_school_v1.py --seeds 42 43 44 --epochs 80 --num_workers 4`

| variant | seed | last epoch | best exists | eval returncode | status |
|---|---:|---:|---|---:|---|
| `page_base` | 42 | 80 | True | 0 | complete |
| `page_base` | 43 | None | None | None | running |

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
- Added: annotation repeated-consistency and line-quality Wilson intervals
- Added: blind second-annotation package for formal IAA
- Added: strong data-rich internal CRNN baselines on the same tri10k test
- Still not fully solved: formal independent inter-annotator agreement
- Still not fully solved: competitive external Russian/Cyrillic HTR baseline beyond cached TrOCR
- Still not fully solved: completed 3-seed page-disjoint from-scratch retraining
- Claim boundary: The new page-disjoint manifests make the required strict retraining feasible and reproducible. Until the long retrain finishes, they should be reported as prepared/queued rather than final result evidence.
