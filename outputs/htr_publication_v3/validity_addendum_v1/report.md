# Publication V3 Validity Addendum v1

This addendum strengthens the v3 package with automated leakage checks, page/line/word stress slices, domain breakdowns, error decomposition, and fixed-penalty dose-response evidence.

## Claim Boundary

- No exact train-test sample/image/file duplication was detected by automated audits.
- At least one high-risk perceptual near-duplicate candidate was detected; it is too small to explain aggregate metrics, but it should be removed or isolated in a strict publication split.
- Page/source-image overlap exists for at least one variant; page-disjoint stress rows must be cited alongside all-test metrics.
- The controlled result remains nuanced: line-context beats the base model, but same-size controls are comparable; the defensible claim is an augmentation/data-volume effect, not a proven unique line-context mechanism.
- Writer-disjoint validation remains unresolved unless reliable writer_id metadata is added.

## Metadata Leakage Audit

Writer metadata limitation: Writer-disjoint validation is not supported by current metadata: only 0/196250 rows (0.00%) have writer_id.

| variant | sample_id overlap | image_path overlap | page overlap | line overlap | text overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tri10k_base` | 0 | 0 | 14 | 95 | 857 |
| `line_context_10k` | 0 | 0 | 14 | 95 | 858 |
| `random_crops_10k_control` | 0 | 0 | 14 | 100 | 959 |
| `school_words_10k_control` | 0 | 0 | 14 | 95 | 989 |

Interpretation: exact `sample_id`/`image_path` overlap is direct leakage. `page_key`/`line_key` overlap is a dependence risk. Text overlap is expected in HTR and is not visual leakage by itself.

## Visual Duplicate Audit

| variant | train paths | test paths | SHA1 overlaps | dHash candidate overlaps | missing train/test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tri10k_base` | 30000 | 5563 | 0 | 1 | 0/0 |
| `line_context_10k` | 39998 | 5563 | 0 | 1 | 0/0 |
| `random_crops_10k_control` | 40000 | 5563 | 0 | 1 | 0/0 |
| `school_words_10k_control` | 40000 | 5563 | 0 | 1 | 0/0 |

dHash candidate assessment: 1 high-risk near-duplicate pair(s) among 1 unique candidate pair(s).

| risk | train sample | test sample | train text | test text | seen in variants |
| --- | --- | --- | --- | --- | --- |
| `high_near_duplicate_risk` | `cyr_phrase_054903` | `cyr_phrase_073129` | долларов сша | ) долларов сша | `tri10k_base`, `line_context_10k`, `random_crops_10k_control`, `school_words_10k_control` |

## Group Stress Evaluation

Stress subsets are computed from existing fixed-penalty test predictions. They are not a retrained group-disjoint experiment; they diagnose whether reported test performance is concentrated on samples whose page/line/word metadata is already represented in training.

| variant | subset | n | mean CER | std CER | mean WER | mean exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tri10k_base` | `all_test` | 5563-5563 | 0.1528 | 0.0089 | 0.5293 | 0.4232 |
| `tri10k_base` | `all_test_minus_high_risk_visual_near_duplicates` | 5562-5562 | 0.1528 | 0.0089 | 0.5293 | 0.4233 |
| `tri10k_base` | `page_disjoint_from_train` | 2000-2000 | 0.1641 | 0.0064 | 0.5132 | 0.4948 |
| `tri10k_base` | `page_seen_in_train` | 2000-2000 | 0.1012 | 0.0088 | 0.4217 | 0.4762 |
| `tri10k_base` | `line_disjoint_from_train` | 2979-2979 | 0.1428 | 0.0080 | 0.4856 | 0.4927 |
| `tri10k_base` | `line_seen_in_train` | 948-948 | 0.0845 | 0.0033 | 0.3935 | 0.4764 |
| `tri10k_base` | `school_page_disjoint_from_train` | 2000-2000 | 0.1641 | 0.0064 | 0.5132 | 0.4948 |
| `tri10k_base` | `school_page_seen_in_train` | 0-0 | n/a | n/a | n/a | n/a |
| `line_context_10k` | `all_test` | 5563-5563 | 0.1358 | 0.0013 | 0.4898 | 0.4641 |
| `line_context_10k` | `all_test_minus_high_risk_visual_near_duplicates` | 5562-5562 | 0.1358 | 0.0013 | 0.4898 | 0.4642 |
| `line_context_10k` | `page_disjoint_from_train` | 2000-2000 | 0.1374 | 0.0015 | 0.4570 | 0.5517 |
| `line_context_10k` | `page_seen_in_train` | 2000-2000 | 0.0915 | 0.0040 | 0.3848 | 0.5162 |
| `line_context_10k` | `line_disjoint_from_train` | 2979-2979 | 0.1211 | 0.0036 | 0.4299 | 0.5490 |
| `line_context_10k` | `line_seen_in_train` | 948-948 | 0.0773 | 0.0011 | 0.3712 | 0.5035 |
| `line_context_10k` | `school_page_disjoint_from_train` | 2000-2000 | 0.1374 | 0.0015 | 0.4570 | 0.5517 |
| `line_context_10k` | `school_page_seen_in_train` | 0-0 | n/a | n/a | n/a | n/a |
| `random_crops_10k_control` | `all_test` | 5563-5563 | 0.1358 | 0.0017 | 0.4894 | 0.4644 |
| `random_crops_10k_control` | `all_test_minus_high_risk_visual_near_duplicates` | 5562-5562 | 0.1358 | 0.0017 | 0.4894 | 0.4645 |
| `random_crops_10k_control` | `page_disjoint_from_train` | 2000-2000 | 0.1448 | 0.0027 | 0.4772 | 0.5293 |
| `random_crops_10k_control` | `page_seen_in_train` | 2000-2000 | 0.0882 | 0.0012 | 0.3703 | 0.5317 |
| `random_crops_10k_control` | `line_disjoint_from_train` | 2899-2899 | 0.1267 | 0.0018 | 0.4464 | 0.5346 |
| `random_crops_10k_control` | `line_seen_in_train` | 1028-1028 | 0.0721 | 0.0005 | 0.3405 | 0.5350 |
| `random_crops_10k_control` | `school_page_disjoint_from_train` | 2000-2000 | 0.1448 | 0.0027 | 0.4772 | 0.5293 |
| `random_crops_10k_control` | `school_page_seen_in_train` | 0-0 | n/a | n/a | n/a | n/a |
| `school_words_10k_control` | `all_test` | 5563-5563 | 0.1366 | 0.0012 | 0.4869 | 0.4661 |
| `school_words_10k_control` | `all_test_minus_high_risk_visual_near_duplicates` | 5562-5562 | 0.1365 | 0.0012 | 0.4869 | 0.4661 |
| `school_words_10k_control` | `page_disjoint_from_train` | 2000-2000 | 0.1371 | 0.0004 | 0.4484 | 0.5538 |
| `school_words_10k_control` | `page_seen_in_train` | 2000-2000 | 0.0933 | 0.0011 | 0.3840 | 0.5190 |
| `school_words_10k_control` | `line_disjoint_from_train` | 2979-2979 | 0.1222 | 0.0013 | 0.4301 | 0.5450 |
| `school_words_10k_control` | `line_seen_in_train` | 948-948 | 0.0781 | 0.0054 | 0.3542 | 0.5239 |
| `school_words_10k_control` | `school_page_disjoint_from_train` | 2000-2000 | 0.1371 | 0.0004 | 0.4484 | 0.5538 |
| `school_words_10k_control` | `school_page_seen_in_train` | 0-0 | n/a | n/a | n/a | n/a |

## Domain Breakdown

Mean CER by source dataset, averaged across seeds.

| variant | dataset | n | mean CER | std CER | mean WER | mean exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tri10k_base` | `cyrillic_handwriting` | 1563-1563 | 0.2043 | 0.0127 | 0.6873 | 0.2638 |
| `tri10k_base` | `hkr_words` | 2000-2000 | 0.1012 | 0.0088 | 0.4217 | 0.4762 |
| `tri10k_base` | `school_notebooks_clean` | 2000-2000 | 0.1641 | 0.0064 | 0.5132 | 0.4948 |
| `line_context_10k` | `cyrillic_handwriting` | 1563-1563 | 0.1903 | 0.0035 | 0.6660 | 0.2853 |
| `line_context_10k` | `hkr_words` | 2000-2000 | 0.0915 | 0.0040 | 0.3848 | 0.5162 |
| `line_context_10k` | `school_notebooks_clean` | 2000-2000 | 0.1374 | 0.0015 | 0.4570 | 0.5517 |
| `random_crops_10k_control` | `cyrillic_handwriting` | 1563-1563 | 0.1854 | 0.0040 | 0.6575 | 0.2952 |
| `random_crops_10k_control` | `hkr_words` | 2000-2000 | 0.0882 | 0.0012 | 0.3703 | 0.5317 |
| `random_crops_10k_control` | `school_notebooks_clean` | 2000-2000 | 0.1448 | 0.0027 | 0.4772 | 0.5293 |
| `school_words_10k_control` | `cyrillic_handwriting` | 1563-1563 | 0.1913 | 0.0035 | 0.6678 | 0.2860 |
| `school_words_10k_control` | `hkr_words` | 2000-2000 | 0.0933 | 0.0011 | 0.3840 | 0.5190 |
| `school_words_10k_control` | `school_notebooks_clean` | 2000-2000 | 0.1371 | 0.0004 | 0.4484 | 0.5538 |

Line-context CER deltas by dataset. Negative means line-context is better.

| comparison | dataset | delta CER |
| --- | --- | ---: |
| `line_vs_base` | `cyrillic_handwriting` | -0.0140 |
| `line_vs_base` | `hkr_words` | -0.0096 |
| `line_vs_base` | `school_notebooks_clean` | -0.0268 |
| `line_vs_random` | `cyrillic_handwriting` | 0.0050 |
| `line_vs_random` | `hkr_words` | 0.0034 |
| `line_vs_random` | `school_notebooks_clean` | -0.0074 |
| `line_vs_school_words` | `cyrillic_handwriting` | -0.0009 |
| `line_vs_school_words` | `hkr_words` | -0.0017 |
| `line_vs_school_words` | `school_notebooks_clean` | 0.0003 |

## Error Decomposition

| variant | substitution rate | deletion rate | insertion rate |
| --- | ---: | ---: | ---: |
| `tri10k_base` | 0.1047 | 0.0284 | 0.0128 |
| `line_context_10k` | 0.0950 | 0.0228 | 0.0116 |
| `random_crops_10k_control` | 0.0942 | 0.0244 | 0.0112 |
| `school_words_10k_control` | 0.0958 | 0.0237 | 0.0125 |

## Fixed-Penalty Dose Response

fixed test-time blank_logit_penalty=-0.4, seed-42 historical checkpoints

| run | line train n | CER | WER | exact | delta CER vs base | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_0_lines` | 0 | 0.1454 | 0.5127 | 0.4411 | 0.0000 | complete |
| `plus_2k_lines` | 1998 | 0.1447 | 0.5104 | 0.4436 | -0.0007 | complete |
| `plus_5k_lines` | 4999 | 0.1360 | 0.4902 | 0.4609 | -0.0095 | complete |
| `plus_10k_lines` | 9998 | 0.1352 | 0.4923 | 0.4636 | -0.0103 | complete |

- Best fixed-penalty dose row is plus_10k_lines with CER=0.1352.
- Largest incremental CER decrease is plus_2k_lines -> plus_5k_lines (delta -0.0087).
- The 5k->10k increment is small, consistent with a plateau rather than a linear data-scaling effect.

## Remaining Publication Risks

- retrain/evaluate a true page-disjoint or writer-disjoint split if metadata and compute allow
- add annotation reliability evidence for the school-line corpus
- add a competitive external Russian/Cyrillic HTR baseline beyond decoder-only TrOCR adaptation
