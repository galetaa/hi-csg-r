# Final Page-Disjoint HKR+School Control Report

Date: 2026-07-03  
Protocol: strict page-disjoint HKR+School split, fixed decoding penalty `-0.4`, 3 seeds (`42`, `43`, `44`)  
Scope: final aggregation after completing `page_random_crops_8k_control` and `page_school_words_8k_control`.

## 1. What Was Completed

All strict page-disjoint variants are complete:

| Variant | Seeds | Train status | Eval status |
|---|---|---|---|
| `page_base` | 42, 43, 44 | complete | complete |
| `page_line_10k` | 42, 43, 44 | complete | complete |
| `page_random_crops_8k_control` | 42, 43, 44 | complete | complete |
| `page_school_words_8k_control` | 42, 43, 44 | complete | complete |

All variants use HKR+School only. The Cyrillic Handwriting dataset is excluded from this strict split because it lacks page/source/writer metadata needed for page-disjoint evaluation.

Source artifacts:

- run status: `outputs/htr_publication_v3/page_disjoint_hkr_school_v1/run_status.json`
- eval summaries: `outputs/htr_publication_v3/page_disjoint_hkr_school_v1/eval_fixed_m04/*/summary.json`
- paired comparisons: `outputs/htr_publication_v3/page_disjoint_hkr_school_v1/paired_fixed_m04/`
- updated addendum: `outputs/htr_publication_v3/remaining_addendum_v1/report.md`
- updated global status report: `outputs/htr_publication_v3/publication_v3_status_report.md`

## 2. Aggregate Results

| Variant | Completed seeds | Mean CER | Std CER | Mean WER | Mean Exact |
|---|---|---:|---:|---:|---:|
| `page_base` | 42, 43, 44 | 0.1483 | 0.0119 | 0.4764 | 0.3946 |
| `page_line_10k` | 42, 43, 44 | 0.1271 | 0.0057 | 0.4227 | 0.4522 |
| `page_random_crops_8k_control` | 42, 43, 44 | 0.1317 | 0.0058 | 0.4362 | 0.4363 |
| `page_school_words_8k_control` | 42, 43, 44 | 0.1210 | 0.0007 | 0.4110 | 0.4604 |

Mean deltas:

| Comparison | Delta CER | Delta WER | Delta Exact | Interpretation |
|---|---:|---:|---:|---|
| `page_line_10k - page_base` | -0.0212 | -0.0537 | +0.0576 | Line augmentation is clearly better than base. |
| `page_line_10k - page_random_crops_8k_control` | -0.0047 | -0.0135 | +0.0159 | Line is slightly better than random-crops control on mean metrics. |
| `page_line_10k - page_school_words_8k_control` | +0.0061 | +0.0117 | -0.0083 | Line is worse than school-words control on mean metrics. |

## 3. Per-Seed Fixed-Penalty Evaluation

| Variant | Seed | n | CER | WER | Exact | Checkpoint epoch |
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

## 4. Paired Line-vs-Base Comparisons

Negative delta means the line variant has lower error than base.

| Seed | n | Delta CER | 95% CI | School Delta CER | School 95% CI | Delta WER | Delta Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4000 | -0.0099 | [-0.0147, -0.0051] | -0.0180 | [-0.0263, -0.0098] | -0.0253 | +0.0238 |
| 43 | 4000 | -0.0403 | [-0.0450, -0.0357] | -0.0396 | [-0.0482, -0.0316] | -0.0990 | +0.1068 |
| 44 | 4000 | -0.0135 | [-0.0181, -0.0089] | -0.0177 | [-0.0255, -0.0097] | -0.0369 | +0.0423 |

Conclusion: `page_line_10k` is consistently better than `page_base` on all 3 seeds. The paired confidence intervals exclude zero for overall CER and School CER.

## 5. Paired Line-vs-Control Comparisons

Negative delta means the line variant has lower error than the control.

| Comparison | Seed | n | Delta CER | 95% CI | School Delta CER | School 95% CI | Delta WER | Delta Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `line_vs_random_crops_control` | 42 | 4000 | +0.0077 | [+0.0031, +0.0126] | +0.0066 | [-0.0017, +0.0149] | +0.0150 | -0.0137 |
| `line_vs_random_crops_control` | 43 | 4000 | -0.0115 | [-0.0159, -0.0070] | -0.0196 | [-0.0273, -0.0119] | -0.0280 | +0.0315 |
| `line_vs_random_crops_control` | 44 | 4000 | -0.0102 | [-0.0149, -0.0054] | -0.0168 | [-0.0253, -0.0088] | -0.0274 | +0.0300 |
| `line_vs_school_words_control` | 42 | 4000 | +0.0119 | [+0.0073, +0.0168] | +0.0168 | [+0.0084, +0.0252] | +0.0238 | -0.0212 |
| `line_vs_school_words_control` | 43 | 4000 | +0.0001 | [-0.0045, +0.0046] | +0.0011 | [-0.0073, +0.0094] | +0.0005 | +0.0013 |
| `line_vs_school_words_control` | 44 | 4000 | +0.0063 | [+0.0019, +0.0106] | +0.0081 | [+0.0004, +0.0158] | +0.0107 | -0.0048 |

Interpretation:

- Against `page_random_crops_8k_control`, `page_line_10k` is mixed by seed: worse on seed 42, better on seeds 43 and 44.
- Against `page_school_words_8k_control`, `page_line_10k` is not better. It is worse on seeds 42 and 44 and statistically indistinguishable on seed 43.
- Therefore, the evidence does not support a unique natural-line-context mechanism.

## 6. Updated Scientific Claim

Supported claim:

> On a strict HKR+School page-disjoint protocol, adding relevant extra training data substantially improves the base CRNN/CTC HTR model. The improvement is reproducible across 3 seeds for the line-augmentation variant relative to the base model.

Partially supported claim:

> Natural-line augmentation is a useful augmentation strategy, but its advantage over same-size alternatives is not robust.

Not supported claim:

> Natural-line context uniquely explains the observed HTR improvement.

Reason: the same-size `page_school_words_8k_control` has the best mean CER, WER, and Exact among the page-disjoint variants:

- CER: `0.1210` vs `0.1271` for line;
- WER: `0.4110` vs `0.4227` for line;
- Exact: `0.4604` vs `0.4522` for line.

## 7. Publication-Level Interpretation

The page-disjoint controls strengthen the work because they prevent an overclaim. The final interpretation should be:

1. The base-vs-line result is real under page-disjoint HKR+School evaluation.
2. The effect is best described as a data-centric/domain-relevant augmentation effect.
3. The line-context explanation alone is too strong.
4. The school-words control suggests that domain-matched additional word samples can be at least as useful as natural-line samples.
5. The remaining publication blockers are not these page-disjoint controls anymore; they are mainly independent annotation agreement and a competitive external Russian/Cyrillic HTR baseline.

