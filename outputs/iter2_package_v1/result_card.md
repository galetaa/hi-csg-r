# Iteration 2 package: data-centric HTR improvement with structural diagnostics

## 1. Summary

Iteration 2 shifted the project from graph-only HTR experiments to a data-centric and structurally controlled HTR pipeline.

Main result:
- corrected School foreground extraction;
- built natural-line contextual augmentation;
- obtained statistically supported image-only HTR gains;
- validated confidence-aware selective prediction;
- confirmed that the current structural extraction is not the main bottleneck on the diagnostic gold subset;
- tested a single graph-fusion pilot and found targeted School benefit but mixed-dataset instability.

## 2. Accepted preprocessing

School foreground method:
- rectangular raw COCO crop;
- whitebalance + line-aware foreground;
- post-binarization polygon filtering;
- method: `rectangular_whitebalance_lineaware_postpoly_v3`.

Quality gate:
- usable: 95.8%
- skeleton_follows_ink: 100%
- neighbor_text_removed: 100%
- ink_loss: 5.0%
- line_residual: 7.5%

## 3. School quality manifests

School lineaware_v3 quality buckets:

train:
- clean_core: 8783 / 10000
- hard_real: 1217 / 10000
- invalid_or_review: 0

val:
- clean_core: 1796 / 2000
- hard_real: 204 / 2000
- invalid_or_review: 0

test:
- clean_core: 1764 / 2000
- hard_real: 236 / 2000
- invalid_or_review: 0

## 4. Full natural-line corpus

Full School COCO line candidates:
- line groups: 60074
- covered word instances: 262301 / 283670
- coverage: 92.47%
- mean words/group: 4.37
- groups with 4+ words: 38125
- geometry outliers excluded: 101

Full line corpus v1:
- usable line groups: 59973

## 5. Line augmentation

Image-only word-level test results:

| model | train_n | line_n | overall CER | HKR CER | Cyrillic CER | School CER | School WER | School exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 30000 | 0 | 0.1453 | 0.0956 | 0.1932 | 0.1575 | 0.5018 | 0.5060 |
| +2k context | 31998 | 1998 | 0.1448 | 0.0974 | 0.1934 | 0.1541 | 0.4983 | 0.5120 |
| +5k context | 34999 | 4999 | 0.1360 | 0.0910 | 0.1879 | 0.1403 | 0.4650 | 0.5425 |
| +10k context | 39998 | 9998 | 0.1351 | 0.0917 | 0.1858 | 0.1389 | 0.4718 | 0.5390 |

Paired bootstrap:
- +5k overall ΔCER: -0.0093, CI [-0.0132, -0.0056]
- +5k School ΔCER: -0.0171, CI [-0.0244, -0.0099]
- +10k overall ΔCER: -0.0102, CI [-0.0141, -0.0061]
- +10k School ΔCER: -0.0186, CI [-0.0262, -0.0112]

Canonical checkpoints:
- CER canonical: +10k context
- balanced canonical: +5k context

## 6. Selective prediction

Canonical selective model:
- HTR model: +10k context
- risk model: confidence_graph

Risk quality:
- feature_only: AUC around 0.60
- model_confidence: AUC around 0.80
- confidence_graph: AUC around 0.81 overall and 0.83 on School

School operating points for +10k confidence_graph:
- strict: coverage 0.3905, CER 0.0491, exact 0.8143
- balanced: coverage 0.6615, CER 0.0685, exact 0.7249
- broad: coverage 0.7795, CER 0.0845, exact 0.6652
- full: coverage 1.0000, CER 0.1389, exact 0.5390

Limitation:
- global thresholds are not coverage-fair across datasets/token types;
- numeric/mixed and HKR/Cyrillic are rejected more often;
- short 1–3 samples are accepted more readily, so short-token errors are overconfident ambiguity errors.

## 7. Structural gold diagnostic

Gold subset:
- total: 200
- structural_usable: 100%
- foreground_ok: 100%
- skeleton_ok: 100%
- graph_ok: 100%

All annotated HTR errors were marked as not explained by structural extraction defects.

Conclusion:
- foreground/skeleton/graph extraction is usable on the diagnostic subset;
- remaining errors are primarily model/ambiguity/token-level rather than extraction failures.

## 8. Graph-fusion pilot

The graph-fusion pilot produced a mixed result. Compared with the image-only +10k model, graph-fusion significantly improved School CER, with the strongest gain on hard_real samples, but significantly degraded HKR and Cyrillic CER. Overall CER was statistically neutral/slightly negative. Zero-graph ablation substantially reduced performance, indicating that the graph branch was actively used.

Compared to image-only +10k seed42:
- overall ΔCER: 0.0036, CI [-0.0003, 0.0076]
- School ΔCER: -0.0100, CI [-0.0172, -0.0027]
- HKR ΔCER: 0.0123, CI [0.0070, 0.0176]
- Cyrillic ΔCER: 0.0101, CI [0.0023, 0.0181]

School hard_real:
- CER 0.2057 -> 0.1794
- exact 0.4025 -> 0.4407

Zero-graph ablation:
- normal graph-fusion CER: 0.1338
- zero-graph CER: 0.1536

Interpretation:
- graph features contain recognition-relevant structural signal for School;
- naive ungated late fusion is not safe as a universal mixed-dataset recognizer;
- a future controlled variant would be dataset-gated graph fusion, but it is not part of Iteration 2.

## 9. Main conclusion

Iteration 2 demonstrates a data-centric HTR improvement:
- corrected foreground extraction;
- natural-line context augmentation gives statistically supported CER gains;
- confidence-aware selective prediction works strongly;
- structural graph features are useful for diagnostics and confidence calibration;
- graph fusion provides School-specific gains, especially on hard_real, but naive global fusion harms non-School datasets.

## 10. Main limitations

- training comparison is still mostly single-seed;
- contextual line crops are not clean isolated line crops;
- structural gold is a diagnostic usability check, not a pixel-level topology benchmark;
- naive global graph fusion hurts non-School datasets and should not be treated as the universal canonical recognizer;
- global selective thresholds are not group-fair;
