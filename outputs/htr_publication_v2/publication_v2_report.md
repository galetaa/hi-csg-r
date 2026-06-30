# Publication Evidence Package v2

## Executive Status

This package raises the research level beyond the previous technical-evidence package by adding same-size augmentation controls, paired confidence intervals, a strict structural-gold addendum, and manifest integrity audits.

The work is now stronger for a thesis defense or a focused conference/workshop submission. It is still not ready for a strong journal or broad SOTA claim because the same-size controls are diagnostic fine-tunes rather than full from-scratch 3-seed runs, and no external transformer/SOTA HTR baseline is included.

## Primary Result Retained From v1

### Table 2. Primary HTR 3-seed result

Aggregate: mean ΔCER=-0.016952, std ΔCER=0.010246, mean relative ΔCER=-10.89%, improved CER seeds=3/3.

| seed | baseline_CER | plus10k_CER | delta_CER | relative_delta_CER | baseline_WER | plus10k_WER | delta_WER | baseline_exact | plus10k_exact | delta_exact |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.145446 | 0.135127 | -0.010319 | -7.09% | 0.512838 | 0.492426 | -0.020412 | 0.440949 | 0.463599 | 0.022650 |
| 43 | 0.148931 | 0.137146 | -0.011785 | -7.91% | 0.515196 | 0.493280 | -0.021916 | 0.435736 | 0.458745 | 0.023009 |
| 44 | 0.162917 | 0.134165 | -0.028752 | -17.65% | 0.556900 | 0.481263 | -0.075637 | 0.396009 | 0.472766 | 0.076757 |
| mean |  |  | -0.016952 | -10.89% |  |  | -0.039321 |  |  | 0.040805 |

### Table 3. Domain-wise HTR result

Natural-line context augmentation gives a seed-stable overall gain, with the strongest and most stable effect on School Notebooks.

| domain | mean_baseline_CER | mean_plus10k_CER | mean_delta_CER | relative_delta_CER | improved_seeds | interpretation |
|---|---|---|---|---|---|---|
| cyrillic_handwriting | 0.203650 | 0.189479 | -0.014170 | -6.75% | 3/3 | improves in all available seeds |
| hkr_words | 0.101083 | 0.091986 | -0.009097 | -8.30% | 2/3 | mostly improves, but not fully seed-stable |
| school_notebooks_clean | 0.163752 | 0.136771 | -0.026981 | -16.38% | 3/3 | improves in all available seeds |

Interpretation: the main result remains the 3-seed natural-line context augmentation effect. The new v2 controls do not replace this primary result; they test whether the effect can plausibly be explained by adding the same amount of ordinary crop data.

## Diagnostic Same-Size Control Protocol

All diagnostic runs resume from `outputs/htr_graph_v1/tri10k_image_only_v1/last.pt`, continue for epochs 81-83, use seed 42, blank logit penalty -0.4, batch size 16, and `max_train_batches=500`. This is an auxiliary causal diagnostic, not a full independent training protocol.

| variant | train n | val CER at checkpoint | test CER | test WER | exact | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `base continuation` | 30000 | 0.1078 | 0.1476 | 0.5214 | 0.4341 | same checkpoint, continued on original tri10k base manifest |
| `natural-line context +10k` | 39998 | 0.1057 | 0.1437 | 0.5176 | 0.4418 | same-size target method: +9998 rendered natural-line context crops |
| `random crop control +10k` | 40000 | 0.1083 | 0.1487 | 0.5188 | 0.4343 | same-size image-only control: balanced ordinary crop samples |
| `School word crop control +10k` | 40000 | 0.1142 | 0.1554 | 0.5304 | 0.4246 | same-size image-only control: extra School word crops without line context |

## Paired Diagnostic Comparisons

Negative delta means the first model named in the comparison has lower CER than the second model named.

| comparison | delta definition | n | delta CER | 95% CI | School delta CER | School 95% CI | delta WER | delta exact | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `line context vs base continuation` | line - base | 5563 | -0.0039 | [-0.0071, -0.0006] | -0.0137 | [-0.0201, -0.0072] | -0.0038 | 0.0077 | CER improvement; CI excludes zero |
| `line context vs random crop control` | line - random | 5563 | -0.0050 | [-0.0084, -0.0016] | -0.0162 | [-0.0226, -0.0095] | -0.0012 | 0.0075 | CER improvement; CI excludes zero |
| `line context vs School word control` | line - School words | 5563 | -0.0117 | [-0.0151, -0.0083] | -0.0145 | [-0.0215, -0.0077] | -0.0128 | 0.0173 | CER improvement; CI excludes zero |
| `random crop control vs base continuation` | random - base | 5563 | 0.0011 | [-0.0023, 0.0045] | 0.0024 | [-0.0043, 0.0092] | -0.0026 | 0.0002 | directionally worse; CI overlaps zero |
| `School word control vs base continuation` | School words - base | 5563 | 0.0078 | [0.0041, 0.0114] | 0.0008 | [-0.0064, 0.0077] | 0.0090 | -0.0095 | CER degradation; CI excludes zero |

## Diagnostic Interpretation

- The line-context model is better than base continuation in the diagnostic protocol: delta CER -0.0039, CI [-0.0071, -0.0006].
- The line-context model is better than the random-crop same-size control: delta CER -0.0050, CI [-0.0084, -0.0016].
- The effect is mainly School-driven: School delta CER is -0.0137 vs base and -0.0162 vs random control.
- The random-crop control is neutral versus base: delta CER +0.0011, CI crosses zero.
- The School-word control is worse overall versus base: delta CER +0.0078, CI excludes zero, while its School-only effect is near neutral.
- This supports the claim that natural-line context provides information not reproduced by adding the same number of ordinary crop samples.
- The claim is still limited: line context slightly worsens Cyrillic in the diagnostic line-vs-base comparison, and the controls were not run as full 3-seed from-scratch experiments.

## Manifest Integrity Audit

### HTR Manifest Integrity Audit v1

This audit checks manifest-level reproducibility risks for the baseline, natural-line context augmentation, and same-size image-only controls.

| manifest | split sizes | train duplicate ids | split id overlap | OOV rows | empty text rows | train composition |
|---|---|---:|---:|---:|---:|---|
| `tri10k_base` | train=30000, val=6000, test=5563 | 0 | 0 | 0 | 0 | cyrillic_handwriting:10000, hkr_words:10000, school_notebooks_clean:10000 |
| `line_context_10k` | train=39998, val=6000, test=5563 | 0 | 0 | 0 | 0 | cyrillic_handwriting:10000, hkr_words:10000, school_notebooks_clean:10000, school_notebooks_line:9998 |
| `random_crops_10k_control` | train=40000, val=6000, test=5563 | 0 | 0 | 0 | 0 | cyrillic_handwriting:13333, hkr_words:13333, school_notebooks_clean:13334 |
| `school_words_10k_control` | train=40000, val=6000, test=5563 | 0 | 0 | 0 | 0 | cyrillic_handwriting:10000, hkr_words:10000, school_notebooks_clean:20000 |

#### Pass/Fail Summary

| manifest | no train duplicates | no split overlap | no OOV | no empty text |
|---|---:|---:|---:|---:|
| `tri10k_base` | yes | yes | yes | yes |
| `line_context_10k` | yes | yes | yes | yes |
| `random_crops_10k_control` | yes | yes | yes | yes |
| `school_words_10k_control` | yes | yes | yes | yes |

#### Interpretation

- A clean audit reduces the risk that the diagnostic control results are caused by sample-id leakage or vocabulary mismatches.
- This is a manifest-level audit only; it does not prove that near-duplicate handwriting images or writer identities are disjoint.

Interpretation: the audit reduces the risk of simple sample-id leakage, train duplicate inflation, OOV mismatch, or empty-text artifacts in the new controls. It does not rule out writer/page dependence or visual near-duplicates.

## Structural Gold Strict Addendum

Structural subset n=200; datasets={'school_notebooks_clean': 156, 'cyrillic_handwriting': 18, 'hkr_words': 26}; strata={'clean_core_correct': 40, 'hard_real_correct': 40, 'hard_real_error': 40, 'high_confidence_error': 30, 'rejected_correct_low_confidence': 30, 'numeric_mixed_rare_format': 20}.

| diagnostic field | count | n | rate | Wilson 95% low |
|---|---:|---:|---:|---:|
| `structural_usable` | 200 | 200 | 1.0000 | 0.9812 |
| `foreground_ok` | 200 | 200 | 1.0000 | 0.9812 |
| `skeleton_ok` | 200 | 200 | 1.0000 | 0.9812 |
| `graph_ok` | 200 | 200 | 1.0000 | 0.9812 |

| issue | minor+ rate | severe/dominant rate |
|---|---:|---:|
| `line_residual` | 22.0% | 11.5% |
| `missed_ink` | 8.5% | 0.0% |

Strict interpretation: this supports diagnostic usability of the generated structures on the sampled cases. It does not prove exact graph topology, pen trajectory, writing order, endpoint correctness, or stroke-level ground truth. The absence of inter-annotator agreement remains a major publication limitation.

## Stronger Baseline Status

### Mixed Cyrillic image-only baselines report — Stage 3.3

#### 1. Purpose

This report compares single-dataset Cyrillic HTR baselines with mixed-dataset image-only baselines. The goal is to determine whether a universal Cyrillic CRNN-CTC model improves cross-domain recognition before graph-aware experiments.

#### 2. Mixed runs

| run | training composition | selected penalty |
|---|---|---:|
| Mixed Cyrillic balanced50k v1 | Balanced training: 50k samples from each Cyrillic dataset. | -0.2 |
| Mixed Cyrillic natural-full v1 | Natural full training: all available train samples from each Cyrillic dataset. | -0.4 |

#### 3. Test CER comparison

| dataset | single full CER | mixed balanced50k CER | mixed natural-full CER | natural-full vs single |
|---|---:|---:|---:|---:|
| Cyrillic Handwriting | 0.1405 | 0.1281 | 0.1208 | 14.0% |
| HKR Words | 0.1525 | 0.0691 | 0.0623 | 59.2% |
| School Notebooks Clean | 0.0838 | 0.1002 | 0.0744 | 11.1% |

#### 4. Full per-dataset metrics for mixed natural-full

| dataset | split | n | CER | WER | exact | pred_len | blank | penalty | epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cyrillic Handwriting | val | 7232 | 0.0723 | 0.2939 | 0.6962 | 7.42 | 0.835 | -0.4 | 46 |
| Cyrillic Handwriting | test | 1563 | 0.1208 | 0.4936 | 0.4562 | 9.40 | 0.842 | -0.4 | 46 |
| HKR Words | val | 6494 | 0.0565 | 0.2555 | 0.6729 | 10.25 | 0.852 | -0.4 | 46 |
| HKR Words | test | 6495 | 0.0623 | 0.2867 | 0.6411 | 10.43 | 0.860 | -0.4 | 46 |
| School Notebooks Clean | val | 24214 | 0.0432 | 0.1793 | 0.8252 | 5.93 | 0.840 | -0.4 | 46 |
| School Notebooks Clean | test | 24623 | 0.0744 | 0.2742 | 0.7327 | 5.85 | 0.824 | -0.4 | 46 |

Interpretation: the project has a stronger internal CRNN-BiLSTM image-only baseline at larger Cyrillic data scale. This helps demonstrate engineering competence and data-scale behavior, but it is not the same experimental setting as tri10k plus line-context augmentation. No TrOCR/ViT/SOTA baseline was completed in this package.

## Reproducibility Snapshot

### Publication Reproducibility Snapshot v1

#### Repository

- commit: `35fcb36b55d719b362a91c80cdd8efb96253d99d`
- dirty working tree: yes
- branch: `main`

#### Environment

- python: `3.11.12 (main, Apr 18 2025, 15:15:27) [GCC 12.2.0]`
- executable: `/home/galetka/.pyenv/versions/3.11.12/bin/python`
- platform: `Linux-6.12.94+deb13-amd64-x86_64-with-glibc2.41`
- torch: `2.9.0+cu128`
- torch CUDA available: `True`
- torch CUDA version: `12.8`
- CUDA devices: `['NVIDIA GeForce RTX 3060 Laptop GPU']`
- transformers: `4.57.1`
- numpy: `2.2.6`

#### Git Status

```text
M tools/train_crnn_ctc.py
?? chats/
?? outputs/htr_publication_v2/
?? tools/audit_htr_manifest_integrity_v1.py
?? tools/build_publication_v2_report.py
?? tools/build_same_size_aug_controls_v1.py
?? tools/build_structural_gold_strict_addendum_v1.py
?? tools/write_publication_repro_snapshot_v1.py
```

#### Reproducibility Caution

A dirty working tree means the snapshot is not a clean immutable release state. For publication, commit or archive the exact code and generated manifests used for the reported runs.

Interpretation: the environment and repository state are now archived in the publication package. The snapshot still records a dirty working tree, so a clean commit/archive is required before submission.

## Updated Claim Matrix

| claim | status after v2 | allowed wording |
|---|---|---|
| Natural-line context improves HTR across 3 seeds | supported by primary v1 3-seed result | Allowed as the main recognition claim. |
| The gain is not merely from adding +10k ordinary crops | supported diagnostically, not fully proven | Allowed only as diagnostic evidence; full same-size 3-seed controls are still required for a strong paper. |
| Benefit is strongest on School Notebooks | supported by v1 domain table and v2 paired controls | Allowed. |
| HI-CSG-R structures are usable for diagnostics | partially supported by structural gold addendum | Allowed with explicit diagnostic-only limitation. |
| Graph topology/trajectory is recovered | not supported | Forbidden. |
| System is SOTA | not supported | Forbidden. |

## Remaining Publication Gaps

1. Run full from-scratch same-size controls over at least three seeds: base tri10k, line-context +10k, random-crop +10k, School-word +10k.
2. Add an external strong HTR baseline, preferably a transformer HTR model or a well-cited Russian/Cyrillic HTR baseline, under the same train/test protocol.
3. Add writer/page-level or near-duplicate leakage audits if metadata permits.
4. Add inter-annotator agreement and stricter pixel/topology metrics for the structural component.
5. Freeze a clean repository state with exact environment, commands, checkpoints, and dataset build scripts.

## Bottom Line

The v2 additions materially improve scientific defensibility. The work can now be argued as an empirical study of natural-line context augmentation for Russian offline HTR with diagnostic structural evidence. It is still not publication-complete for a high-standard venue because the strongest alternative explanations are reduced but not eliminated by full independent controls.
