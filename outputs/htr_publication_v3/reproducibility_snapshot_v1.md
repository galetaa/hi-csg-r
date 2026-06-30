# Publication Reproducibility Snapshot v1

## Repository

- commit: `35fcb36b55d719b362a91c80cdd8efb96253d99d`
- dirty working tree: yes
- branch: `main`

## Environment

- python: `3.11.12 (main, Apr 18 2025, 15:15:27) [GCC 12.2.0]`
- executable: `/home/galetka/.pyenv/versions/3.11.12/bin/python`
- platform: `Linux-6.12.94+deb13-amd64-x86_64-with-glibc2.41`
- torch: `2.9.0+cu128`
- torch CUDA available: `False`
- torch CUDA version: `12.8`
- CUDA devices: `[]`
- transformers: `4.57.1`
- numpy: `2.2.6`

## Git Status

```text
M tools/evaluate_crnn_ctc.py
 M tools/train_crnn_ctc.py
?? chats/
?? docs/publication_hardening_plan_v1.md
?? outputs/htr_publication_v2/
?? outputs/htr_publication_v3/
?? tools/audit_htr_manifest_integrity_v1.py
?? tools/build_annotation_reliability_addendum_v1.py
?? tools/build_full_same_size_control_comparisons_v1.py
?? tools/build_publication_v2_report.py
?? tools/build_publication_v3_remaining_addendum_v1.py
?? tools/build_publication_v3_status_report.py
?? tools/build_publication_v3_validity_addendum_v1.py
?? tools/build_same_size_aug_controls_v1.py
?? tools/build_structural_gold_strict_addendum_v1.py
?? tools/create_page_disjoint_hkr_school_manifests_v1.py
?? tools/evaluate_trocr_baseline_v1.py
?? tools/run_full_same_size_controls_v1.py
?? tools/run_page_disjoint_hkr_school_v1.py
?? tools/run_publication_v3_dose_fixed_eval_v1.py
?? tools/run_publication_v3_long_pipeline.py
?? tools/train_trocr_baseline_v1.py
?? tools/write_publication_repro_snapshot_v1.py
```

## Reproducibility Caution

A dirty working tree means the snapshot is not a clean immutable release state. For publication, commit or archive the exact code and generated manifests used for the reported runs.
