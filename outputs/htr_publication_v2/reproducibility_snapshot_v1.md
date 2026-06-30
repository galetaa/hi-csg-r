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
- torch CUDA available: `True`
- torch CUDA version: `12.8`
- CUDA devices: `['NVIDIA GeForce RTX 3060 Laptop GPU']`
- transformers: `4.57.1`
- numpy: `2.2.6`

## Git Status

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

## Reproducibility Caution

A dirty working tree means the snapshot is not a clean immutable release state. For publication, commit or archive the exact code and generated manifests used for the reported runs.
