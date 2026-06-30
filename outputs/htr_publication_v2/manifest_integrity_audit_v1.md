# HTR Manifest Integrity Audit v1

This audit checks manifest-level reproducibility risks for the baseline, natural-line context augmentation, and same-size image-only controls.

| manifest | split sizes | train duplicate ids | split id overlap | OOV rows | empty text rows | train composition |
|---|---|---:|---:|---:|---:|---|
| `tri10k_base` | train=30000, val=6000, test=5563 | 0 | 0 | 0 | 0 | cyrillic_handwriting:10000, hkr_words:10000, school_notebooks_clean:10000 |
| `line_context_10k` | train=39998, val=6000, test=5563 | 0 | 0 | 0 | 0 | cyrillic_handwriting:10000, hkr_words:10000, school_notebooks_clean:10000, school_notebooks_line:9998 |
| `random_crops_10k_control` | train=40000, val=6000, test=5563 | 0 | 0 | 0 | 0 | cyrillic_handwriting:13333, hkr_words:13333, school_notebooks_clean:13334 |
| `school_words_10k_control` | train=40000, val=6000, test=5563 | 0 | 0 | 0 | 0 | cyrillic_handwriting:10000, hkr_words:10000, school_notebooks_clean:20000 |

## Pass/Fail Summary

| manifest | no train duplicates | no split overlap | no OOV | no empty text |
|---|---:|---:|---:|---:|
| `tri10k_base` | yes | yes | yes | yes |
| `line_context_10k` | yes | yes | yes | yes |
| `random_crops_10k_control` | yes | yes | yes | yes |
| `school_words_10k_control` | yes | yes | yes | yes |

## Interpretation

- A clean audit reduces the risk that the diagnostic control results are caused by sample-id leakage or vocabulary mismatches.
- This is a manifest-level audit only; it does not prove that near-duplicate handwriting images or writer identities are disjoint.
