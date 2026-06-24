# Result Card: Graph Fusion Iteration 2 Context-10k Pilot

## Setup

- Control: image-only +10k contextual School line augmentation.
- Candidate: image + 39 lineaware graph features with invalid graph rows masked.
- Contextual line train samples use `graph_valid=false`; word-level samples use graph features.
- Test split is unchanged word-level tri10k mixed.

## Main Table

| model | overall CER | overall WER | exact | HKR CER | Cyrillic CER | School CER | School WER | School exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| image-only +10k | 0.1351 | 0.4924 | 0.4636 | 0.0917 | 0.1858 | 0.1389 | 0.4718 | 0.5390 |
| graph-fusion | 0.1338 | 0.4786 | 0.4562 | 0.0920 | 0.1997 | 0.1253 | 0.4355 | 0.5730 |
| zero-graph | 0.1536 | 0.5366 | 0.4152 | 0.1066 | 0.2316 | 0.1391 | 0.4692 | 0.5425 |

## Key Paired CI vs Image-only +10k

| scope | mean per-sample ΔCER | 95% CI | aggregate ΔCER |
|---|---:|---:|---:|
| `overall` | 0.0036 | [-0.0003, 0.0076] | 0.0066 |
| `hkr_words` | 0.0123 | [0.0070, 0.0176] | 0.0081 |
| `cyrillic_handwriting` | 0.0101 | [0.0023, 0.0181] | 0.0162 |
| `school_notebooks_clean` | -0.0100 | [-0.0172, -0.0027] | -0.0080 |

## Preliminary Conclusion

The graph branch is not ignored: zero-graph inference degrades strongly. The final interpretation depends on the paired CIs: the pilot should be treated as targeted if School improves but Cyrillic degrades.
