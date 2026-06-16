# Reproducibility

## 1. Canonical evidence inputs

| evidence | path |
|---|---|
| final H1 statistics | `outputs/final_evidence_v1/h1_final_statistical_report_v2.json` |
| H2 manual audit | `outputs/h2_gold_audit_v1/h2_manual_audit_summary_v2.json` |
| random-100 validation | `outputs/h2_gold_audit_v1/school_foreground_v3_random/random_100_summary.json` |
| controlled School foreground conclusion | `outputs/final_evidence_v1/school_fg_v3_controlled_conclusion.json` |
| graph-feature cross-evaluation | `outputs/htr_graph_v1/cross_eval_v3/cross_eval_report.json` |
| H3 final diagnostic summary | `outputs/h3_graph_quality_v1/after_school_fg_v3_auto/h3_after_school_foreground_v3_summary.json` |

## 2. Final graph-ready manifest

```text
data/experiments/htr_graph_v1/graph_ready/
tri10k_mixed_school_fg_v3_auto/
````

Expected files:

```text
train.jsonl
val.jsonl
test.jsonl
vocab.json
summary.json
```

## 3. Final retained graph checkpoint

```text
outputs/htr_graph_v1/
tri10k_graph_fusion_v2_lowcap_all/
best.pt
```

The graph-v3 retrained checkpoint is retained only as a negative
controlled experiment.

## 4. Final preprocessing configuration

```text
School Notebooks:
    method: school_dark_auto
    primary threshold: 145
    fallback threshold: 120
    fallback trigger: foreground fraction > 0.35
    minimum connected-object size: 4
```

## 5. Robustness configuration

```text
source samples: 5563
conditions per sample: 15
total joined distorted records: 83445
bootstrap iterations: 5000
permutations: 20000
cluster unit: clean source sample
seed: 20260616
```

## 6. H1 source outputs

```text
outputs/robustness_v2_recomputed/
    graph_vector_v2/
    paired_analysis/
    paired_corpus_v3/
    robustness_graph_modes_v2.json
    robustness_graph_modes_v2.md
```

## 7. H2 source outputs

```text
outputs/h2_gold_audit_v1/
    h2_manual_audit_summary_v2.json
    school_foreground_v3/
    school_foreground_v3_random/
```

## 8. H3 source outputs

```text
outputs/h3_graph_quality_v1/
after_school_fg_v3_auto/
```

## 9. Required environment

The project requires:

* Python;
* PyTorch;
* NumPy;
* Pillow;
* OpenCV;
* scikit-image;
* SciPy;
* scikit-learn.

Exact installed versions should be archived from the active environment:

```bash
python --version
python -m pip freeze > outputs/final_documentation_v3/pip_freeze.txt
```

## 10. Repository state

Before archiving the final experiment:

```bash
git status --short
git rev-parse HEAD
git log -1 --oneline
```

Store the output in:

```text
outputs/final_documentation_v3/repository_state.txt
```

## 11. Determinism notes

Deterministic or fixed-seed components include:

* dataset manifest generation;
* robustness corruption seed;
* random-100 sampling seed;
* bootstrap seed;
* permutation-test seed.

Neural-network training may still vary because of device-specific and
library-level nondeterminism.
