# Experiment registry

| component                   | final status                    | retained artifact                           |
| --------------------------- | ------------------------------- | ------------------------------------------- |
| image-only recognizer       | primary absolute baseline       | image-only clean and robustness outputs     |
| graph-vector v2             | retained graph-aware checkpoint | `tri10k_graph_fusion_v2_lowcap_all/best.pt` |
| gated graph model           | analyzed, not primary           | robustness v1 outputs                       |
| graph-v3 retrain            | rejected checkpoint             | retained as controlled negative result      |
| old School preprocessing    | rejected                        | historical audit only                       |
| border suppression v1       | rejected                        | diagnostic experiment only                  |
| global threshold 145        | candidate only                  | superseded by auto rule                     |
| `school_dark_auto`          | accepted                        | final School foreground method              |
| H1 descriptive robustness   | completed                       | robustness mode comparison                  |
| H1 paired statistics        | completed                       | paired corpus v3                            |
| H2 diagnostic audit         | completed                       | manual audit summary                        |
| H2 random validation        | completed                       | random-100 summary                          |
| H3 diagnostics              | completed                       | after-school-fg-v3 summary                  |
| new HTR architecture search | frozen                          | no further experiments                      |
| graph-fusion CER tuning     | frozen                          | no further experiments                      |

## Final accepted configuration

```text
recognition baseline:
    image-only model

retained graph-aware model:
    graph-vector v2

School Notebooks graph preprocessing:
    school_dark_auto

primary H1 inference:
    paired corpus relative-degradation advantage

primary H2 evidence:
    diagnostic HKR/Cyrillic audit
    + independent School random-100 validation

primary H3 evidence:
    structural-core subgroup high-error detection
```
