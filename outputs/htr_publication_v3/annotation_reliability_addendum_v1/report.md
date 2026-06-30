# Annotation Reliability Addendum v1

This is repeated-annotation consistency on overlapping samples. It is useful reliability evidence, but it is not a formal inter-annotator agreement unless the two files are confirmed to be independent annotators.

## Repeated Annotation Consistency

- overlap n: 40
- pilot file: `outputs/h2_gold_audit_v1/annotations/annotation_pilot_40_filled.csv`
- comparison file: `outputs/h2_gold_audit_v1/annotations/annotation_100_filled.csv`

| field | n | agreement | Wilson 95% CI | Cohen kappa | weighted kappa |
|---|---:|---:|---:|---:|---:|
| `audit_usable` | 40 | 0.900 | [0.769, 0.960] | 0.286 | n/a |
| `ink_visible_ok` | 40 | 0.875 | [0.739, 0.945] | -0.064 | n/a |
| `skeleton_follows_ink` | 40 | 0.925 | [0.801, 0.974] | 0.625 | n/a |
| `missed_visible_stroke` | 35 | 1.000 | [0.901, 1.000] | 1.000 | n/a |
| `spurious_stroke` | 34 | 0.971 | [0.851, 0.995] | 0.653 | n/a |
| `endpoint_error` | 37 | 0.946 | [0.823, 0.985] | 0.471 | n/a |
| `junction_error` | 37 | 0.811 | [0.658, 0.905] | 0.000 | n/a |
| `loop_error` | 35 | 1.000 | [0.901, 1.000] | 1.000 | n/a |
| `critical_topology_error` | 40 | 0.900 | [0.769, 0.960] | 0.444 | n/a |
| `graph_quality_0_3` | 40 | 0.800 | [0.652, 0.895] | 0.350 | 0.612 |

## Independent Blind Second Annotation Package

The blind second-annotation package is prepared. It becomes formal IAA evidence only after a genuinely independent annotator fills the expected CSV and the scorer reports adequate overlap.

- package ready: True
- browser: `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_browser.html`
- template CSV: `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_second_template.csv`
- expected filled CSV: `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_second_filled.csv`
- score report: `outputs/htr_publication_v3/independent_annotation_v1/scored/report.md`
- second CSV exists: False
- minimally complete rows: 0
- formal IAA ready: False

## Line Quality Audit Rates

| audit | field | direction | count/n | rate | Wilson 95% CI |
|---|---|---|---:|---:|---:|
| `full_natural_lines_150` | `valid_line` | positive | 145/150 | 0.967 | [0.924, 0.986] |
| `full_natural_lines_150` | `correct_order` | positive | 149/150 | 0.993 | [0.963, 0.999] |
| `full_natural_lines_150` | `complete_enough` | positive | 125/150 | 0.833 | [0.766, 0.884] |
| `full_natural_lines_150` | `good_for_line_train` | positive | 149/150 | 0.993 | [0.963, 0.999] |
| `full_natural_lines_150` | `neighbor_noise` | issue | 8/150 | 0.053 | [0.027, 0.102] |
| `rendered_line_sanity_80` | `readable` | positive | 80/80 | 1.000 | [0.954, 1.000] |
| `rendered_line_sanity_80` | `correct_crop` | positive | 51/80 | 0.637 | [0.528, 0.734] |
| `rendered_line_sanity_80` | `good_for_htr` | positive | 80/80 | 1.000 | [0.954, 1.000] |
| `natural_lines_120` | `valid_line` | positive | 118/120 | 0.983 | [0.941, 0.995] |
| `natural_lines_120` | `correct_order` | positive | 120/120 | 1.000 | [0.969, 1.000] |
| `natural_lines_120` | `good_for_train_aug` | positive | 120/120 | 1.000 | [0.969, 1.000] |
| `natural_lines_120` | `missing_words` | issue | 74/120 | 0.617 | [0.527, 0.699] |
| `natural_lines_120` | `neighbor_noise` | issue | 0/120 | 0.000 | [-0.000, 0.031] |
| `lineaware_quality_gate_120` | `usable` | positive | 115/120 | 0.958 | [0.906, 0.982] |
| `lineaware_quality_gate_120` | `skeleton_follows_ink` | positive | 120/120 | 1.000 | [0.969, 1.000] |
| `lineaware_quality_gate_120` | `ink_loss` | issue | 6/120 | 0.050 | [0.023, 0.105] |
| `lineaware_quality_gate_120` | `line_residual` | issue | 9/120 | 0.075 | [0.040, 0.136] |

## Publication Interpretation

- Strongest supported claim: The existing filled audits provide quantitative quality-control rates and repeated-annotation consistency for a 40-sample overlap. A blind second-annotation package is prepared but is not itself agreement evidence.
- Not supported: A formal inter-annotator agreement claim is not supported until a genuinely independent second annotator fills the blind package and the scoring report shows adequate agreement.
- Remaining requirement: Have a second independent annotator fill the blind package and rerun `python tools/score_independent_annotation_v1.py`.

Fields with kappa below 0.6:
- `audit_usable`: kappa=0.286, agreement=0.900
- `ink_visible_ok`: kappa=-0.064, agreement=0.875
- `endpoint_error`: kappa=0.471, agreement=0.946
- `junction_error`: kappa=0.000, agreement=0.811
- `critical_topology_error`: kappa=0.444, agreement=0.900
- `graph_quality_0_3`: kappa=0.350, agreement=0.800
