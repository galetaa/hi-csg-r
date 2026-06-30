# Independent Annotation v1 Protocol

Purpose: formal second-pass annotation for inter-annotator agreement on structural HTR preprocessing quality.

Samples: 100 rows from `outputs/h2_gold_audit_v1/annotations/annotation_100_filled.csv`, shuffled with a fixed seed.

Blinding:
- Previous labels are not shown.
- CER, model prediction, target text, structural risk score, and audit-cell stratum are not shown.
- The annotator sees only sample identity, dataset/level/category, image path, and four visual views.

Fields:
- `audit_usable`: 1 if the sample can be audited, 0 otherwise.
- `exclusion_reason`: `ok`, `illegible_or_bad_crop`, `background_dominates`, `target_ambiguous`, `too_short`, or `non_text_fragment`.
- `ink_visible_ok`: 1 if main ink is visible/preserved.
- `skeleton_follows_ink`: 1 if skeleton follows the main ink strokes.
- `missed_visible_stroke`, `spurious_stroke`, `endpoint_error`, `junction_error`, `loop_error`: binary error flags.
- `critical_topology_error`: 1 if the structural representation has a major topology defect.
- `graph_quality_0_3`: 0 unusable, 1 weak, 2 usable, 3 good.

Scoring:
- Fill/export the browser CSV as `outputs/htr_publication_v3/independent_annotation_v1/blind_annotation_second_filled.csv`.
- Run `python tools/score_independent_annotation_v1.py`.
- The scorer writes agreement, Cohen kappa, and weighted kappa for `graph_quality_0_3`.

Publication boundary:
- This supports formal IAA only if the second annotator is genuinely independent from the first annotation pass.
- AI-generated or same-person repeated annotation must be reported as repeated consistency, not independent IAA.
