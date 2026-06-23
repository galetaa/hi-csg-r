# Structural Gold Annotation Schema

This annotation is about structural representation quality, not whether HTR prediction is correct.

## Fields

`structural_usable`, `foreground_ok`, `skeleton_ok`, `graph_ok`:
- `1` yes
- `0` no

Severity fields:
- `0` none
- `1` minor
- `2` severe/dominant

Severity fields are:
- `line_residual`
- `neighbor_noise`
- `missed_ink`
- `false_ink`
- `false_branches`
- `broken_strokes`
- `overconnected`
- `segmentation_issue`

`htr_error_explained_by_structure`:
- `yes`
- `partial`
- `no`
- `not_applicable`

Use `not_applicable` when HTR prediction is correct.

## Criteria

`structural_usable = 1` if the sample can be used for structural representation analysis:
- foreground generally follows ink
- skeleton generally follows strokes
- no dominant noise, ruling, or neighboring text

`foreground_ok = 1` if:
- main ink is preserved
- no strong false foreground
- no critical line residual

`skeleton_ok = 1` if:
- skeleton follows main strokes
- strokes do not break massively
- dominant branches are not caused by noise or ruling

`graph_ok = 1` if:
- topology visually reflects main strokes
- no severe overbranching or overconnection

## HTR Error Explained By Structure

Only evaluate this for samples where HTR prediction is wrong.

`yes`:
The error is clearly linked to missed ink, false ink, broken strokes, line residual, or neighbor noise.

`partial`:
A structural defect exists, but the error may also be visual, linguistic, or ambiguous.

`no`:
Structure looks good; the error is likely model-side, linguistic, or ambiguous.

`not_applicable`:
HTR prediction is correct.

## Gold Subset Acceptance

Target acceptance after annotating 200 samples:
- `structural_usable >= 85%`
- `foreground_ok >= 85%`
- `skeleton_ok >= 80%`
- `graph_ok >= 75%`
