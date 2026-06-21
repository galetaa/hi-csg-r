# Final experiment status — v2

## Completed

- Image-only baseline evaluation.
- Graph-vector and gated-fusion evaluation.
- Synthetic robustness evaluation across 15 conditions.
- End-to-end recomputation of graph features under distortion.
- Paired cluster bootstrap and permutation analysis for H1.
- H2 manual diagnostic audit.
- School Notebooks foreground failure diagnosis.
- Deterministic `school_dark_auto` preprocessing repair.
- Independent random-100 foreground validation.
- Graph-feature cross-evaluation with old and new checkpoints.
- H3 graph-derived high-error analysis.
- Consolidated evidence and manuscript text generation.

## Experimental freeze

- Do not add new HTR architectures.
- Do not perform further graph-fusion CER tuning.
- Do not retrain graph-v3 again.
- Do not add new synthetic corruption families for this study.
- Do not reinterpret relative robustness as absolute superiority.

## Remaining work

- Integrate Results, Discussion, and Limitations into the manuscript.
- Prepare final figures from existing experiment outputs.
- Write Methods with exact data splits and preprocessing definitions.
- Verify all manuscript numbers against generated JSON evidence.
- Freeze tables and archive reproducibility commands.
