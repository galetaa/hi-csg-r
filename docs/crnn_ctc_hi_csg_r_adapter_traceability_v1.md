# CRNN-CTC HI-CSG-R Adapter v1: Traceability Audit

Audit date: 2026-07-30.

This document separates implementation readiness from experimental completion.
The implementation is ready; the full scientific Definition of Done is not complete
until the preregistered experiments are actually run.

## Work Packages

| WP | Requirement | Implementation evidence | Current status |
|---:|---|---|---|
| 0 | Frozen protocol and seed configs | `docs/crnn_ctc_hi_csg_r_adapter_protocol_v1.md`, seven YAML configs | Complete |
| 1 | Input/checkpoint audit | `tools/audit_adapter_inputs_v1.py`; canonical report under `outputs/htr_adapter_v1/input_audit/` | Complete |
| 2 | 20-feature x-aligned builder | `src/htr/xaligned_hi_csg_r.py`, builder CLI | Complete; full corpus build pending |
| 3 | Feature audit and 30-sample browser | audit and visualization CLIs | Code complete; full audit and manual acceptance pending |
| 4 | Dataset and output-length collate | `src/htr/dataset_adapter.py` | Complete |
| 5 | Adapter, gate, auxiliary CTC, strict loader | `src/htr/model_hi_csg_r_adapter.py` | Complete |
| 6 | Unit/integration tests | three required test modules | Complete |
| 7 | Trainer/evaluator and metadata | training/evaluation CLIs and runtime module | Complete |
| 8 | One-sample and 64-128 sample overfit | notebook smoke section and trainer support | One-sample technical smoke passed; preregistered 128-sample gates pending |
| 9 | Seed-42 development comparison | automated preliminary and full validation gates | Code complete; runs pending |
| 10 | Final three seeds | frozen configs and guarded notebook stage | Pending validation gate |
| 11 | One-time final evaluation | freeze registry and double-guarded notebook stage | Pending final checkpoints |
| 12 | Paired statistics and report | bootstrap, Holm comparison, report tools | Code complete; final inputs pending |

## Specification Coverage

### Data and alignment

- Nodes use `min(floor(x_norm * T), T - 1)`.
- Original edge polylines are split into adjacent point segments.
- Segment midpoint selects the bin and segment length supplies the weight.
- The only feature smoothing kernel is `[0.25, 0.50, 0.25]`.
- Actual image width determines `T = max(width // 4, 1)`.
- Real empty bins and batch padding have distinct masks.
- Robustness feature builds consume the distorted image path in each manifest.

### Model

- Fusion is after the existing visual projection and before the existing BiLSTM.
- Dimensions are `[B,T,20] -> [B,T,256]`, gate `[B,T,1]`.
- Joint fusion is `LayerNorm(V + aG)`.
- A zero-residual bypass exists only to guarantee exact initial equivalence.
- Final graph projection weight and bias are zero initialized.
- Gate final bias is `-1.5`.
- Added parameter count is 119,429, below the 400,000 limit.

### Controls

- `M0-FT` and `M3` use matched width-bucket batches and shared-layer policy.
- `M2` zeros normalized features 11-20 while retaining identical parameters.
- Shuffle maps are within-domain, width/ink matched, contain no self-pairs, and are
  persisted as JSON.
- Canonical M0, M0-FT, M2, M3, M3-shuffle, and historical M1 can all enter the final
  comparison notebook.

### Leakage controls

- Normalizer provenance is tied to the train-manifest SHA256.
- Evaluation rejects a blank penalty different from checkpoint metadata.
- Validation gate is machine checked.
- The notebook requires a successful validation gate and freeze registry before test.
- Final test execution is explicitly one-shot in the registry.

### Metrics and artifacts

- Evaluator writes overall, domain, length, token-type, and graph-quality strata.
- Per-sample rows contain edit counts for paired analysis.
- Bootstrap reports CER CI, wins/losses/ties, domain and length deltas, delta WER, and
  delta exact match.
- Comparison applies Holm correction across declared comparisons.
- Trainer writes best/last checkpoints, config, history, summaries, predictions,
  stdout/stderr logs, and hardware/runtime metadata.

## Verified Locally

- Canonical input audit for seeds 42/43/44: PASS.
- Canonical split counts: 39,998 train, 6,000 validation, and 5,563 test.
- Real image feature record: shape `[185, 20]`, no NaN/Inf.
- Real-record quality mapping and local/global count consistency audit: PASS.
- Train-only normalizer serialization, provenance, and default policies: PASS.
- Strict canonical loader: PASS.
- Real-checkpoint initial equivalence: exact, maximum absolute delta `0.0`.
- Base parameters: 3,900,892; adapter/gate/auxiliary/fusion parameters: 119,429
  (3.062% increase).
- M0-FT training/evaluation smoke: PASS.
- M3 warm-up/joint training/evaluation smoke: PASS.
- Unit/integration tests (`13 passed`) and scoped Ruff checks: PASS.
- Canonical M0 and M3 evaluator paths, including graph-quality strata: PASS.
- Historical prediction compatibility and paired-bootstrap self-check on 5,563 samples: PASS.
- The 36-cell experiment notebook executes completely with all heavy and final-test
  switches disabled; its embedded input audit and tests pass.
- All CLI entry points import and expose help successfully.

## Not Yet Scientifically Complete

The following cannot be marked complete by code inspection:

1. full feature construction for every main and additional manifest;
2. automatic full-corpus audit and manual inspection of 30 examples;
3. preregistered one-sample and 128-sample overfit runs;
4. M0-FT/M3/M2 seed-42 development runs and validation gate;
5. seeds 43/44 after a positive gate;
6. the one-time mixed/domain/page-disjoint/robustness test;
7. final paired confidence intervals, tables, figures, report, and H4 classification.

These stages are encoded in
`notebooks/htr_adapter_v1_full_experiment.ipynb`. Until they are executed in order,
the correct status is **implementation complete, experiments pending**, not full
Definition of Done.
