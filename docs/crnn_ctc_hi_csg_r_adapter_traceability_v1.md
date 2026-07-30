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
| 2 | 20-feature x-aligned builder | `src/htr/xaligned_hi_csg_r.py`, builder CLI | Complete; all main and additional manifests built |
| 3 | Feature audit and 30-sample browser | audit and visualization CLIs | Complete; full audit PASS and 30 examples inspected |
| 4 | Dataset and output-length collate | `src/htr/dataset_adapter.py` | Complete |
| 5 | Adapter, gate, auxiliary CTC, strict loader | `src/htr/model_hi_csg_r_adapter.py` | Complete |
| 6 | Unit/integration tests | three required test modules | Complete |
| 7 | Trainer/evaluator and metadata | training/evaluation CLIs and runtime module | Complete |
| 8 | One-sample and 64-128 sample overfit | notebook smoke section and trainer support | Complete; preregistered smoke gate PASS |
| 9 | Seed-42 development comparison | automated preliminary and full validation gates | Complete; preliminary gate STOP |
| 10 | Final three seeds | frozen configs and guarded notebook stage | Not run; blocked by seed-42 STOP |
| 11 | One-time final evaluation | freeze registry and double-guarded notebook stage | Not run; blocked by seed-42 STOP |
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
- Builder `1.0.1` deterministically orders node-free isolated-loop pixels before
  segment aggregation. This is a recorded pre-training bug fix to the `1.0.0`
  implementation; it does not change the frozen feature list or model protocol.

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
- Smoke subsets may reuse the full-train normalizer only when the canonical
  normalizer train manifest is passed explicitly and every subset row exactly
  matches a row in that SHA-verified parent manifest.
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
- All 21 main/additional feature-build summaries: PASS with builder `1.0.1`.
- Full main-split feature audit: PASS over 51,561 records with no record failures.
- Node, endpoint, and junction count deltas are exactly zero; effective edge-length
  deltas are below `3e-11`, and float32 record reconstruction deltas below `2.3e-5`.
- The 30-example visual browser contains 10 Cyrillic, 10 HKR, and 10 School samples
  stratified across clean/medium/hard; manual alignment acceptance: PASS.
- Real image feature record: shape `[185, 20]`, no NaN/Inf.
- Real-record quality mapping and local/global count consistency audit: PASS.
- Train-only normalizer serialization, provenance, and default policies: PASS.
- Strict canonical loader: PASS.
- Real-checkpoint initial equivalence: exact, maximum absolute delta `0.0`.
- Base parameters: 3,900,892; adapter/gate/auxiliary/fusion parameters: 119,429
  (3.062% increase).
- M0-FT training/evaluation smoke: PASS.
- M3 warm-up/joint training/evaluation smoke: PASS.
- Unit/integration tests (`15 passed`) and scoped Ruff checks: PASS.
- Canonical M0 and M3 evaluator paths, including graph-quality strata: PASS.
- Historical prediction compatibility and paired-bootstrap self-check on 5,563 samples: PASS.
- The 37-cell staged experiment notebook executes completely in `check` mode:
  all 21 code cells run, input audit and tests pass, every inactive block emits a
  visible `SKIP`, and the final artifact dashboard is rendered.
- The same 37-cell notebook executes completely in `prepare` mode: all 21 code
  cells run, all feature builds/audits/visualizations/tests pass, 55 rich display
  outputs are rendered, and no error output is present.
- The notebook executes completely in `smoke` mode on CUDA with 21/21 code cells
  and no error outputs. The one-sample CER reaches `0.0`; auxiliary-only loss on
  128 samples decreases from `7.94` to `2.85`; the full-128 CER decreases from
  `0.288` to `0.070`; all six machine-checked smoke conditions pass.
- Seed-42 M0-FT and M3 runs completed all 25 joint epochs (plus five M3 warm-up
  epochs). Validation CER is `0.079537` for M0-FT and `0.082196` for M3, a
  `-3.342%` relative improvement (that is, a degradation).
- M3 degrades CER in all three core validation domains by `0.002104` to
  `0.003033`. Correct graphs are slightly better than shuffled graphs
  (`0.082196` versus `0.083004`), and gate/gradient diagnostics pass, but the
  frozen primary and domain gates fail.
- The seed-42 notebook now records this expected protocol `STOP` without an error
  output, reuses the verified completed training runs on repetition, and blocks
  M2, seeds 43/44, and test. H4 is classified as exploratory.
- All CLI entry points import and expose help successfully.

## Stopped Per Frozen Protocol

The following cannot be marked complete by code inspection:

The frozen seed-42 continuation gate failed. Consequently, the following stages are
intentionally not executed rather than pending:

1. M2 seed-42;
2. M0-FT/M3 seeds 43/44;
3. the one-time mixed/domain/page-disjoint/robustness test;
4. final paired test confidence intervals.

The negative validation result and exploratory H4 classification are recorded in
`outputs/htr_adapter_v1/statistical_analysis/validation_gate/validation_gate.md`.

These stages are encoded in
`notebooks/htr_adapter_v1_full_experiment.ipynb`. Until they are executed in order,
the correct status is **implementation complete, experiments pending**, not full
Definition of Done.
