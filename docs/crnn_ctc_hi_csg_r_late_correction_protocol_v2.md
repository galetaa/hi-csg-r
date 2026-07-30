# CRNN-CTC HI-CSG-R Late Correction Protocol v2

**Version:** 2.0  
**Frozen:** 2026-07-30  
**Previous protocol:** `crnn_ctc_hi_csg_r_adapter_protocol_v1`  
**Feature source:** `hi_csg_r_xaligned_v1`  
**Split seed:** `20260730`

## 1. Status of v1

The v1 result remains a completed negative experiment:

```text
M0-FT seed42 validation micro-CER = 0.079537
M3 correct validation micro-CER = 0.082196
M3 shuffled validation micro-CER = 0.083004
M3 relative degradation = 3.342%
correct-vs-shuffled advantage = 0.000808 CER
```

V1 artifacts, stopping decision, and conclusion must not be overwritten.
Blocked v1 runs cannot be restarted under the v1 protocol name.

## 2. Research question

Test whether a bounded HI-CSG-R correction of frozen baseline CTC logits can
reduce CER when intervention is restricted to non-empty graph bins and weighted
by visual uncertainty.

Positive evidence requires:

1. improvement over the matched frozen image-only baseline;
2. correct graph better than matched shuffle;
3. an independent holdout gate;
4. reproducibility on final seeds 42, 43, and 44;
5. one-shot final test evaluation;
6. no material degradation of core domains;
7. exactly zero correction in empty bins.

## 3. Architectural boundary

The image recognizer is unchanged and frozen:

```text
image -> CNN -> projection -> BiLSTM -> classifier -> Z_base
```

The only added path is:

```text
x-aligned HI-CSG-R [T,20]
-> post-normalization empty mask
-> masked pooling at kernels 1/5/9
-> graph adapter [T,128]
-> uncertainty/risk-aware bounded gate
-> delta CTC logits
Z_final = Z_base + alpha * gate * delta
```

CNN, projection, BiLSTM, and baseline classifier remain `requires_grad=False`.
The complete v2 trainable module must contain fewer than 400,000 parameters,
preferably fewer than 250,000.

## 4. Independent development data

The canonical enhanced train set of 39,998 samples is split group-wise:

| Split | Samples | Per-domain target |
|---|---:|---:|
| adapter_v2_train | 35,498 | remainder |
| adapter_v2_dev | 3,000 | 1,000 |
| adapter_v2_holdout | 1,500 | 500 |

Grouping priority:

```text
writer_id
-> source_metadata.page_id / page_id
-> source_group / line_group_id
-> source image
-> normalized image path
-> sample_id
```

Sample ID, image path, exact SHA1, and group overlap must all equal zero.
The normalizer and risk quantiles are fitted only on adapter_v2_train.

A fresh `B0-dev-v2` image-only CRNN-CTC is trained only on adapter_v2_train
and selected only on adapter_v2_dev. Holdout is not evaluated during training.

## 5. Preflight D1-D3

Before v2 training, run inference-only diagnostics on v1:

- D1 blank penalties: `-0.8,-0.6,-0.5,-0.4,-0.3,-0.2,0.0`;
- D2 graph scales: `0,0.10,0.25,0.50,0.75,1.0`;
- D3 strict non-empty graph mask, correct and shuffled.

Preflight freezes the development blank penalty and `alpha_max`.
Default `alpha_max` is 0.25.

Full V2-1/V2-2 development is allowed when correct graph is better than
shuffle, the branch is active, and D2 or D3 improves M3 by at least 0.0005
absolute CER. If the last condition fails, only V2-1 is allowed as a final
architectural check.

## 6. Input masks

The batch contains:

- `time_mask`: real CRNN timestep;
- `nonempty_graph_mask`: local graph/ink is present;
- `padding_mask`: batch-only padding.

Normalization order is mandatory:

```python
x_norm = normalizer.transform(x_raw)
x_norm = x_norm * nonempty_graph_mask[..., None]
```

Post-normalization graph input, graph embedding, delta logits, gate, and final
correction must be exactly zero for every empty real bin and padded timestep.

## 7. Features and risk

All 20 v1 features are retained. `ambiguous_edge_fraction` is diagnostic-only
when its train standard deviation is below `1e-6`.

V2-1:

```text
gate = time * nonempty * uncertainty * learned_gate
```

V2-2:

```text
risk = 0.30*component_scaled
     + 0.40*short_branch_fraction
     + 0.30*warning_scaled
reliability = exp(-2*risk)
gate = time * nonempty * uncertainty * reliability * learned_gate
```

Risk scaling uses train-only q05/q50/q95. Risk attenuation is not described as
a calibrated probability of graph correctness.

## 8. Visual uncertainty

Uncertainty is deterministic and target-free:

```text
p = softmax(Z_base.detach())
entropy = normalized categorical entropy
margin_uncertainty = 1 - (p_top1 - p_top2)
u = clamp(0.5*entropy + 0.5*margin_uncertainty, 0, 1)
```

## 9. Late correction

The correction head consumes detached visual hidden state and graph embedding.
Its final linear layer is initialized to zero.

```text
alpha = alpha_max * sigmoid(alpha_logit)
alpha_logit_init = -6
Z_final = Z_base + alpha * gate * delta_logits
```

With `alpha=0`, max absolute difference from baseline logits must be below
`1e-6`.

## 10. Loss and training

```text
L = CTC(Z_final)
  + lambda_pres * baseline_preservation_KL
  + lambda_aux(epoch) * graph_aux_CTC
```

Preservation temperature is 1.5. Development starts at
`lambda_pres=0.05`. Only the best dev variant may be repeated at 0.10.

Auxiliary schedule:

| Epochs | lambda_aux |
|---|---:|
| 1-3 | 0.15 |
| 4-6 | 0.05 |
| 7+ | 0.00 |

Optimizer is AdamW with LR `3e-4`, weight decay `1e-4`, gradient clipping
`5.0`, and batch size 16. Development has at most 20 epochs, minimum 8,
patience 5, and checkpoint selection by dev micro-CER. A reproducible
domain-balanced sampler approximates one third per core domain.

## 11. Permitted development runs

| ID | Configuration |
|---|---|
| B0-dev-v2 | fresh image-only reduced-train baseline |
| V2-1-dev-p05 | mask + uncertainty, lambda_pres=0.05 |
| V2-2-dev-p05 | V2-1 + risk attenuation |
| V2-best-dev-p10 | at most one repeat of the best variant |

The baseline plus at most three correction runs is the absolute limit.

V2 variant dev gate:

- relative CER improvement at least 1%;
- correct graph better than shuffle;
- no domain worse by more than 0.003 CER;
- Exact decreases by no more than 0.005;
- empty correction max below `1e-7`.

## 12. Holdout

The selected configuration is frozen before one-shot holdout evaluation.
The positive gate requires:

- relative CER improvement at least 2%;
- correct graph better than shuffle;
- at least two of three domains not worse;
- no domain worse by more than 0.003 CER;
- Exact not below baseline;
- WER degradation no more than 0.003.

Failure blocks final seeds and the original test.

## 13. Final protocol

Only after a positive holdout:

1. obtain matched M0-FT checkpoints for seeds 42/43/44;
2. freeze each backbone;
3. train the fixed v2 correction on canonical train;
4. select by canonical validation micro-CER;
5. open final test once after all three seeds are frozen;
6. run correct/shuffle controls and paired bootstrap/Holm analysis.

## 14. Hard stops

Stop before test if any occurs:

- empty correction is nonzero;
- frozen backbone hash changes;
- V2-1 is worse by more than 1% and preflight provides no allowed remedy;
- both V2-1 and V2-2 are worse;
- correct graph is not better than shuffle;
- holdout gate fails;
- two core domains degrade;
- an unregistered alpha/lambda/feature/architecture search is required;
- CNN or BiLSTM must be unfrozen.

After STOP, test, new gates, new encoders, feature changes, and extra
hyperparameter values are prohibited.

## 15. Primary and secondary endpoints

Primary endpoint is micro-CER. Secondary endpoints are macro-CER, WER, Exact,
char/word edits, domain CER, page-disjoint CER, and absolute robustness CER.

Technical behavior, lower CTC loss, nonzero gradients, gate variability,
zero-graph degradation, one-sample overfit, or an isolated small subgroup do
not independently confirm H4-v2.

## 16. Result tiers

- Minimum: holdout relative CER improvement >=2%, correct > shuffle, Exact not
  lower, two domains not worse, at least two of three final seeds improve.
- Clear: mean final test CER improvement >=5%, 3/3 seeds, paired CI below zero,
  no major domain degradation, Exact improves at least one percentage point.
- Strong: mean final test CER improvement >=8%, School/hard improves >=10%,
  3/3 seeds, page-disjoint not worse, most robustness conditions not worse.
- Negative: holdout fails, correct <= shuffle, seed instability, two-domain
  degradation, negligible paired effect, or alpha returns effectively to zero.

## 17. Reproducibility

Required artifacts include configs and hashes, manifests and split hashes,
normalizer, risk quantiles, checkpoints, histories, predictions, shuffle maps,
statistics, evidence manifest, SHA256SUMS, reports, figures, and thesis text.

Any scientific change after this freeze requires a documented protocol
amendment with its reason and must never rewrite v1.

