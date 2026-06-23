# Confidence-Aware Selective Prediction v1

## Setup

Risk models are fit on validation exact-error labels and evaluated on test.

Compared risk sources:
- `feature_only`: graph/foreground/quality features only
- `model_confidence`: CTC confidence features only
- `confidence_graph`: CTC confidence + graph/foreground/quality features

Confidence features:
- mean max probability
- mean entropy
- CTC blank ratio
- decoded length per frame
- average decoded-character confidence
- frame-level margin
- sequence score proxy

## Risk Quality

| model | risk method | AUC all | AUC School | ECE all | ECE School |
|---|---|---:|---:|---:|---:|
| baseline | feature_only | 0.6021 | 0.6216 | 0.0647 | 0.0827 |
| baseline | model_confidence | 0.8020 | 0.8144 | 0.0445 | 0.0734 |
| baseline | confidence_graph | 0.8148 | 0.8220 | 0.0175 | 0.0420 |
| +5k context | feature_only | 0.6120 | 0.6280 | 0.0472 | 0.0735 |
| +5k context | model_confidence | 0.7990 | 0.8185 | 0.0439 | 0.0866 |
| +5k context | confidence_graph | 0.8158 | 0.8286 | 0.0226 | 0.0276 |
| +10k context | feature_only | 0.6036 | 0.5987 | 0.0569 | 0.0808 |
| +10k context | model_confidence | 0.7967 | 0.8199 | 0.0383 | 0.0772 |
| +10k context | confidence_graph | 0.8122 | 0.8284 | 0.0251 | 0.0198 |

## Selective School Coverage

| model | risk method | coverage | CER | WER | exact |
|---|---|---:|---:|---:|---:|
| +5k context | model_confidence | 0.50 | 0.0523 | 0.2160 | 0.7870 |
| +5k context | model_confidence | 0.80 | 0.0866 | 0.3578 | 0.6488 |
| +5k context | model_confidence | 1.00 | 0.1403 | 0.4650 | 0.5425 |
| +5k context | confidence_graph | 0.50 | 0.0531 | 0.2080 | 0.7950 |
| +5k context | confidence_graph | 0.80 | 0.0887 | 0.3559 | 0.6506 |
| +5k context | confidence_graph | 1.00 | 0.1403 | 0.4650 | 0.5425 |
| +10k context | model_confidence | 0.50 | 0.0529 | 0.2270 | 0.7780 |
| +10k context | model_confidence | 0.80 | 0.0870 | 0.3563 | 0.6525 |
| +10k context | model_confidence | 1.00 | 0.1389 | 0.4718 | 0.5390 |
| +10k context | confidence_graph | 0.50 | 0.0551 | 0.2230 | 0.7840 |
| +10k context | confidence_graph | 0.80 | 0.0884 | 0.3563 | 0.6519 |
| +10k context | confidence_graph | 1.00 | 0.1389 | 0.4718 | 0.5390 |

## Interpretation

Feature-only risk is now a useful weak baseline, but it is not enough for strong selective prediction. Model confidence is the main signal: it raises exact-error AUC to about 0.80 overall and about 0.82 on School.

Adding graph/foreground features to confidence gives a small but consistent AUC improvement and a clear calibration improvement. The strongest calibration result is +10k `confidence_graph` on School with ECE 0.0198.

For a practical selective setting, use `confidence_graph` as the default risk model. At 80% School coverage it gives about 0.088 CER for +10k, compared with 0.139 CER at full coverage.

## Files

- `selective_summary.json`
- `selective_summary.md`
- `coverage_curves.csv`
- `risk_table_by_bucket.csv`
- `calibration_bins.csv`
- `confidence_predictions/*.jsonl`
