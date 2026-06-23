# Selective Prediction - Iteration 2 Line Augmentation

## Scope

Models compared on the same tri10k mixed word-level test samples:
- baseline image-only
- +5k School contextual natural-line crops
- +10k School contextual natural-line crops

Risk score is feature-only and uses graph/foreground extremeness, School quality rules, graph warnings, and short-text risk. It does not use model confidence or prediction outputs.

## Main Results

| model | CER | WER | exact | risk AUC all | risk AUC School |
|---|---:|---:|---:|---:|---:|
| baseline | 0.1453 | 0.5134 | 0.4411 | 0.5037 | 0.4581 |
| +5k context | 0.1360 | 0.4907 | 0.4605 | 0.4964 | 0.4714 |
| +10k context | 0.1351 | 0.4924 | 0.4636 | 0.5071 | 0.4800 |

## School clean_core vs hard_real

| model | bucket | n | CER | WER | exact |
|---|---|---:|---:|---:|---:|
| baseline | clean_core | 1764 | 0.1477 | 0.4895 | 0.5193 |
| baseline | hard_real | 236 | 0.2307 | 0.5932 | 0.4068 |
| +5k context | clean_core | 1764 | 0.1312 | 0.4450 | 0.5629 |
| +5k context | hard_real | 236 | 0.2088 | 0.6144 | 0.3898 |
| +10k context | clean_core | 1764 | 0.1291 | 0.4538 | 0.5573 |
| +10k context | hard_real | 236 | 0.2126 | 0.6059 | 0.4025 |

## Answers

1. Improvement on School appears in both clean_core and hard_real.
   - +5k CER delta: clean_core -0.0165, hard_real -0.0219.
   - +10k CER delta: clean_core -0.0186, hard_real -0.0182.

2. Current graph/foreground diagnostics are useful for coarse bucket analysis but weak for exact-error risk filtering.
   - Risk AUC is near chance overall and below chance-to-weak on School.
   - Individual School graph features retain weak CER signal: cc_count rho about 0.17-0.19, fg/skel fraction about 0.13-0.19.

3. H3-style structural signal did not clearly strengthen after line augmentation.
   - CER improves, but risk/exact AUC stays near 0.5.
   - The model became better, while feature-only error predictability remains weak.

4. Feature-only selective curves reduce CER at lower coverage, but this is not strong enough yet as a confidence mechanism.
   - For +10k, School CER is 0.1154 at 50% low-risk coverage vs 0.1389 at 100%.
   - Exact does not improve monotonically with this score, so it should not be used as a production confidence filter.

## Files

- `selective_summary.json`
- `selective_summary.md`
- `risk_table_by_bucket.csv`
- `coverage_curves.csv`
