# Seed confirmation report v1

## Purpose

This report checks whether the primary HTR improvement from natural-line context augmentation is stable across three training seeds.

## Models

- `baseline`: image-only baseline
- `plus_10k`: image-only + 10k natural-line context augmentation

## Per-seed CER deltas

| seed | baseline CER | +10k CER | ΔCER | relative ΔCER | baseline WER | +10k WER | ΔWER | baseline exact | +10k exact | Δexact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.145446 | 0.135127 | -0.010319 | -7.09% | 0.512838 | 0.492426 | -0.020412 | 0.440949 | 0.463599 | 0.022650 |
| 43 | 0.148931 | 0.137146 | -0.011785 | -7.91% | 0.515196 | 0.493280 | -0.021916 | 0.435736 | 0.458745 | 0.023009 |
| 44 | 0.162917 | 0.134165 | -0.028752 | -17.65% | 0.556900 | 0.481263 | -0.075637 | 0.396009 | 0.472766 | 0.076757 |

## Aggregate

- mean ΔCER: `-0.016952`
- std ΔCER: `0.010246`
- mean relative ΔCER: `-10.89%`
- improved CER seeds: `3/3`
- mean ΔWER: `-0.039321`
- mean Δexact: `0.040805`
- interpretation: +10k natural-line context improves CER in all three seeds; primary HTR gain is seed-stable.

## Model means

| model | mean CER | std CER | mean WER | std WER | mean exact | std exact |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.152431 | 0.009247 | 0.528311 | 0.024787 | 0.424232 | 0.024580 |
| plus_10k | 0.135479 | 0.001521 | 0.488990 | 0.006705 | 0.465037 | 0.007120 |

## Strict interpretation

This result supports the primary HTR claim only if +10k improves CER consistently across seeds. It does not prove graph-fusion superiority. It supports the data/context part of the final experimental protocol.