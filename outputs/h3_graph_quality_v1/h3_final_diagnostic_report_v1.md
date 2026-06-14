# H3 final diagnostic report — v1

## 1. Verdict

```text
Global single-feature H3: not supported
Stratified multifeature H3: partially supported
Overall H3: partial support
```

The current graph features do not strongly explain CER as individual global variables. However, structural graph descriptors provide useful high-error detection in some stratified subsets, especially HKR word samples.

## 2. Global single-feature result

Best global Spearman feature: `dir_v_frac` with r = -0.1049.
Best single-feature high-error detector: `dir_v_frac` with ROC-AUC = 0.5658.

Interpretation: this is weak. It is below the useful diagnostic threshold.

## 3. Multifeature stratified result

| feature set | group | n | ROC-AUC | PR-AUC | lift | top20 precision |
|---|---|---:|---:|---:|---:|---:|
| `structural_core` | `hkr_words|word|unknown` | 1090 | 0.6733 | 0.3466 | 1.6356 | 0.3761 |
| `all_non_geometry` | `hkr_words|word|unknown` | 1090 | 0.6733 | 0.3466 | 1.6356 | 0.3761 |
| `geometry_control` | `hkr_words` | 2000 | 0.5705 | 0.2399 | 1.1643 | 0.2575 |
| `quality_only` | `cyrillic_handwriting` | 1563 | 0.5000 | 0.2386 | 1.0000 | 0.2386 |

## 4. Methodological interpretation

The useful signal comes from multifeature structural descriptors, not from `warning_count`. Therefore, the current warning proxy should not be used as the main graph-confidence measure.

The result is not strong enough to claim that graph features explain recognition errors globally. It is strong enough to justify using graph structural descriptors for failure-case triage and for selecting samples for gold structural annotation.

## 5. Consequence for the project

```text
Do not add new HTR architectures.
Do not claim H3 is fully confirmed.
Use H3 results to guide H2 gold-subset sampling and failure analysis.
```

## 6. Next step

Build an H2/H3 audit candidate pool with four types of samples: high-error/high-structural-risk, high-error/low-structural-risk, low-error/high-structural-risk, and low-error/low-structural-risk. This will expose whether graph structure actually corresponds to visible stroke failures.