# Executive summary

## Research objective

The project investigates canonical graph descriptors of visible
handwriting strokes as an intermediate representation for offline
Russian-English handwritten text recognition.

The representation is intentionally not an estimate of pen order,
velocity, pressure, or real online trajectory. It describes reproducible
visible structure extracted from a static image.

## Main findings

### Relative robustness

The graph-vector recognizer shows lower relative CER degradation than
the image-only recognizer under the tested visual distortions.

- image-only relative degradation:
  **33.77%**
- graph-model relative degradation:
  **21.72%**
- relative advantage:
  **12.05%**
- 95% paired cluster-bootstrap interval:
  **9.37%–14.81%**
- one-sided paired permutation p:
  **0.000050**

This is a relative sensitivity result, not an absolute recognition
advantage.

### Absolute recognition

The graph model remains worse on distorted images by
**0.06297 CER**.

Strong H1 is therefore rejected.

### Visible graph quality

The original School Notebooks graph failures were traced to foreground
extraction. The deterministic `school_dark_auto` method was validated on
an independent random sample of 100 test items.

- raw good-fix rate:
  **92.00%**
- strict usable rate:
  **89.00%**
- skeleton-follows-ink rate:
  **96.00%**
- real-ink loss:
  **4.00%**
- residual background artifacts:
  **7.00%**

### Diagnostic value

The strongest structural high-error detector was localized to
`hkr_words|word|unknown`.

- ROC-AUC: **0.6723**
- PR-AUC: **0.3532**
- top-20% precision: **0.3853**

## Final interpretation

The graph representation is useful for:

- relative robustness analysis;
- preprocessing validation;
- structural inspection;
- localized failure triage.

It is not currently a successful replacement for the image-only
recognizer and does not provide superior absolute CER.
