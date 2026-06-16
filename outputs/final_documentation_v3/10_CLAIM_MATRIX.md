```

# Final claim matrix

## Frozen thesis claim

> Canonical visible-stroke graph descriptors provide a reproducible intermediate representation for offline handwriting analysis. They show statistically supported value for relative robustness analysis, foreground preprocessing validation, and localized recognition-error triage. Current graph-fusion models do not outperform a strong image-only recognizer in absolute character error rate.

## Claim status

| claim                                            | status                                 | mandatory qualification                 |
| ------------------------------------------------ | -------------------------------------- | --------------------------------------- |
| Graph-aware HTR is more accurate.                | not supported                          | Absolute CER is worse.                  |
| Graph-vector HTR has lower relative degradation. | supported                              | Restricted to tested distortions.       |
| Graph-vector HTR has lower absolute degradation. | not supported                          | Overall absolute advantage is negative. |
| School foreground v3 repairs extraction.         | supported on sampled test distribution | Report residual failures.               |
| Foreground repair improves recognition.          | not supported                          | Cross-evaluation change is very small.  |
| Graph descriptors detect difficult samples.      | partially supported                    | Signal is subgroup-specific.            |
| Structural risk is graph correctness.            | not supported                          | Treat as difficulty indicator.          |
| Graph is reconstructed pen trajectory.           | false by design                        | Use “visible-stroke structure”.         |

## Safe claims

- The graph-vector model has a statistically supported relative robustness advantage under the tested synthetic distortions.
- The graph-vector model remains worse than the image-only baseline in clean and distorted absolute CER.
- `school_dark_auto` substantially repairs School Notebooks foreground extraction on an independently sampled test subset.
- Improved visible graph extraction does not materially improve the tested graph-fusion recognizer.
- Multifeature structural descriptors provide localized value for high-error sample triage.
- The generated graph describes visible static stroke structure and not the true online pen trajectory.

## Claims to avoid

- Graph-aware recognition is superior to image-only recognition.
- Strong H1 is confirmed.
- Visible graph topology is preserved uniformly across all datasets and acquisition conditions.
- The random-100 School Notebooks result generalizes to all handwriting domains.
- Foreground v3 materially improves recognition accuracy.
- Structural risk is a direct gold measurement of graph correctness.
- The graph recovers the true online writing trajectory.
