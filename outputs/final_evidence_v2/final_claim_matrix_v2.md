# Final claim matrix — v2

| claim | status | required wording |
|---|---|---|
| Graph-aware HTR outperforms image-only HTR. | not supported | Do not claim. |
| Graph-vector HTR is less sensitive to tested distortions in relative CER terms. | supported | State together with worse absolute CER. |
| Graph-vector HTR has lower absolute degradation. | not supported | Report the negative absolute-advantage estimate. |
| The graph pipeline preserves visible structure on audited HKR/Cyrillic samples. | partially supported | Restrict to the diagnostic audit evidence. |
| `school_dark_auto` repairs School Notebooks foreground extraction. | supported on sampled test distribution | Report random-100 rates and remaining ink-loss/artifact failures. |
| Foreground v3 improves graph-fusion recognition. | not supported | It gives only a very small inference-time CER change. |
| Graph descriptors identify difficult samples. | partially supported | Restrict to localized multifeature results. |
| Structural risk directly measures graph correctness. | not supported | Describe it as a hard-sample indicator. |
| The graph reconstructs real pen trajectory. | not supported by design | Use visible-stroke structural representation. |

## Frozen thesis claim

> Canonical visible-stroke graph descriptors provide a reproducible intermediate representation for offline handwriting analysis. They show statistically supported value for relative robustness analysis, foreground-preprocessing validation, and localized recognition-error triage. However, current graph-fusion models do not outperform a strong image-only recognizer in absolute character error rate.
