# 01 — Research claims

## Main claim

Canonical visible-stroke graph descriptors provide a useful intermediate diagnostic representation for offline handwritten text recognition. They support robustness analysis, high-error sample triage, and manual failure analysis. Current graph-aware recognition models do not outperform a strong image-only baseline in absolute CER.

## Safe claims

- Canonical visible-stroke graph descriptors are diagnostically useful in some stratified settings.
- Graph-aware HTR variants are relatively less sensitive to distortions, but not better recognizers in absolute CER.
- HKR/Cyrillic graph extraction preserved visible stroke structure reasonably well in the diagnostic audit subset.
- School-notebooks failures are dominated by upstream crop/binarization border artifacts.
- The current structural risk score is a hard-sample indicator, not a direct graph-quality score.

## Unsafe claims

- Do not claim that graph-aware recognition beats the image-only baseline.
- Do not claim H1 is fully confirmed.
- Do not claim H2 holds uniformly across all datasets.
- Do not claim structural risk is equivalent to graph quality.
- Do not present school-notebooks failures as pure graph-topology failures.

## Exact recommended wording

> Canonical visible-stroke graph descriptors provide a reproducible structural representation for offline handwriting images. In the current experiments, these descriptors are useful for robustness analysis and failure triage, but graph-aware recognition models remain worse than the image-only baseline in absolute CER. Manual audit further shows that some severe graph failures arise from upstream crop and binarization artifacts, especially in school-notebook samples.

## One-sentence contribution

The contribution is an interpretable visible-stroke graph diagnostic framework for offline handwriting, not a superior recognizer.
