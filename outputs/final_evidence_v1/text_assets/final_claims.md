# Final claims — v1

## 1. Main claim

Canonical visible-stroke graph descriptors provide a useful intermediate diagnostic representation for offline handwritten text recognition. They support robustness analysis and failure triage, but the current graph-aware recognizers do not outperform a strong image-only baseline in absolute CER.

## 2. Short thesis claim

The project demonstrates that visible-stroke structural descriptors can expose robustness and preprocessing failure modes in offline HTR, while also showing that naive graph fusion is not sufficient to improve recognition accuracy.

## 3. H1 claim

Graph-aware variants show lower relative CER degradation under synthetic visual distortions, but because they have substantially worse clean and distorted absolute CER, H1 is only partially supported.

## 4. H2 claim

Manual diagnostic audit suggests that the graph pipeline preserves visible stroke structure reasonably well on HKR and Cyrillic samples. School-notebooks samples are excluded from the graph-topology preservation claim because their failures are dominated by crop/border/binarization artifacts.

## 5. H3 claim

Graph-derived structural descriptors provide useful but localized high-error detection in stratified subsets. However, individual global features are weak, and structural risk should not be interpreted as graph quality.

## 6. Claims to avoid

- Graph-aware HTR beats image-only HTR.
- H1 is confirmed.
- H2 is confirmed uniformly across all datasets.
- School-notebooks failures prove the graph abstraction fails.
- Structural risk is the same as graph quality.
- Current graph features reconstruct real pen trajectory.

## 7. Abstract-style paragraph

We investigate canonical visible-stroke graph descriptors as an intermediate representation for offline Russian-English handwritten text recognition. The representation is not intended to reconstruct real pen trajectories, but to capture reproducible visible stroke structure from static images. Across robustness, diagnostic, and manual audit experiments, graph-derived descriptors show partial value for relative robustness analysis and high-error sample triage. However, graph-aware recognition models do not outperform a strong image-only baseline in absolute CER, and manual audit reveals that failures in school-notebook samples are dominated by upstream crop and binarization artifacts. These results support graph descriptors as an interpretability and failure-analysis tool rather than as a standalone path to improved recognition accuracy.
