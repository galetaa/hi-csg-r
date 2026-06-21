# Discussion

## Relative robustness without absolute superiority

The main robustness result is deliberately narrower than a claim of superior recognition. The graph-vector model exhibited a 12.05% reduction in relative CER degradation compared with the image-only baseline, and this effect was supported by paired cluster bootstrap and permutation testing. At the same time, the graph model started from a substantially worse clean CER and remained worse on distorted images. Relative stability therefore indicates lower sensitivity around the model's own error level, not a better HTR system.

This distinction is important because a weaker model can show a smaller proportional degradation partly because its initial error rate is already high. The paired analysis reduces, but does not eliminate, this interpretive limitation. For that reason, strong H1 is rejected and only a partial sensitivity claim is retained.

## Meaning of foreground repair

The School Notebooks investigation demonstrates that graph quality depends critically on upstream foreground extraction. The original skeleton failures were not evidence that the visible-stroke graph abstraction was intrinsically unsuitable. They were caused by page background being classified as foreground before skeletonization.

The independent random validation produced a strict usable rate of 89.00%. This supports the generality of `school_dark_auto` within the sampled School Notebooks test distribution. Nevertheless, the remaining ink-loss and residual-artifact cases show that the preprocessing rule is not perfect and should not be treated as universal.

## Why graph repair did not improve recognition

Improving visible graph quality did not materially improve the graph-fusion recognizer. This is not contradictory. A graph can be more faithful as a structural description while still adding little information beyond the convolutional image representation, or while being fused at an ineffective architectural location. The controlled cross-evaluation indicates that foreground v3 was compatible with the retained graph-v2 checkpoint, but a new training run was worse independently of which feature manifest was used.

The result therefore separates representation quality from fusion utility: a cleaner structural representation is useful for analysis and diagnostics, but it is not sufficient by itself to improve CTC recognition.

## Diagnostic role of graph descriptors

The strongest H3 result reached ROC-AUC 0.6723 in `hkr_words|word|unknown`. This level is meaningful for ranking or triage but insufficient for a general error predictor. The weak global correlations and localized multifeature gains suggest that structural difficulty interacts with dataset, text level, and writing style.

Graph-derived risk should therefore be used to prioritize manual inspection, detect suspicious preprocessing, or stratify evaluation. It should not be interpreted as a calibrated measurement of graph correctness.

## Main contribution

The strongest contribution of the project is a reproducible visible-stroke structural layer between offline handwriting images and recognition output. Its value lies in making preprocessing and structural failure modes measurable. The negative recognition result is also informative: simple global graph-vector fusion is not enough to convert structural descriptors into improved transcription.
