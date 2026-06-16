# Discussion

## 1. Relative robustness and absolute recognition

The principal positive result is a statistically supported relative
robustness advantage of 12.05%. The cluster-
bootstrap interval, 9.37%–14.81%,
does not include zero, and the paired permutation test yields
p=0.000050.

This result must not be interpreted as superior recognition. The graph
model starts from a higher clean CER and remains worse on distorted
images by 0.06297 CER.

A weaker model can exhibit smaller proportional degradation because its
initial error is already high. Reporting both relative and absolute
effects is therefore necessary.

## 2. Robustness mechanisms

The strongest relative advantages occur for:

- additive noise;
- reduced contrast;
- thinning of strokes.

These perturbations alter local pixel evidence while leaving part of the
coarse structural organization recoverable.

Stroke thickening provides no advantage. This suggests that graph
descriptors are not uniformly invariant to all morphology changes.

Blur is inconclusive under the strict combined criterion: the
permutation result is positive, but the bootstrap confidence interval
crosses zero.

## 3. Foreground extraction as a structural bottleneck

The original School Notebooks failure illustrates that graph
construction cannot compensate for an incorrect foreground mask.
Background classified as foreground generates artificial components,
skeleton branches, endpoints, and graph edges.

The `school_dark_auto` repair reached a strict usable rate of
89.00% on an independent random sample.

The remaining 4.00% ink-loss rate
and 7.00% residual-
artifact rate show that the method is a substantial correction rather
than a universal segmentation solution.

## 4. Representation quality versus fusion utility

Foreground repair visibly improves skeleton and graph plausibility but
does not materially improve HTR.

This separates two questions:

1. Is the structural representation faithful enough for inspection?
2. Does the current recognition architecture use it effectively?

The experiments support the first question more strongly than the
second.

A global graph vector broadcast across all sequence positions may be too
coarse to improve local character recognition. However, further
architecture search is outside the scope of the frozen study because
the current evidence does not justify continued CER-driven tuning.

## 5. Diagnostic role

The strongest H3 result reaches ROC-AUC 0.6723 in
`hkr_words|word|unknown`.

This is useful for ranking difficult samples, prioritizing manual review,
or selecting examples for structural inspection. It is not sufficient
as a universal confidence score.

The weak global correlations indicate that recognition difficulty is
not determined by one graph statistic. It emerges from interactions
between writing style, text level, dataset, preprocessing, and model
behaviour.

## 6. Scientific contribution

The contribution is a controlled empirical characterization of a
visible-stroke structural representation.

The project demonstrates:

- how to construct reproducible graph descriptors from static handwriting;
- how upstream preprocessing failures propagate into graph topology;
- how to validate a dataset-specific foreground repair;
- how to separate relative robustness from absolute recognition quality;
- how to use graph features for localized failure triage;
- why simple global graph fusion does not automatically improve HTR.

The negative recognition result is therefore part of the contribution,
rather than a reason to discard the structural representation.
