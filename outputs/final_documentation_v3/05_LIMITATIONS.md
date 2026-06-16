# Limitations

## 1. Absolute HTR quality

The graph-aware models do not outperform the image-only baseline in
clean or distorted absolute CER.

The robustness result is restricted to relative sensitivity.

## 2. Dependence on the clean baseline

Relative degradation is normalized by clean CER. Since the graph model
starts from a worse baseline, proportional degradation can appear more
favourable even when absolute error remains higher.

The study therefore reports relative advantage, absolute degradation,
and absolute distorted CER separately.

## 3. Synthetic perturbations

The robustness protocol uses controlled synthetic blur, Gaussian noise,
contrast reduction, stroke thinning, and stroke thickening.

It does not reproduce the complete distribution of:

- camera blur;
- JPEG artifacts;
- shadows;
- page curvature;
- bleed-through;
- ink variation;
- mixed illumination;
- real scanning defects.

## 4. Manual H2 audit selection

The original H2 audit subset was selected diagnostically across error
and structural-risk strata.

Its rates characterize failure modes and cannot be interpreted as
population estimates.

## 5. Random School Notebooks validation

The independent validation contains 100 items from one test
split.

It does not establish uniform behaviour across:

- training and validation splits;
- unseen notebook collections;
- different crop-generation procedures;
- different acquisition devices;
- other handwriting datasets.

Annotations were produced by one evaluator. Inter-rater agreement was
not measured.

## 6. Remaining segmentation errors

Foreground v3 removed real ink in
4.00% of the random sample and
retained background artifacts in
7.00%.

The preprocessing method is accepted for the current pipeline but is
not a perfect foreground segmenter.

## 7. No exhaustive graph ground truth

The project does not include complete gold node-edge annotations for
every sample.

Most automated graph-quality descriptors are proxies. Visual audit
supports plausibility but does not establish exact topological
correctness.

## 8. Training variance

The graph-v3 retraining run was worse under both old and repaired feature
manifests.

This identifies a training-run effect but does not quantify full
seed-to-seed variance.

## 9. Localized H3 evidence

The strongest error-detection result is restricted to a particular
dataset and text-level subgroup.

It should not be generalized to the entire mixed corpus.

## 10. Offline structure only

The graph encodes visible static stroke structure.

It does not recover:

- stroke order;
- pen lifts;
- pressure;
- velocity;
- acceleration;
- writer motor dynamics;
- true online trajectory.
