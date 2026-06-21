# Limitations

## Absolute recognition performance

The graph-aware models do not outperform the image-only baseline in clean or distorted absolute CER. All robustness claims must therefore be stated as relative sensitivity results rather than recognition superiority.

## Synthetic distortion protocol

The robustness evaluation uses synthetic blur, additive noise, contrast reduction, and morphological stroke changes. These perturbations are controlled and reproducible but do not cover the full distribution of real scanning, camera, compression, paper, ink, and page-layout degradation.

## Relative-degradation estimand

Relative degradation depends on each model's clean error rate. Because the graph model starts from a worse baseline, proportional changes can favor it even when absolute errors remain higher. The study reports corpus-level relative and absolute effects separately to prevent these quantities from being conflated.

## Manual graph audit

The initial H2 audit was diagnostically selected across error and risk strata and should not be interpreted as a population estimate. Its purpose was failure-mode discovery and structural inspection.

## Random School Notebooks validation

The independent validation includes 100 randomly sampled items from one test split. It does not establish performance on all splits, all notebook sources, or unseen acquisition settings. The annotations were produced by one evaluator, and no inter-rater agreement estimate is available.

## Remaining preprocessing errors

Foreground v3 retained background artifacts in 7.00% of the random validation sample and removed some visible ink in 4.00%. The method is therefore a substantial repair, not a perfect segmentation solution.

## Lack of gold graph topology

The project does not contain exhaustive node-edge gold annotations for the visible-stroke graph. Most automated graph-quality variables are structural proxies. Manual skeleton inspection establishes plausibility but not exact topological accuracy.

## Model and optimization variance

The controlled graph-fusion retraining experiment produced a worse checkpoint under both old and repaired feature manifests. This demonstrates sensitivity to training dynamics, but a single retrain does not quantify full seed-to-seed variance.

## Localized H3 evidence

The strongest graph-based high-error detection result is localized to a particular dataset and text-level subgroup. It should not be generalized to all samples or interpreted as a universal confidence estimator.

## Offline visible structure, not pen trajectory

The generated graph represents reproducible visible stroke structure in a static image. It does not reconstruct writing order, pen lifts, pressure, velocity, or the true online trajectory.
