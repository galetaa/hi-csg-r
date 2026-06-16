# Methods

## 1. Problem formulation

Let an offline grayscale handwriting crop be denoted by \(I\). The
pipeline constructs a deterministic foreground mask \(F\), a skeleton
\(S\), and a set of graph-derived structural descriptors \(g(I)\).

The target representation is a canonical graph of visible stroke
structure. It is not intended to reconstruct the actual temporal pen
trajectory.

The recognition task predicts a character sequence from the image,
optionally conditioned on the graph descriptor vector.

## 2. Data organization

The final mixed test evaluation contains 5,563 samples:

- 1,563 Cyrillic Handwriting samples;
- 2,000 HKR samples;
- 2,000 School Notebooks samples.

Dataset identity is retained in all manifests for grouped evaluation.

Train, validation, and test partitions are represented by JSONL
manifests. Each graph-ready row contains:

- `sample_id`;
- image path;
- target transcription;
- dataset and text-level metadata;
- `graph_feature_names`;
- `graph_features`;
- graph warning information;
- preprocessing metadata.

## 3. Image preprocessing

Images are converted to grayscale.

For Cyrillic Handwriting and HKR, foreground extraction uses the
dataset-specific thresholding configuration established in the graph
feature extractor.

For School Notebooks, the final method is `school_dark_auto`:

1. threshold dark pixels at intensity 145;
2. remove connected foreground objects smaller than 4 pixels;
3. calculate foreground fraction;
4. retain threshold 145 if foreground fraction is at most 0.35;
5. otherwise repeat extraction with threshold 120.

The method was introduced because local adaptive binarization frequently
classified darker notebook background as foreground.

## 4. Skeleton and graph descriptors

The binary foreground mask is skeletonized.

The final graph vector contains 39 non-textual descriptors covering:

- crop width, height, and aspect ratio;
- foreground fraction;
- foreground bounding-box geometry;
- connected-component statistics;
- skeleton pixel fraction and component count;
- graph node and edge counts;
- average degree;
- endpoint, branch-point, and isolated-node counts;
- degree histogram;
- horizontal, vertical, and diagonal direction fractions;
- stroke-width statistics;
- graph warning count.

`text_len` is excluded from recognition and diagnostic feature sets to
prevent target-length leakage.

## 5. Recognition models

### Image-only model

The primary baseline is a convolutional recurrent CTC recognizer using
only the grayscale image.

### Graph-vector fusion model

The graph-vector model contains:

- a convolutional image encoder;
- adaptive vertical pooling;
- a graph MLP;
- temporal broadcasting of the global graph embedding;
- concatenation of image and graph representations;
- a bidirectional recurrent sequence encoder;
- a CTC classifier.

Graph features are standardized using mean and standard deviation
estimated from the training manifest.

### Controlled foreground cross-evaluation

To separate preprocessing effects from training-run effects, two
checkpoints were evaluated with two feature manifests:

- old checkpoint + old graph features;
- old checkpoint + foreground-v3 graph features;
- new checkpoint + old graph features;
- new checkpoint + foreground-v3 graph features.

## 6. Robustness protocol

Five distortion families were evaluated at three severity levels:

| family | mild | medium | strong |
|---|---:|---:|---:|
| Gaussian blur kernel | 3 | 5 | 7 |
| Gaussian-noise sigma | 8 | 16 | 24 |
| contrast alpha | 0.75 | 0.55 | 0.40 |
| stroke thinning kernel | 2 | 2 | 3 |
| stroke thickening kernel | 2 | 2 | 3 |

This produces 15 distorted conditions per source sample.

Graph features were recomputed from each distorted image. This prevents
the graph branch from receiving inherited clean-image descriptors.

## 7. Robustness estimands

For each model:

\[
D_{abs} = CER_{distorted} - CER_{clean}
\]

\[
D_{rel} =
\frac{CER_{distorted} - CER_{clean}}
     {CER_{clean}}
\]

The primary relative robustness advantage is:

\[
A_{rel} =
D_{rel}^{image}
-
D_{rel}^{graph}
\]

Positive \(A_{rel}\) means that the graph model degrades less in
relative terms.

Absolute distorted CER is reported separately.

## 8. Statistical analysis

The inferential robustness analysis uses:

- all 5,563 clean source samples;
- all 15 distortion conditions;
- cluster resampling by clean source sample;
- 5,000 paired bootstrap iterations;
- 20,000 paired permutations.

All distortion observations belonging to a clean sample remain in the
same resampling cluster.

The primary confidence interval concerns corpus-level relative
degradation advantage.

## 9. H2 manual audit

A diagnostic audit examined graph extraction quality across:

- HKR;
- Cyrillic Handwriting;
- School Notebooks.

The audit recorded:

- usability;
- critical topology errors;
- whether the skeleton follows visible ink;
- border/background artifacts;
- graph quality on a 0–3 scale;
- inferred failure stage.

The initial audit subset was selected diagnostically and is not used as
a population estimate.

## 10. Independent School Notebooks validation

After development of `school_dark_auto`, an independent random sample of
100 School Notebooks test items was evaluated.

The strict usable criterion required:

- no visible real-ink removal;
- no remaining dominant background artifact;
- skeleton following the visible handwriting.

Wilson intervals were calculated for proportion estimates.

## 11. H3 diagnostic analysis

Graph descriptors were evaluated as predictors of recognition error.

Analyses included:

- global Spearman correlations;
- structural feature subsets;
- geometry controls;
- graph-quality proxy features;
- stratification by dataset and text level;
- five-fold stratified cross-validation;
- logistic regression for top-quantile high-error detection;
- ROC-AUC;
- PR-AUC;
- top-20% precision.

The diagnostic score is interpreted as sample difficulty, not as direct
gold graph quality.
