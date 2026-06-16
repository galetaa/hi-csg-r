# Canonical visible-stroke graph descriptors for offline handwritten text recognition

---

```

# Аннотация

В работе исследуется каноническое графовое представление видимой
штриховой структуры для распознавания рукописного текста по статическим
изображениям. Предлагаемое представление не восстанавливает реальную
траекторию движения пера, а описывает воспроизводимую структуру
видимых штрихов, выделенную из офлайн-изображения.

Экспериментальная оценка выполнена на смешанном русско-английском
наборе данных и включает распознавание, анализ устойчивости,
визуальный аудит графов, проверку предобработки и диагностику ошибок.
В испытаниях с 15 типами и уровнями искажений графовая модель показала
преимущество по относительной деградации CER, равное
12.05%, с 95%-м кластерным bootstrap-интервалом
9.37%–14.81% и односторонним
перестановочным p-значением
0.000050.

При этом абсолютная CER графовой модели осталась хуже, поэтому сильная
гипотеза о превосходстве графового распознавателя была отклонена.

Для School Notebooks была выявлена ошибка выделения переднего плана и
разработан детерминированный метод `school_dark_auto`. На независимой
случайной выборке из 100 примеров строгая доля пригодных
результатов составила
89.00%.

Графовые дескрипторы также показали локализованную ценность для поиска
трудных примеров: лучший результат достиг ROC-AUC
0.6723.

Полученные результаты подтверждают ценность графового представления
для анализа устойчивости, проверки предобработки и диагностики отказов,
но не подтверждают повышение абсолютной точности распознавания при
использовании исследованных схем графового слияния.

## Ключевые слова

распознавание рукописного текста; офлайн-почерк; граф штрихов;
скелетизация; структурные признаки; CTC; устойчивость; анализ ошибок.

---

```

# Abstract

This study investigates canonical graph descriptors of visible stroke
structure as an intermediate representation for offline handwritten
text recognition. The representation does not reconstruct the actual
pen trajectory; instead, it captures reproducible visible structure
from static handwriting images.

The experimental evaluation covers recognition, robustness, visual
graph audit, preprocessing validation, and error diagnostics on a mixed
Russian-English handwriting corpus. Across 15 distortion conditions,
the graph-vector model achieved a relative CER degradation advantage of
12.05%, with a 95% paired cluster-bootstrap
interval of 9.37%–14.81% and a
one-sided paired permutation p-value of
0.000050.

The graph model nevertheless retained worse absolute CER, and the strong
hypothesis of superior graph-aware recognition was rejected.

A foreground-extraction failure was identified for School Notebooks,
leading to the deterministic `school_dark_auto` repair. Independent
validation on 100 randomly sampled test items yielded a
strict usable rate of 89.00%.

Graph descriptors also showed localized value for high-error detection,
with a best ROC-AUC of 0.6723.

The results support visible-stroke graph descriptors as tools for
relative robustness analysis, preprocessing validation, and failure
triage, but not as a currently superior route to absolute recognition
accuracy.

## Keywords

handwritten text recognition; offline handwriting; stroke graph;
skeletonization; structural descriptors; CTC; robustness; error
analysis.

---

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

---

# Results

## 1. Recognition performance

| model/checkpoint | feature manifest | CER | WER | exact |
|---|---|---:|---:|---:|
| graph-v2 | old | 0.13970 | 0.49042 | 0.43897 |
| graph-v2 | foreground v3 | 0.13943 | 0.48975 | 0.43969 |
| graph-v3 retrain | old | 0.15396 | 0.52364 | 0.40626 |
| graph-v3 retrain | foreground v3 | 0.15338 | 0.52190 | 0.40823 |

Replacing the old graph features with foreground-v3 features changed the
retained graph-v2 CER by
-0.00027.

The new checkpoint remained worse with both feature manifests. Its
degradation was therefore caused primarily by the training run rather
than by foreground repair.

## 2. Descriptive robustness

| model | clean CER | mean distorted CER | absolute delta | relative degradation |
|---|---:|---:|---:|---:|
| image-only | 0.08224 | 0.11365 | 0.03141 | 38.20% |
| graph-vector, recomputed features | 0.13943 | 0.16971 | 0.03028 | 21.72% |

The graph-vector recognizer has lower proportional degradation but worse
clean and distorted absolute CER.

## 3. Paired corpus robustness

| metric | result |
|---|---:|
| image-only relative degradation | 33.77% |
| graph relative degradation | 21.72% |
| relative advantage | 12.05% |
| relative advantage 95% CI | 9.37%–14.81% |
| one-sided permutation p | 0.000050 |
| absolute degradation advantage | -0.00333 |
| absolute advantage 95% CI | -0.00528–-0.00137 |
| graph − image distorted CER | 0.06297 |

The graph model has a statistically supported relative robustness
advantage. It does not have a positive absolute degradation advantage
and remains worse in absolute distorted CER.

## 4. Robustness by distortion family

| family | image relative | graph relative | advantage | 95% CI | p | verdict |
|---|---:|---:|---:|---:|---:|---|
| `blur` | 23.23% | 20.66% | 2.58% | -0.44%–5.52% | 0.028399 | inconclusive |
| `low_contrast` | 31.19% | 15.72% | 15.48% | 11.87%–19.17% | 0.000050 | supported |
| `noise` | 32.81% | 11.80% | 21.01% | 17.95%–24.11% | 0.000050 | supported |
| `thick_strokes` | 14.68% | 14.77% | -0.09% | -2.90%–2.77% | 0.529624 | not supported |
| `thin_strokes` | 66.94% | 45.64% | 21.29% | 17.05%–25.97% | 0.000050 | supported |

Relative robustness is supported for low contrast, additive noise, and
stroke thinning. Blur is inconclusive under the combined confidence-
interval and permutation criterion. Stroke thickening provides no
relative advantage.

## 5. Original H2 diagnostic audit

| subset | n | critical topology error | skeleton follows ink | mean graph quality |
|---|---:|---:|---:|---:|
| HKR + Cyrillic | 77 | 2.60% | 96.10% | 2.870 |
| School Notebooks, old preprocessing | 23 | 95.65% | 0.00% | 0.826 |

The School Notebooks failure was localized to foreground extraction.

## 6. Independent random-100 foreground validation

| metric | count | rate |
|---|---:|---:|
| raw good fix | 92/100 | 92.00% |
| partial fix | 8/100 | 8.00% |
| bad fix | 0/100 | 0.00% |
| strict usable | 89/100 | 89.00% |
| real ink erased | 4/100 | 4.00% |
| residual artifact | 7/100 | 7.00% |
| skeleton follows ink | 96/100 | 96.00% |

The random validation supports `school_dark_auto` for the sampled School
Notebooks test distribution.

## 7. H3 graph diagnostics

| metric | result |
|---|---:|
| best global feature | `graph_endpoint_count` |
| global Spearman r | 0.0981 |
| structural feature set | `structural_core` |
| subgroup | `hkr_words|word|unknown` |
| n | 1090 |
| ROC-AUC | 0.6723 |
| PR-AUC | 0.3532 |
| PR-AUC lift | 1.6666 |
| top-20% precision | 0.3853 |

Individual global descriptors have weak correlations with CER.
Multifeature graph descriptors provide useful but localized high-error
detection.

---

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

---

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

---

# Conclusion

This study evaluated canonical visible-stroke graph descriptors as an
intermediate representation for offline handwritten text recognition.

The graph-vector recognizer demonstrated a statistically supported
relative robustness advantage of 12.05%, with a
95% cluster-bootstrap interval of
9.37%–14.81%.

Strong H1 was nevertheless rejected because:

- the absolute degradation advantage was not positive;
- the graph model had worse clean CER;
- the graph model had worse distorted CER;
- the final distorted CER gap was 0.06297.

The structural audit showed that graph plausibility depends strongly on
foreground extraction. The `school_dark_auto` repair achieved a strict
usable rate of 89.00% on an independent
random School Notebooks test sample.

Graph-derived descriptors also provided localized high-error detection,
with a best ROC-AUC of 0.6723. Their value is therefore
diagnostic rather than universally predictive.

The final contribution is a reproducible structural analysis framework
for:

- visible-stroke graph extraction;
- robustness evaluation;
- preprocessing failure diagnosis;
- structural audit;
- difficult-sample triage.

Current graph-fusion models do not provide superior recognition
accuracy. Future work should focus on localized structural
representations, stronger graph supervision, and broader real-world
degradation evaluation rather than continued tuning of the current
global graph-vector architecture.
