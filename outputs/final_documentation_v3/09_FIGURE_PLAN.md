# Final figure plan and captions

## Figure 1. End-to-end HI-CSG-R pipeline

### Content

A horizontal pipeline:

```text
grayscale crop
→ foreground extraction
→ skeleton
→ canonical graph descriptors
→ graph-aware HTR / diagnostics
```

### Caption

**Figure 1.** Overview of the HI-CSG-R pipeline. The method converts an
offline grayscale handwriting crop into a deterministic visible-stroke
foreground mask, skeleton, and canonical structural descriptor vector.
The graph represents visible static structure and does not reconstruct
the real pen trajectory.

## Figure 2. School Notebooks preprocessing failure and repair

### Panels

1. original crop;
2. old adaptive foreground;
3. old skeleton;
4. `school_dark_auto` foreground;
5. repaired skeleton.

### Caption

**Figure 2.** Representative School Notebooks foreground-extraction
failure and deterministic repair. The old preprocessing classifies
darker page background as foreground, producing artificial skeleton
structure. `school_dark_auto` suppresses the background while retaining
the visible handwriting.

### Source

```text
outputs/h2_gold_audit_v1/school_foreground_v3/
outputs/h2_gold_audit_v1/school_foreground_v3_random/
```

## Figure 3. Relative robustness by distortion family

### Plot

Grouped bars or point estimates with 95% intervals:

* image-only relative degradation;
* graph-model relative degradation;
* relative advantage interval.

### Caption

**Figure 3.** Relative CER degradation under five synthetic distortion
families. The graph-vector model shows statistically supported relative
advantages for low contrast, additive noise, and stroke thinning. Blur
is inconclusive under the combined criterion, and stroke thickening
shows no advantage.

### Source

```text
outputs/robustness_v2_recomputed/
paired_corpus_v3/paired_corpus_v3.json
```

## Figure 4. Relative robustness versus absolute CER

### Plot

Two-axis or paired-panel visualization:

* relative degradation;
* mean distorted CER.

### Caption

**Figure 4.** Relative robustness does not imply superior absolute
recognition. Although the graph-vector model degrades less
proportionally, its clean and distorted CER remain higher than those of
the image-only baseline.

## Figure 5. Random-100 School Notebooks validation

### Plot

Bar chart with Wilson intervals:

* good fix;
* partial fix;
* strict usable;
* ink loss;
* residual artifact;
* skeleton follows ink.

### Caption

**Figure 5.** Independent validation of `school_dark_auto` on 100
randomly sampled School Notebooks test items. The method achieves high
visual repair and skeleton-following rates while retaining a small
residual ink-loss and background-artifact rate.

## Figure 6. H3 structural high-error detection

### Plot

ROC or precision-ranking plot for the best structural-core subgroup.

### Caption

**Figure 6.** Localized high-error detection using multifeature graph
descriptors. The strongest signal is observed in the HKR word subgroup,
while global individual-feature correlations remain weak.

### Source

```text
outputs/h3_graph_quality_v1/
after_school_fg_v3_auto/
```

## Figure-production rule

All final figures must:

* derive values from machine-readable JSON;
* avoid manually retyping numerical results;
* include sample counts;
* distinguish descriptive and inferential statistics;
* state when intervals are bootstrap or Wilson intervals;
* avoid implying absolute HTR superiority.
