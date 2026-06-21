# Final limitations — v1

## 1. Recognition performance limitation

The graph-aware recognition models do not outperform the image-only baseline in absolute CER. Their lower relative degradation under distortions should not be interpreted as better recognition performance. Because their clean CER is substantially worse, relative degradation alone is insufficient evidence of superior robustness as an HTR system.

## 2. H1 limitation

H1 is supported only in a weak relative sense. The evidence shows reduced relative degradation, not improved absolute recognition. The robust interpretation is therefore limited to sensitivity analysis, not to a claim of a better recognizer.

## 3. H2 audit limitation

The H2 manual audit used a diagnostic candidate pool selected across CER/risk quadrants. It is not a random population sample. Therefore its rates characterize failure modes and provide audit evidence, but they should not be reported as dataset-level graph-quality estimates.

## 4. School-notebooks preprocessing limitation

The school-notebooks subset is dominated by crop/border/binarization artifacts. These failures occur before canonical graph construction. They should be reported as upstream preprocessing failures, not as direct failures of the graph abstraction. A simple border-suppression rule was tested and rejected because it was unreliable.

## 5. H3 limitation

H3 is only partially supported. Global single-feature graph metrics are weak correlates of CER. Multifeature structural descriptors can detect high-error samples in some stratified subsets, but the effect is localized. The current structural risk score is a hard-sample indicator, not a direct graph-quality score.

## 6. Graph quality limitation

Most graph-quality values used in automated experiments are proxy structural descriptors rather than gold graph accuracy measurements. A larger manually annotated gold subset would be required to estimate graph preservation quantitatively at population level.

## 7. Generalization limitation

The current evidence is strongest for the audited HKR and Cyrillic subsets. It does not establish uniform behavior across all handwriting sources, crop styles, page backgrounds, or scanning conditions.

## 8. Recommended wording

Use cautious language: partial support, diagnostic utility, failure triage, robustness analysis, and preprocessing limitation. Avoid language suggesting state-of-the-art recognition, full confirmation, or uniform graph quality across datasets.
