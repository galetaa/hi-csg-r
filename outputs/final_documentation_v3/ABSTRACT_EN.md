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
