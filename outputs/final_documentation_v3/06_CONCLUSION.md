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
