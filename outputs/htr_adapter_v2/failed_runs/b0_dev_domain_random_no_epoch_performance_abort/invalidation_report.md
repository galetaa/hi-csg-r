# Aborted B0 sampler performance check

**Status:** `ABORTED_BEFORE_FIRST_EPOCH`  
**Scientific result:** none  
**Holdout/test evaluated:** no

The first domain-balanced implementation mixed arbitrary image widths and
produced `2.1766x` mean padding overhead. It was stopped before completing
epoch 1 and produced no checkpoint or validation metrics.

The replacement sampler preserves the same per-domain counts, samples,
seed-based selection and batch size, but aligns the three domain streams by
width quantile. Mean measured padding overhead is `1.7564x`.

This directory is provenance only and does not count as a development model.
