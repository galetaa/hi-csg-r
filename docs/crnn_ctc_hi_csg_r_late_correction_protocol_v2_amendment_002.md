# Amendment 002: width-matched domain-balanced batching

**Protocol:** `crnn_ctc_hi_csg_r_late_correction_protocol_v2`  
**Date:** 2026-07-30  
**Scientific configuration changed:** no

The domain-balanced sampler is unchanged with respect to:

- seed;
- number of batches;
- effective batch size;
- per-domain counts in every batch;
- random sample selection and cycling policy;
- train/dev manifests;
- targets and model-error independence.

Its three selected domain streams are sorted by image width before constructing
batches, and the completed balanced batches are then shuffled. This matches
width quantiles across domains and reduces mean padding overhead from `2.1766x`
to `1.7564x`.

The random-width domain-balanced process was stopped before completing epoch 1.
It produced no checkpoint, validation evaluation, or scientific result.
