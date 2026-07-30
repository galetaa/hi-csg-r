# HI-CSG-R Late Correction v2 protocol freeze

**Frozen before development training:** yes  
**Implementation commit:** `0e8ee2c`  
**Preflight:** `CONTINUE_FULL`  
**Blank logit penalty:** `-0.4`  
**Alpha maximum:** `0.25`  
**Split seed:** `20260730`

The frozen split contains 35,498 train, 3,000 dev, and 1,500 holdout samples.
Dev and holdout contain exactly 1,000/500 samples from each core domain.

All pairwise sample ID, image path, hierarchy group, and exact image SHA1
overlaps are zero. Normalizer and structural-risk quantiles were fitted on the
reduced train split only.

Complete SHA256 values are stored in `protocol_freeze.json`. Scientific
changes after this point require a protocol amendment and must not modify the
v1 result.

