# HI-CSG-R Late Correction v2 amendment 001 freeze

**Implementation commit:** `5e2fd5b`  
**Scientific configuration changed:** no  
**Holdout evaluated:** no  
**Test evaluated:** no

The initial B0 partial run used width-bucket batches and was invalidated after
11 of 80 epochs. No correction model was trained from it. The compliant fresh
B0 run uses the domain-balanced sampler already required by protocol section
14.6.

The invalid checkpoint is retained only for provenance under
`outputs/htr_adapter_v2/failed_runs/` and is explicitly excluded from
selection and reporting.
