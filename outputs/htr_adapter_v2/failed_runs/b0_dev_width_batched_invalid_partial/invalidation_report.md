# Invalidated B0-dev-v2 partial run

**Status:** `INVALIDATED_BEFORE_V2_TRAINING`  
**Holdout evaluated:** no  
**Test evaluated:** no

The run used canonical width-bucket batching for epochs 1-11. Protocol v2
section 14.6 requires domain-balanced development batches. The mismatch was
detected before B0 completion and before any V2-1/V2-2 training.

This checkpoint is retained only as provenance and must not be used as a base
checkpoint, development baseline, candidate, or scientific result.

- completed epochs: `11/80`
- last validation CER: `0.17197334808143055`
- last validation Exact: `0.35`
- invalid checkpoint SHA256:
  `432412f0349b93c98e262ae5fb4bdd291403863f98950da5cfe8e6c257e84909`
- corrected B0 config SHA256:
  `88ce9892369a4fe01a97fe2a75025c64c77729f3bfae065bdfdf0008a9e9f155`

The compliant B0 run restarts from random initialization with the same seed,
data, architecture, optimizer, schedule, and epoch budget. Only the sampler is
corrected to the already specified domain-balanced policy.
