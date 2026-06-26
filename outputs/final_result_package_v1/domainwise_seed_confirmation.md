# Domain-wise seed confirmation v1

## Purpose

This report checks whether the +10k natural-line context improvement is distributed across domains or concentrated in the hard notebook domain.

## Overall domain-wise verdict

- verdict: `STRONG_HARD_DOMAIN`
- claim: Natural-line context is especially supported for the School/hard domain, with no domain consistently worsened across all seeds.

## Domain summary

| domain | seeds | mean baseline CER | mean +10k CER | mean ΔCER | relative ΔCER | improved seeds | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| cyrillic_handwriting | 3 | 0.203650 | 0.189479 | -0.014170 | -6.75% | 3/3 | improves in all available seeds |
| hkr_words | 3 | 0.101083 | 0.091986 | -0.009097 | -8.30% | 2/3 | mostly improves, but not fully seed-stable |
| school | 3 | 0.163752 | 0.136771 | -0.026981 | -16.38% | 3/3 | improves in all available seeds |
| school_notebooks_clean | 3 | 0.163752 | 0.136771 | -0.026981 | -16.38% | 3/3 | improves in all available seeds |

## Per-seed domain deltas

| seed | domain | baseline CER | +10k CER | ΔCER | relative ΔCER |
|---:|---|---:|---:|---:|---:|
| 42 | cyrillic_handwriting | 0.193371 | 0.185807 | -0.007564 | -3.91% |
| 43 | cyrillic_handwriting | 0.199107 | 0.191315 | -0.007792 | -3.91% |
| 44 | cyrillic_handwriting | 0.218471 | 0.191316 | -0.027156 | -12.43% |
| 42 | hkr_words | 0.095176 | 0.091718 | -0.003458 | -3.63% |
| 43 | hkr_words | 0.096248 | 0.096557 | 0.000308 | 0.32% |
| 44 | hkr_words | 0.111824 | 0.087683 | -0.024141 | -21.59% |
| 42 | school | 0.158262 | 0.138929 | -0.019333 | -12.22% |
| 43 | school | 0.162400 | 0.135402 | -0.026998 | -16.62% |
| 44 | school | 0.170594 | 0.135983 | -0.034611 | -20.29% |
| 42 | school_notebooks_clean | 0.158262 | 0.138929 | -0.019333 | -12.22% |
| 43 | school_notebooks_clean | 0.162400 | 0.135402 | -0.026998 | -16.62% |
| 44 | school_notebooks_clean | 0.170594 | 0.135983 | -0.034611 | -20.29% |

## Strict interpretation

If the improvement is concentrated in School/hard-domain data, the final thesis claim should be phrased as hard-domain/context improvement, not as universal improvement across all Russian handwriting datasets.