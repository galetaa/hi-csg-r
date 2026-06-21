# School foreground v3 random validation

## 1. Sampling

- independently sampled test samples: 100
- selected preprocessing: `school_dark_auto`

## 2. Results

| metric | count | rate | 95% Wilson CI |
|---|---:|---:|---:|
| school_dark_auto selected | 100/100 | 1.000 | 0.963–1.000 |
| good fix | 92/100 | 0.920 | 0.850–0.959 |
| partial fix | 8/100 | 0.080 | 0.041–0.150 |
| bad fix | 0/100 | 0.000 | 0.000–0.037 |
| real ink erased | 4/100 | 0.040 | 0.016–0.098 |
| background artifact remains | 7/100 | 0.070 | 0.034–0.137 |
| skeleton follows ink | 96/100 | 0.960 | 0.902–0.984 |
| strict good | 89/100 | 0.890 | 0.814–0.937 |
| strict usable | 89/100 | 0.890 | 0.814–0.937 |

## 3. Annotation QA

| sample | target | issue |
|---|---|---|
| `school_notebooks_test_0305762` | `тем` | `good_fix_with_remaining_artifact` |
| `school_notebooks_test_0322172` | `она` | `good_fix_with_remaining_artifact` |
| `school_notebooks_test_0323236` | `не` | `good_fix_with_erased_ink` |

## 4. Verdict

`school_dark_auto` is supported by an independent random validation sample. The raw good-fix rate is reported together with the stricter usable rate, which requires no erased ink, no remaining dominant background artifact, and a skeleton that follows visible ink.

The result supports generalization to the sampled School Notebooks test distribution. It does not establish performance across all splits, all possible acquisition conditions, or independent annotators.
