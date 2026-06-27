# Table 3. Domain-wise HTR result

Natural-line context augmentation gives a seed-stable overall gain, with the strongest and most stable effect on School Notebooks.

| domain | mean_baseline_CER | mean_plus10k_CER | mean_delta_CER | relative_delta_CER | improved_seeds | interpretation |
|---|---|---|---|---|---|---|
| cyrillic_handwriting | 0.203650 | 0.189479 | -0.014170 | -6.75% | 3/3 | improves in all available seeds |
| hkr_words | 0.101083 | 0.091986 | -0.009097 | -8.30% | 2/3 | mostly improves, but not fully seed-stable |
| school_notebooks_clean | 0.163752 | 0.136771 | -0.026981 | -16.38% | 3/3 | improves in all available seeds |
