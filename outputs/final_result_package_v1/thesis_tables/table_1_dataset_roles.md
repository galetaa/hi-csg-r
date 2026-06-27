# Table 1. Dataset roles

| dataset | language | script | level | role | used_in_primary_htr | used_in_structural_diagnostics | used_in_selective_prediction | used_in_stress_analysis | primary_claim | limitations |
|---|---|---|---|---|---|---|---|---|---|---|
| Cyrillic Handwriting | Russian | Cyrillic | word/phrase crops | core crop-domain | yes | yes | yes | yes | Improves across 3 seeds in domain-wise aggregation. | Crop-domain data; not the hardest notebook layout. |
| HKR Words | Russian | Cyrillic | word/phrase crops | secondary word-domain | yes | yes | yes | yes | Improves on average; not fully seed-stable domain-wise. | HKR domain has 2/3 improved seeds. |
| School Notebooks | Russian | Cyrillic | word crops and natural-line context | hard notebook domain | yes | yes | yes | yes | Strongest and most stable HTR gain. | Line crops are contextual, not clean isolated line crops. |
| HWR200 | Russian | Cyrillic | diagnostic/stress | diagnostic/stress only | no | limited | no | yes | Not part of primary HTR claim. | Used only as supporting stress/diagnostic material. |
| HKR Forms | Russian | Cyrillic | diagnostic/stress | diagnostic/stress only | no | limited | no | yes | Not part of primary HTR claim. | Used only as supporting stress/diagnostic material. |
| IAM | English | Latin | background only | optional background | no | no | no | no | Does not support Russian-domain claim. | Different language/script; not used for final Russian HTR claim. |
