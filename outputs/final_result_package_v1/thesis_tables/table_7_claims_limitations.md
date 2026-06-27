# Table 7. Claims and limitations

| claim | supported_by | strength | limitation | allowed_wording | forbidden_wording |
|---|---|---|---|---|---|
| +10k natural-line context improves HTR. | 3 seeds; seed provenance; domain-wise aggregation. | strong | Strongest on School; HKR is not 3/3 stable. | Natural-line context augmentation improves image-only HTR across 3 seeds, especially on School Notebooks. | The improvement is equally stable in every domain. |
| HI-CSG-R is diagnostically usable. | Structural gold diagnostic subset. | moderate/strong diagnostic | Not a pixel-level topology benchmark. | HI-CSG-R provides a structurally usable diagnostic representation. | HI-CSG-R recovers true pen trajectory or full topology. |
| Selective prediction provides a reliability layer. | Canonical +10k confidence/graph-quality checks and leakage clearance. | secondary applied | Coverage thresholds are not group-fair globally. | Selective prediction supports risk-aware filtering. | Selective prediction improves full-coverage CER. |
| Graph-fusion improves recognition. | Single exploratory graph-fusion pilot. | weak/exploratory | No seed-stable superiority; mixed domain effects. | Graph-fusion shows limited/domain-dependent effects and is exploratory. | Graph-fusion proves universal recognition superiority. |
