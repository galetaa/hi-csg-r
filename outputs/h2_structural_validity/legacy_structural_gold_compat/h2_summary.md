# H2 structural validity closure

Review samples: **200**

## Overall rates

| field | n valid | rate | count 1 | count 0 |
|---|---:|---:|---:|---:|
| foreground_ok | 200 | 1.0000 | 200 | 0 |
| skeleton_usable | 200 | 1.0000 | 200 | 0 |
| graph_usable | 200 | 1.0000 | 200 | 0 |
| components_reasonable | 200 | 1.0000 | 200 | 0 |
| endpoints_reasonable | 200 | 1.0000 | 200 | 0 |
| junctions_reasonable | 200 | 1.0000 | 200 | 0 |
| usable_for_diagnostics | 200 | 1.0000 | 200 | 0 |
| loops_preserved | 0 | NA | 0 | 0 |
| severe_topology_error | 200 | 0.1150 | 23 | 177 |

## Acceptance status

- minimally supported: `True`
- strongly supported: `True`

## Failure taxonomy

| failure type | count | rate |
|---|---:|---:|
| none | 137 | 0.6850 |
| background_noise | 45 | 0.2250 |
| foreground_loss | 17 | 0.0850 |
| false_bridges | 1 | 0.0050 |

## Interpretation

H2 should be interpreted as structural usability of HI-CSG-R, not as exact recovery of pen trajectory or perfect topology.