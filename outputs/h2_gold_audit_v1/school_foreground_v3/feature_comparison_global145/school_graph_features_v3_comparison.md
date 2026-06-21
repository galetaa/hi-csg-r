# School graph features v3 comparison

old: `data/experiments/htr_graph_v1/features/tri10k`
new: `data/experiments/htr_graph_v1/features/tri10k_school_fg_v3_global145`

## Summary by split

| split | n | old fg | new fg | old skel | new skel | old warnings | new warnings | old high-fg | new high-fg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 10000 | 0.1955 | 0.1679 | 0.0574 | 0.0430 | 0.0002 | 0.0477 | 0.0001 | 0.0477 |
| `val` | 2000 | 0.2073 | 0.1602 | 0.0628 | 0.0437 | 0.0000 | 0.0335 | 0.0000 | 0.0335 |
| `test` | 2000 | 0.2026 | 0.1559 | 0.0618 | 0.0440 | 0.0000 | 0.0175 | 0.0000 | 0.0175 |

## Interpretation

A useful foreground fix should reduce excessive foreground and skeleton clutter without collapsing skeletons to near zero. Inspect the contact sheet and this table together; numeric reduction alone is not sufficient.