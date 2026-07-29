# HI-CSG-R Adapter Input Audit v1

Overall status: **PASS**

| check | status | detail |
|---|---|---|
| `manifest_sample_counts` | **PASS** | {"expected": [39998, 6000, 5563], "actual": [39998, 6000, 5563]} |
| `missing_images` | **PASS** | {"train": 0, "val": 0, "test": 0} |
| `missing_graphs` | **PASS** | {"train": 0, "val": 0, "test": 0} |
| `graph_sample_id_mismatch` | **PASS** | {"train": 0, "val": 0, "test": 0} |
| `duplicate_sample_id` | **PASS** | {"train": 0, "val": 0, "test": 0} |
| `train_val_test_sample_overlap` | **PASS** | {"train_val": 0, "train_test": 0, "val_test": 0} |
| `train_val_test_path_overlap` | **PASS** | {"train_val": 0, "train_test": 0, "val_test": 0} |
| `vocab_mismatch` | **PASS** | {"expected": "a5914689766f1923c1b7538b73c87dc3efd3f1b08454e60a5b6875120c18710d", "checkpoint_vocab_hashes": ["a5914689766f1923c1b7538b73c87dc3efd3f1b08454e60a5b6875120c18710d", ... |
| `checkpoint_seed_mismatch` | **PASS** | {"42": 42, "43": 43, "44": 44} |
| `checkpoint_metadata` | **PASS** | {"42": {"epoch": 73, "has_model_state": true, "has_config": true}, "43": {"epoch": 76, "has_model_state": true, "has_config": true}, "44": {"epoch": 79, "has_model_state": true,... |

## Manifests

| split | n | missing images | duplicate ids | graph sources |
|---|---:|---:|---:|---|
| `train` | 39998 | 0 | 0 | `{"rebuildable_current_image": 39998}` |
| `val` | 6000 | 0 | 0 | `{"rebuildable_current_image": 6000}` |
| `test` | 5563 | 0 | 0 | `{"rebuildable_current_image": 5563}` |

## Checkpoints

| seed | epoch | seed match | model state | SHA256 |
|---:|---:|---:|---:|---|
| 42 | 73 | True | True | `dab5192fee3a0b575401fa15d3f66826beb19438dba870670b3f3369bedc66bf` |
| 43 | 76 | True | True | `d1c90f5f3fad307d9a54339eb239fc7f9ed0fb4b0484da6021602cf00c57188a` |
| 44 | 79 | True | True | `3be0676d169d99b1cb817b99aa8f0a8ee43b35881a03770e39a4244df3fb8410` |
