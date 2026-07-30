# HI-CSG-R adapter v2 split audit

**Status:** PASS

## Counts

| Split | Total | Cyrillic | HKR | School |
|---|---:|---:|---:|---:|
| train | 35498 | 8500 | 8500 | 18498 |
| dev | 3000 | 1000 | 1000 | 1000 |
| holdout | 1500 | 500 | 500 | 500 |

## Leakage checks

- sample overlap: `{'train_dev': 0, 'train_holdout': 0, 'dev_holdout': 0}`
- path overlap: `{'train_dev': 0, 'train_holdout': 0, 'dev_holdout': 0}`
- group overlap: `{'train_dev': 0, 'train_holdout': 0, 'dev_holdout': 0}`
- SHA1 overlap: `{'train_dev': 0, 'train_holdout': 0, 'dev_holdout': 0}`
- missing images: `{'train': 0, 'dev': 0, 'holdout': 0}`
- missing feature records: `{'train': 0, 'dev': 0, 'holdout': 0}`
- failures: `[]`
