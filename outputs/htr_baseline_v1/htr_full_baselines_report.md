# HTR full baselines report — Stage 3

## 1. Setup

```text
model: height-preserving CRNN + BiLSTM + CTC
input: OCR-preprocessed grayscale images
decode: greedy CTC, blank penalty selected on validation
status: image-only HTR baseline stage completed
```

## 2. Overall metrics

| dataset | split | n | CER | WER | exact | pred_len | empty | blank | penalty | epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IAM full v1 | train | 11134 | 0.0007 | 0.0037 | 0.9719 | 43.22 | 0.000 | 0.685 | -0.500 | 76 |
| IAM full v1 | val | 1180 | 0.0670 | 0.2277 | 0.2508 | 42.71 | 0.000 | 0.673 | -0.500 | 76 |
| IAM full v1 | test | 1039 | 0.0637 | 0.2202 | 0.2685 | 43.64 | 0.000 | 0.680 | -0.500 | 76 |
| Cyrillic Handwriting full v1 | train | 65031 | 0.0037 | 0.0241 | 0.9729 | 7.47 | 0.000 | 0.821 | -0.400 | 63 |
| Cyrillic Handwriting full v1 | val | 7232 | 0.0809 | 0.3268 | 0.6645 | 7.41 | 0.000 | 0.825 | -0.400 | 63 |
| Cyrillic Handwriting full v1 | test | 1563 | 0.1405 | 0.5470 | 0.3986 | 9.43 | 0.000 | 0.831 | -0.400 | 63 |
| HKR Words full v1 | train | 51954 | 0.0029 | 0.0177 | 0.9714 | 11.17 | 0.000 | 0.806 | -0.600 | 72 |
| HKR Words full v1 | val | 6494 | 0.1311 | 0.4642 | 0.4315 | 10.15 | 0.000 | 0.828 | -0.600 | 72 |
| HKR Words full v1 | test | 6495 | 0.1525 | 0.5186 | 0.3718 | 10.27 | 0.000 | 0.835 | -0.600 | 72 |
| School Notebooks full v1 | train | 234833 | 0.0061 | 0.0372 | 0.9630 | 5.86 | 0.000 | 0.831 | -0.200 | 47 |
| School Notebooks full v1 | val | 24214 | 0.0492 | 0.1947 | 0.8094 | 5.92 | 0.000 | 0.845 | -0.200 | 47 |
| School Notebooks full v1 | test | 24623 | 0.0838 | 0.3060 | 0.6998 | 5.83 | 0.000 | 0.829 | -0.200 | 47 |

## 3. Test-set comparison

| dataset | level | language | test n | test CER | test WER | test exact |
|---|---|---|---:|---:|---:|---:|
| IAM | line | en | 1039 | 0.0637 | 0.2202 | 0.2685 |
| Cyrillic Handwriting | word/phrase | ru | 1563 | 0.1405 | 0.5470 | 0.3986 |
| HKR Words | word/phrase | ru_kk | 6495 | 0.1525 | 0.5186 | 0.3718 |
| School Notebooks Clean | word/phrase | ru | 24623 | 0.0838 | 0.3060 | 0.6998 |

## 4. School Notebooks category breakdown

### train
| group | n | CER | WER | exact |
|---|---:|---:|---:|---:|
| `school_notebooks_clean|phrase|pupil_comment` | 1 | 0.0000 | 0.0000 | 1.0000 |
| `school_notebooks_clean|phrase|pupil_text` | 3344 | 0.0270 | 0.1705 | 0.8065 |
| `school_notebooks_clean|phrase|teacher_comment` | 86 | 0.0422 | 0.2035 | 0.7674 |
| `school_notebooks_clean|word|pupil_comment` | 9900 | 0.0110 | 0.0388 | 0.9612 |
| `school_notebooks_clean|word|pupil_text` | 217278 | 0.0056 | 0.0353 | 0.9652 |
| `school_notebooks_clean|word|teacher_comment` | 4224 | 0.0049 | 0.0215 | 0.9789 |

### val
| group | n | CER | WER | exact |
|---|---:|---:|---:|---:|
| `school_notebooks_clean|phrase|pupil_text` | 308 | 0.1695 | 0.9632 | 0.0325 |
| `school_notebooks_clean|phrase|teacher_comment` | 6 | 0.4597 | 1.0000 | 0.0000 |
| `school_notebooks_clean|word|pupil_comment` | 928 | 0.1615 | 0.3448 | 0.6638 |
| `school_notebooks_clean|word|pupil_text` | 22396 | 0.0416 | 0.1767 | 0.8272 |
| `school_notebooks_clean|word|teacher_comment` | 576 | 0.0970 | 0.2344 | 0.7743 |

### test
| group | n | CER | WER | exact |
|---|---:|---:|---:|---:|
| `school_notebooks_clean|phrase|pupil_text` | 144 | 0.1613 | 0.7778 | 0.1736 |
| `school_notebooks_clean|phrase|teacher_comment` | 1 | 0.4000 | 1.0000 | 0.0000 |
| `school_notebooks_clean|word|pupil_comment` | 1224 | 0.1642 | 0.3595 | 0.6454 |
| `school_notebooks_clean|word|pupil_text` | 22818 | 0.0793 | 0.3024 | 0.7038 |
| `school_notebooks_clean|word|teacher_comment` | 436 | 0.0682 | 0.1858 | 0.8142 |

## 5. Interpretation

IAM gives the lowest CER, but it is line-level English and should not be directly compared with word-level Russian crop datasets by exact match.

Cyrillic Handwriting and HKR Words both improve strongly from mini10k to full training. HKR remains methodologically important because its split is text-grouped.

School Notebooks full is the strongest Russian crop baseline by validation CER. The main `word|pupil_text` category is substantially easier than `word|pupil_comment`; phrase groups should be treated as secondary because of smaller sample counts.

## 6. Stage 3 conclusion

```text
[x] IAM full baseline
[x] Cyrillic Handwriting full baseline
[x] HKR Words full baseline
[x] School Notebooks full baseline
[x] blank-collapse resolved
[x] image-only HTR baseline stage completed
```

## 7. Next recommended stage

Next: build mixed-dataset Cyrillic baselines, then add graph-aware features and compare image-only versus image+graph models.
