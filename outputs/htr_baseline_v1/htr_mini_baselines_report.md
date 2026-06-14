# HTR mini-baselines report — Stage 3

## 1. Purpose

This report summarizes the first image-only HTR baselines for the HI-CSG-R project. All runs use OCR-preprocessed grayscale images and a height-preserving CRNN + BiLSTM + CTC model. Decode blank penalties are selected on validation splits and then applied consistently to train/val/test evaluation.

## 2. Overall metrics

| dataset | split | n | CER | WER | exact | pred_len | empty | blank | penalty | epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IAM mini10k v1 | train | 10000 | 0.0011 | 0.0059 | 0.9573 | 43.20 | 0.000 | 0.708 | -0.300 | 71 |
| IAM mini10k v1 | val | 1180 | 0.0695 | 0.2321 | 0.2051 | 42.60 | 0.000 | 0.686 | -0.300 | 71 |
| IAM mini10k v1 | test | 1039 | 0.0666 | 0.2236 | 0.2551 | 43.50 | 0.000 | 0.695 | -0.300 | 71 |
| Cyrillic Handwriting mini10k v1 | train | 10000 | 0.0005 | 0.0022 | 0.9975 | 7.41 | 0.000 | 0.822 | -0.900 | 72 |
| Cyrillic Handwriting mini10k v1 | val | 2000 | 0.1602 | 0.5506 | 0.4390 | 7.33 | 0.000 | 0.828 | -0.900 | 72 |
| Cyrillic Handwriting mini10k v1 | test | 1563 | 0.2556 | 0.7890 | 0.1708 | 9.28 | 0.000 | 0.840 | -0.900 | 72 |
| HKR Words mini10k v1 | train | 10000 | 0.0137 | 0.0838 | 0.8653 | 11.13 | 0.000 | 0.826 | -1.210 | 40 |
| HKR Words mini10k v1 | val | 2000 | 0.2099 | 0.6405 | 0.2345 | 9.97 | 0.000 | 0.837 | -1.210 | 40 |
| HKR Words mini10k v1 | test | 2000 | 0.2139 | 0.6804 | 0.1975 | 10.25 | 0.000 | 0.842 | -1.210 | 40 |
| School Notebooks mini10k v1 | train | 10000 | 0.0017 | 0.0108 | 0.9891 | 5.87 | 0.000 | 0.814 | -1.000 | 74 |
| School Notebooks mini10k v1 | val | 2000 | 0.1517 | 0.4795 | 0.5240 | 5.79 | 0.000 | 0.830 | -1.000 | 74 |
| School Notebooks mini10k v1 | test | 2000 | 0.2237 | 0.6195 | 0.3875 | 5.66 | 0.000 | 0.823 | -1.000 | 74 |

## 3. Test-set comparison

| dataset | level | language | test n | test CER | test WER | test exact |
|---|---|---|---:|---:|---:|---:|
| IAM | line | en | 1039 | 0.0666 | 0.2236 | 0.2551 |
| Cyrillic Handwriting | word/phrase | ru | 1563 | 0.2556 | 0.7890 | 0.1708 |
| HKR Words | word/phrase | ru_kk | 2000 | 0.2139 | 0.6804 | 0.1975 |
| School Notebooks Clean | word/phrase | ru | 2000 | 0.2237 | 0.6195 | 0.3875 |

## 4. School Notebooks category breakdown

### train
| group | n | CER | WER | exact |
|---|---:|---:|---:|---:|
| `school_notebooks_clean|phrase|pupil_text` | 112 | 0.0025 | 0.0089 | 0.9821 |
| `school_notebooks_clean|phrase|teacher_comment` | 4 | 0.0000 | 0.0000 | 1.0000 |
| `school_notebooks_clean|word|pupil_comment` | 445 | 0.0019 | 0.0067 | 0.9933 |
| `school_notebooks_clean|word|pupil_text` | 9261 | 0.0017 | 0.0112 | 0.9888 |
| `school_notebooks_clean|word|teacher_comment` | 178 | 0.0000 | 0.0000 | 1.0000 |

### val
| group | n | CER | WER | exact |
|---|---:|---:|---:|---:|
| `school_notebooks_clean|phrase|pupil_text` | 29 | 0.2687 | 0.9655 | 0.0345 |
| `school_notebooks_clean|word|pupil_comment` | 84 | 0.3573 | 0.6667 | 0.3333 |
| `school_notebooks_clean|word|pupil_text` | 1839 | 0.1372 | 0.4628 | 0.5411 |
| `school_notebooks_clean|word|teacher_comment` | 48 | 0.2770 | 0.5000 | 0.5000 |

### test
| group | n | CER | WER | exact |
|---|---:|---:|---:|---:|
| `school_notebooks_clean|phrase|pupil_text` | 13 | 0.3098 | 1.0000 | 0.0000 |
| `school_notebooks_clean|word|pupil_comment` | 112 | 0.3186 | 0.5804 | 0.4196 |
| `school_notebooks_clean|word|pupil_text` | 1838 | 0.2171 | 0.6208 | 0.3868 |
| `school_notebooks_clean|word|teacher_comment` | 37 | 0.2345 | 0.5405 | 0.4595 |

## 5. Interpretation

### 5.1 IAM

IAM achieves the lowest CER among the mini-baselines. This is not directly comparable to the Russian crop datasets, because IAM is English line-level recognition with longer targets. Exact match is naturally lower for line-level targets.

### 5.2 Cyrillic Handwriting

Cyrillic Handwriting shows strong train memorization and moderate test performance. The test split is relatively small and appears harder than validation.

### 5.3 HKR Words

HKR Words has stable validation and test CER. This is methodologically important because the split is text-grouped, so target-text leakage is controlled.

### 5.4 School Notebooks

School Notebooks performs strongly on the main pupil_text word category. pupil_comment and teacher_comment are harder and have smaller sample counts, so they should be reported separately.

## 6. Stage 3 status

```text
[x] HTR manifests created
[x] CTC-ready manifests created
[x] One-sample overfit passed
[x] Tiny64 overfit passed
[x] Cyrillic mini10k baseline passed
[x] HKR Words mini10k baseline passed
[x] School Notebooks mini10k baseline passed
[x] IAM mini10k baseline passed
```

## 7. Next step

The next recommended step is full single-dataset baselines, starting with IAM and Cyrillic Handwriting, then HKR Words, then School Notebooks. After full image-only baselines are established, graph-aware experiments can begin.
