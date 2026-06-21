# Graph fusion v3 school foreground comparison

## 1. Overall

| model | CER | WER | exact | ΔCER vs image-only | ΔCER vs graph-v2 |
|---|---:|---:|---:|---:|---:|
| `image_only` | 0.08224 | 0.33502 | 0.62448 | 0.00000 | n/a |
| `graph_v2_old_features` | 0.13970 | 0.49042 | 0.43897 | +0.05746 | 0.00000 |
| `graph_v3_school_fg_auto` | 0.15338 | 0.52190 | 0.40823 | +0.07114 | +0.01368 |

## 2. By dataset

| dataset | image CER | graph-v2 CER | graph-v3 CER | v3-v2 CER | v3-image CER |
|---|---:|---:|---:|---:|---:|
| `cyrillic_handwriting` | n/a | 0.19968 | 0.21758 | +0.01790 | n/a |
| `cyrillic_handwriting|phrase|None` | 0.14451 | n/a | n/a | n/a | n/a |
| `cyrillic_handwriting|word|None` | 0.11016 | n/a | n/a | n/a | n/a |
| `hkr_words` | n/a | 0.08803 | 0.10185 | +0.01382 | n/a |
| `hkr_words|phrase|None` | 0.04864 | n/a | n/a | n/a | n/a |
| `hkr_words|word|None` | 0.06611 | n/a | n/a | n/a | n/a |
| `school_notebooks_clean` | n/a | 0.15635 | 0.16445 | +0.00810 | n/a |
| `school_notebooks_clean|phrase|pupil_text` | 0.13963 | n/a | n/a | n/a | n/a |
| `school_notebooks_clean|word|pupil_comment` | 0.17294 | n/a | n/a | n/a | n/a |
| `school_notebooks_clean|word|pupil_text` | 0.07100 | n/a | n/a | n/a | n/a |
| `school_notebooks_clean|word|teacher_comment` | 0.06101 | n/a | n/a | n/a | n/a |

## 3. Strict interpretation

Graph fusion v3 does not improve over graph fusion v2 in absolute CER.
Graph fusion v3 still does not beat image-only in absolute CER.

Even if v3 improves graph-fusion CER, this should be interpreted as a controlled preprocessing-pipeline improvement, not as evidence that graph fusion architecture itself is solved.