# 08 — Defense / reviewer Q&A
    
    Q1. Did graph-aware recognition beat the image-only baseline?
    
    No. The image-only baseline has better absolute CER. Graph-aware variants show lower relative degradation under distortions, but they are worse in clean and distorted absolute CER.
    
    Q2. Is H1 confirmed?
    
    Only weakly. The strong form is not confirmed. The supported claim is lower relative degradation, not better HTR performance.
    
    Q3. Is the graph a pen-trajectory reconstruction?
    
    No. It is a canonical visible-stroke graph extracted from offline images. It represents visible stroke structure, not true writing dynamics.
    
    Q4. Does high structural risk mean bad graph quality?
    
    No. Manual audit shows that structural risk often marks difficult samples rather than visible skeleton failure. It is a hard-sample indicator, not a graph-quality score.
    
    Q5. Why are school-notebooks so bad?
    
    Because crop/border/background artifacts are binarized as foreground. This corrupts skeletons and graphs upstream of graph construction.
    
    Q6. Does school-notebooks invalidate the graph representation?
    
    No. It reveals a preprocessing limitation. HKR/Cyrillic audited samples show much better structural preservation.
    
    Q7. Why not fix school-notebooks immediately?
    
    A simple border-suppression rule was tested and rejected because it either did nothing or removed handwriting. A proper fix requires dataset-specific preprocessing, not quick graph tuning.
    
    Q8. What is the main contribution?
    
    The main contribution is a reproducible visible-stroke graph diagnostic framework for offline handwriting recognition, with evidence for robustness analysis, failure triage, and preprocessing failure detection.
    
    Q9. What remains future work?
    
    Better preprocessing for notebook data, larger gold graph annotation, calibrated graph-quality prediction, and only then renewed graph-aware recognition experiments.
    