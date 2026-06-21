# 07 — Improvement roadmap
    
    Principle
    
    Do not return to architecture chasing. Improvements should target failure modes revealed by evidence.
    
    Priority 1 — School-notebooks preprocessing
    
    Problem:
    
    crop/border/background artifacts are binarized as foreground
    
    Goal:
    
    improve foreground extraction before skeletonization
    
    Allowed experiments:
    
    crop-border masking before binarization;
    background normalization;
    border-line detection;
    notebook-specific foreground cleanup;
    regenerated skeleton/graph quality audit on the same 23 samples.
    
    Do not retrain HTR for this step.
    
    Priority 2 — H2 gold subset expansion
    
    Goal:
    
    move from diagnostic audit to a more reliable gold subset
    
    Possible work:
    
    sample random HKR/Cyrillic cases;
    manually annotate graph quality;
    estimate population-level graph preservation with confidence intervals.
    Priority 3 — Better graph-quality score
    
    Problem:
    
    structural risk is not graph quality
    
    Goal:
    
    train/calibrate a graph-quality predictor using manual labels
    
    Use manual H2 labels as supervision.
    
    Priority 4 — Robustness follow-up
    
    Only after preprocessing is fixed:
    
    regenerate clean/distorted graph features;
    rerun H1 aggregation;
    check whether graph-aware robustness remains.
    Priority 5 — Model improvement
    
    Only if previous steps succeed:
    
    freeze image encoder baseline;
    test graph input as auxiliary diagnostic head;
    avoid claiming accuracy improvement unless absolute CER improves.
    