# 04 — Limitations and threats to validity
    
    1. Recognition performance
    
    Graph-aware models do not outperform the image-only baseline in absolute CER. Any claim of improved recognition accuracy would be unsupported.
    
    2. Relative robustness
    
    Lower relative degradation under distortions is meaningful only as a sensitivity signal. Because graph-aware models start with worse clean CER, relative degradation cannot be interpreted as superior HTR robustness by itself.
    
    3. Diagnostic audit sampling
    
    The H2 audit subset is deliberately selected across CER/risk quadrants. It is not a random sample and should not be used to estimate population-level graph quality.
    
    4. School-notebooks preprocessing
    
    School-notebooks failures are dominated by crop/border/binarization artifacts. These failures occur before graph construction. They should be reported separately from graph-topology failures.
    
    5. Structural risk
    
    The structural risk score is useful as a hard-sample indicator, but manual audit shows it is not equivalent to visible graph quality.
    
    6. Gold graph annotation
    
    The project does not yet include a large independent gold graph annotation set. Therefore H2 is supported through diagnostic manual audit rather than full population-level graph accuracy estimation.
    
    7. Dataset specificity
    
    Evidence is strongest for the audited HKR and Cyrillic samples. Generalization to other handwriting sources, scanning conditions, crop procedures, and background types remains limited.
    
    8. Model search
    
    Architecture experiments should be frozen. Additional model chasing risks obscuring the main methodological contribution.
    