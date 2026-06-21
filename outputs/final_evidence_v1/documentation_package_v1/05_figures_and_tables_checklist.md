# 05 — Figures and tables checklist
    
    Required figures
    Figure 1 — Pipeline diagram
    
    Show:
    
    image → foreground mask → skeleton → canonical graph descriptors → HTR / diagnostics
    
    Caption must say:
    
    The graph is a canonical visible-stroke structure, not a real pen trajectory reconstruction.
    Figure 2 — H1 robustness comparison
    
    Use a bar/table plot comparing:
    
    clean CER
    mean distorted CER
    relative degradation
    
    Models:
    
    image-only
    graph-vector
    gated dist
    
    Main message:
    
    graph-aware models have lower relative degradation but worse absolute CER.
    Figure 3 — H2 good examples
    
    Show HKR/Cyrillic examples with:
    
    original
    binary
    skeleton
    overlay
    
    Main message:
    
    visible stroke structure is mostly preserved in audited HKR/Cyrillic samples.
    Figure 4 — H2 school-notebooks failure
    
    Show school-notebooks crop/border artifact:
    
    original
    binary with border artifact
    bad skeleton
    overlay
    
    Main message:
    
    failure is upstream crop/binarization, not pure graph topology.
    Figure 5 — H3 diagnostic signal
    
    Show the best multifeature high-error detection result:
    
    structural_core
    hkr_words|word
    ROC-AUC / PR-AUC / top20 precision
    
    Main message:
    
    localized diagnostic value, not global graph-quality scoring.
    Figure 6 — Failure taxonomy
    
    Show staged failure types:
    
    input_crop
    binarization
    skeletonization
    graph_topology
    recognition difficulty
    Required tables
    Hypothesis verdict table.
    H1 robustness table.
    H2 audit summary table.
    H3 diagnostic result table.
    Safe/unsafe claims table.
    Figures to avoid
    
    Avoid figures that imply:
    
    graph-aware model is the best recognizer;
    structural risk is graph quality;
    school-notebooks failures are pure graph-topology failures;
    offline graph equals real pen trajectory.
    