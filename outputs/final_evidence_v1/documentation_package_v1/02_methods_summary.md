# 02 — Methods summary
    
    1. Representation
    
    The project uses canonical visible-stroke graphs as an intermediate representation for offline handwriting images.
    
    Important distinction:
    
    The graph is not a reconstruction of real pen trajectory.
    It is a canonical graph of visible stroke structure extracted from a static image.
    
    The intended role is interpretability, structural diagnostics, robustness analysis, and failure triage.
    
    2. Recognition models
    
    The experiments compare:
    
    image-only CRNN/CTC baseline;
    graph-vector fusion model;
    gated local graph/dist-map fusion model.
    
    The image-only model is the main absolute-CER baseline. Graph-aware models are evaluated as structural/robustness variants, not as guaranteed accuracy improvements.
    
    3. Robustness evaluation
    
    Robustness is evaluated by comparing clean CER with CER under visual/structural distortions.
    
    Key interpretation rule:
    
    Lower relative degradation is not enough to claim better HTR if clean and distorted absolute CER are worse.
    
    Therefore H1 is evaluated in both absolute and relative terms.
    
    4. Graph diagnostics
    
    H3 evaluates whether graph-derived structural descriptors help identify high-error samples.
    
    Two levels are distinguished:
    
    global single-feature correlations;
    stratified multifeature high-error detection.
    
    The second is more meaningful in the current results.
    
    5. Manual H2 audit
    
    The H2 audit uses a diagnostic candidate pool selected across CER/risk quadrants:
    
    A: high CER + high structural risk
    B: high CER + low structural risk
    C: low CER + high structural risk
    D: low CER + low structural risk
    
    This is not a random population sample. It is used for failure-mode characterization and structural sanity checking.
    
    6. Failure staging
    
    Manual audit distinguishes:
    
    ok
    input_crop
    binarization
    skeletonization
    graph_topology
    illegible
    
    This prevents upstream preprocessing artifacts from being misreported as graph-topology failures.
    