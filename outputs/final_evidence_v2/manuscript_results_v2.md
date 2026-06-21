# Results

## Recognition baselines and graph fusion

The image-only recognizer remained the strongest model in absolute recognition quality. The retained graph-vector checkpoint achieved a clean CER of 0.1397, while replacing its School Notebooks graph features with the repaired foreground-v3 features changed CER only slightly to 0.1394. A newly trained graph-fusion checkpoint reached a worse CER of 0.1534. Cross-evaluation showed that this degradation persisted with both old and repaired manifests, so it was not caused by foreground v3.

## H1: robustness

Across the 5,563 clean test samples and 15 distortion conditions, the paired corpus-level analysis yielded an image-only relative CER degradation of 33.77% and a graph-model degradation of 21.72%. The resulting relative robustness advantage was 12.05%, with a 95% paired cluster-bootstrap interval of 9.37%–14.81% and a one-sided permutation p-value of 0.000050.

The absolute degradation advantage was -0.00333, with a 95% interval of -0.00528–-0.00137. Furthermore, distorted CER remained higher for the graph model by 0.06297. Thus, the graph model was relatively less sensitive to distortion but remained the worse recognizer in absolute terms.

## H2: visible-structure preservation

In the original diagnostic audit, the combined HKR and Cyrillic subset contained 77 samples. The critical-topology-error rate was 2.60%, while the skeleton followed visible ink in 96.10% of samples. The initial School Notebooks failure was traced to foreground extraction rather than to graph construction.

An independent random validation on 100 School Notebooks test samples selected `school_dark_auto` in all inspected cases. The raw good-fix rate was 92.00%, while the strict usable rate was 89.00%. Real-ink loss was observed in 4.00% of samples, residual background artifacts in 7.00%, and the resulting skeleton followed visible ink in 96.00%.

## H3: diagnostic value

The strongest multifeature diagnostic result used `structural_core` in the subgroup `hkr_words|word|unknown` with n=1090. It achieved ROC-AUC 0.6723, PR-AUC 0.3532, and top-20% precision 0.3853. The signal was therefore useful but localized rather than global.
