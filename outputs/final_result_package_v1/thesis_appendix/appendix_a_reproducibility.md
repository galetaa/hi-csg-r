# Appendix A. Reproducibility

- repo commit hash at package generation: `2c1ee75`
- final package root: `outputs/final_result_package_v1`
- primary test manifest: `data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/test.jsonl`
- canonical +10k test manifest: `data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/test.jsonl`
- baseline checkpoints: `outputs/htr_graph_v1/tri10k_image_only_v1* / best.pt`
- +10k checkpoints: `outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1* / best.pt`
- seeds: 42, 43, 44
- blank penalty protocol: final test evaluation uses validation-selected/evaluation-fixed penalty; seed42 provenance confirms `blank_logit_penalty=-0.4`.
- final scripts: `tools/build_results_inventory_v1.py`, `tools/check_seed42_provenance_v1.py`, `tools/build_seed_confirmation_report_v1.py`, `tools/build_domainwise_seed_confirmation_v1.py`, `tools/check_selective_prediction_canonical_v1.py`, `tools/write_selective_leakage_clearance_v1.py`.
