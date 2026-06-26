# Seed42 provenance check

Verdict: **PASS**

Seed42 combined summary is reproducible enough for primary 3-seed reporting.

## Checks

| check | status | details |
|---|---|---|
| summary_exists | PASS | `{"path": "outputs/htr_graph_v1/eval_tri10k_image_only_v1_test_final/summary.json"}` |
| manifest_exists | PASS | `{"path": "data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/test.jsonl", "manifest_n": 5563}` |
| checkpoint_exists | PASS | `{"path": "outputs/htr_graph_v1/tri10k_image_only_v1/best.pt", "size_bytes": 46847353}` |
| summary_n_expected | PASS | `{"summary_n": 5563}` |
| manifest_n_expected | PASS | `{"manifest_n": 5563}` |
| manifest_n_matches_summary_n | PASS | `{"n": 5563}` |
| blank_logit_penalty_expected | PASS | `{"blank_logit_penalty": -0.4}` |
| predictions_recomputed | PASS | `{"path": "outputs/htr_graph_v1/eval_tri10k_image_only_v1_test_final/predictions.jsonl", "n": 5563.0, "valid_n": 5563.0, "cer": 0.1454455104383119, "wer": 0.5128377973515494, "exact": 0.44094912816825454}` |
| cer_matches_recomputed | PASS | `{"summary": 0.1454455104383119, "recomputed": 0.1454455104383119, "tol": 1e-08}` |
| wer_matches_recomputed | PASS | `{"summary": 0.5128377973515494, "recomputed": 0.5128377973515494, "tol": 1e-08}` |
| exact_matches_recomputed | PASS | `{"summary": 0.44094912816825454, "recomputed": 0.44094912816825454, "tol": 1e-08}` |