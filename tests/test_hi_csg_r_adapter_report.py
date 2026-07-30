from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from tools.make_hi_csg_r_adapter_final_report_v1 import negative_validation_report


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_negative_validation_report_uses_only_validation_artifacts(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate.json"
    m0 = tmp_path / "m0.json"
    m3 = tmp_path / "m3.json"
    shuffled = tmp_path / "shuffled.json"
    m0_domains = tmp_path / "m0_domains.json"
    m3_domains = tmp_path / "m3_domains.json"
    history = tmp_path / "history.jsonl"
    output = tmp_path / "report"

    write_json(
        gate,
        {
            "status": "STOP",
            "relative_improvement": -0.03,
            "conditions": {"relative_improvement_2pct": False},
        },
    )
    base_summary = {
        "cer": 0.08,
        "wer": 0.3,
        "exact": 0.6,
        "macro_cer": 0.09,
    }
    write_json(m0, base_summary)
    write_json(
        m3,
        {
            **base_summary,
            "cer": 0.083,
            "gate": {"std": 0.02},
            "graph_aux_cer": 0.9,
        },
    )
    write_json(shuffled, {**base_summary, "cer": 0.084})
    write_json(
        m0_domains,
        {"domain": {"cer": 0.08, "exact": 0.6}},
    )
    write_json(
        m3_domains,
        {"domain": {"cer": 0.083, "exact": 0.58}},
    )
    history.write_text(
        json.dumps(
            {
                "stage": "joint",
                "graph_adapter_grad_norm": 2.0,
                "gate_grad_norm": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    negative_validation_report(
        Namespace(
            validation_gate=str(gate),
            m0_summary=str(m0),
            m3_summary=str(m3),
            shuffle_summary=str(shuffled),
            m0_domain_summary=str(m0_domains),
            m3_domain_summary=str(m3_domains),
            m3_history=str(history),
            figure_a=None,
            out_dir=str(output),
        )
    )

    report = json.loads((output / "final_report.json").read_text(encoding="utf-8"))
    assert report["hypothesis_h4"] == "exploratory"
    assert report["test_evaluated"] is False
    assert report["status"] == "complete_negative_result"
    assert (output / "final_report.md").exists()
    assert (output / "validation_overall_table.csv").exists()
    assert (output / "validation_domain_table.csv").exists()
    assert (output / "figure_b_seed42_validation.png").exists()
    assert (output / "method_and_results_sections_ru.md").exists()
