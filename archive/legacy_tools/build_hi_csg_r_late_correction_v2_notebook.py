from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="notebooks/htr_adapter_v2_late_correction_full_experiment.ipynb",
    )
    args = parser.parse_args()
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        markdown(
            """
# HI-CSG-R Late Correction v2: полный эксперимент

Этот notebook исполняет frozen protocol v2 по стадиям. Он сохраняет v1,
не открывает holdout до выбора dev-кандидата и не открывает test до положительного
holdout gate. `STOP` является научным результатом, а не ошибкой notebook.
"""
        ),
        code(
            """
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display

start = Path.cwd().resolve()
ROOT = next(
    (
        candidate
        for candidate in (start, *start.parents)
        if (candidate / "src").is_dir() and (candidate / "tools").is_dir()
    ),
    None,
)
if ROOT is None:
    raise RuntimeError(f"Корень репозитория не найден из cwd={start}")
PYTHON = str(ROOT / ".venv/bin/python")
OUT = ROOT / "outputs/htr_adapter_v2"
DATA = ROOT / "data/experiments/htr_adapter_v2"
LOGS = OUT / "notebook/logs"
LOGS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def show_json(path, title=None):
    value = load_json(path)
    display(Markdown(f"### {title or Path(path).name}"))
    display(value)
    return value

def module(name, *args, log_name=None, allowed=(0,)):
    command = [PYTHON, "-m", name, *map(str, args)]
    label = log_name or name.rsplit(".", 1)[-1]
    log_path = LOGS / f"{label}.stdout.log"
    display(Markdown(f"**RUN `{label}`**\\n\\n`{' '.join(command)}`"))
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    state = "PASS" if returncode == 0 else "STOP" if returncode == 2 else "FAIL"
    display(Markdown(f"**{state} `{label}`**; log: `{log_path}`"))
    if returncode not in allowed:
        raise RuntimeError(f"{label}: exit={returncode}; log={log_path}")
    return returncode

def resume_args(run_dir):
    run_dir = Path(run_dir)
    return (
        ("--resume", run_dir / "last.pt")
        if (run_dir / "last.pt").exists() and not (run_dir / "train_summary.json").exists()
        else ()
    )

display(Markdown(f"**Workspace:** `{ROOT}`  \\n**Python:** `{PYTHON}`"))
"""
        ),
        markdown("## 1. Frozen protocol и неизменность v1"),
        code(
            """
display(Markdown((ROOT / "docs/crnn_ctc_hi_csg_r_late_correction_protocol_v2.md").read_text()))
show_json(OUT / "protocol_freeze/protocol_freeze.json", "Protocol freeze")
display(Markdown(
    "V1 остается отдельным отрицательным результатом: "
    "`outputs/htr_adapter_v1/final_report/full_execution_report_ru.md`."
))
"""
        ),
        markdown("## 2. Tests и статическая проверка"),
        code(
            """
module("pytest", "-q", *sorted(str(path) for path in (ROOT / "tests").glob("test_htr_adapter_v2_*.py")), log_name="pytest_v2")
v2_tools = sorted({
    str(path)
    for pattern in (
        "*hi_csg_r*adapter*v2*.py",
        "*hi_csg_r*late_correction*v2*.py",
        "train_crnn_ctc_adapter_v2_baseline.py",
        "prepare_hi_csg_r_adapter_v2_features.py",
    )
    for path in (ROOT / "tools").glob(pattern)
})
module(
    "ruff", "check",
    "src/htr/adapter_runtime_v2.py",
    "src/htr/dataset_adapter_v2.py",
    "src/htr/losses_adapter_v2.py",
    "src/htr/masked_pooling.py",
    "src/htr/model_hi_csg_r_late_correction_v2.py",
    "src/htr/uncertainty.py",
    *v2_tools,
    log_name="ruff_v2",
)
"""
        ),
        markdown("## 3. Preflight D1-D3"),
        code(
            """
preflight_path = OUT / "preflight/preflight_report.json"
if not preflight_path.exists():
    module(
        "tools.diagnose_hi_csg_r_adapter_v1_for_v2",
        "--m0_checkpoint", "outputs/htr_adapter_v1/m0_ft_seed42/best.pt",
        "--m3_checkpoint", "outputs/htr_adapter_v1/m3_full_seed42/best.pt",
        "--manifest", "data/experiments/htr_adapter_v1/manifests/val.jsonl",
        "--shuffle_map", "data/experiments/htr_adapter_v1/shuffle_maps/val_seed42.json",
        "--vocab", "data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/vocab.json",
        "--normalizer", "data/experiments/htr_adapter_v1/normalizer/train_stats.json",
        "--out_dir", preflight_path.parent,
        allowed=(0, 2),
    )
preflight = show_json(preflight_path, "Preflight decision")
penalty_table = pd.read_csv(OUT / "preflight/blank_penalty_sweep.csv")
display(penalty_table)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for model, values in preflight["blank_penalty"].items():
    xs = sorted(float(value) for value in values)
    axes[0].plot(
        xs,
        [values[str(value)]["cer"] for value in xs],
        marker="o",
        label=model,
    )
axes[0].axvline(
    preflight["decision"]["selected_blank_logit_penalty"],
    color="black",
    linestyle="--",
    linewidth=1,
)
axes[0].set(xlabel="blank logit penalty", ylabel="Validation CER", title="D1 calibration")
axes[0].legend()
scale_values = preflight["graph_scale"]
scale_x = sorted(float(value) for value in scale_values)
axes[1].plot(
    scale_x,
    [scale_values[str(value)]["cer"] for value in scale_x],
    marker="o",
    color="#0f766e",
)
axes[1].set(xlabel="v1 graph residual scale", ylabel="Validation CER", title="D2 graph strength")
for axis in axes:
    axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## 4. Independent split и train-only statistics"),
        code(
            """
split_audit = OUT / "split_audit/split_audit.json"
if not split_audit.exists():
    module(
        "tools.create_hi_csg_r_adapter_v2_split",
        "--train_manifest", "data/experiments/htr_adapter_v1/manifests/train.jsonl",
        "--out_dir", DATA / "splits",
        "--features_out_dir", DATA / "features",
        "--dev_per_domain", 1000,
        "--holdout_per_domain", 500,
        "--seed", 20260730,
    )
    module(
        "tools.audit_hi_csg_r_adapter_v2_split",
        "--split_dir", DATA / "splits",
        "--out_dir", split_audit.parent,
    )
feature_audit = OUT / "feature_audit/feature_audit.json"
if not feature_audit.exists():
    module(
        "tools.prepare_hi_csg_r_adapter_v2_features",
        "--train_manifest", DATA / "features/train.jsonl",
        "--dev_manifest", DATA / "features/dev.jsonl",
        "--holdout_manifest", DATA / "features/holdout.jsonl",
        "--normalizer_out", DATA / "normalizer/train_stats.json",
        "--risk_stats_out", DATA / "normalizer/risk_stats.json",
        "--audit_out", feature_audit,
        log_name="prepare_v2_features",
    )
show_json(split_audit, "Split audit")
show_json(feature_audit, "Feature/normalizer audit")
"""
        ),
        markdown("## 5. Smoke/overfit gate"),
        code(
            """
smoke = show_json(OUT / "smoke/smoke_gate.json", "Smoke gate")
if smoke["status"] != "PASS":
    display(Markdown("**STOP:** full development запрещен до исправления программной ошибки."))
"""
        ),
        markdown("## 6. Fresh B0-dev-v2"),
        code(
            """
b0 = OUT / "b0_dev_seed42/train_summary.json"
if smoke["status"] == "PASS" and not b0.exists():
    module(
        "tools.train_crnn_ctc_adapter_v2_baseline",
        "train", "--config", "configs/htr_adapter_v2/b0_dev_seed42.yaml",
        *resume_args(OUT / "b0_dev_seed42"),
        log_name="b0_dev_seed42",
    )
if b0.exists():
    show_json(b0, "B0-dev-v2 training")
    b0_history = pd.DataFrame(load_json(OUT / "b0_dev_seed42/history.json"))
    display(b0_history.tail())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(b0_history["epoch"], b0_history["train_loss"], color="#2563eb")
    axes[0].set(title="B0 train loss", xlabel="epoch", ylabel="CTC loss")
    axes[1].plot(b0_history["epoch"], b0_history["val_CER"], label="CER")
    axes[1].plot(b0_history["epoch"], b0_history["val_exact"], label="Exact")
    axes[1].set(title="B0 development metrics", xlabel="epoch")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
else:
    display(Markdown("B0 не запущен из-за предыдущего STOP."))
"""
        ),
        markdown("## 7. V2-1 development"),
        code(
            """
v21 = OUT / "v2_1_dev_p05_seed42/train_summary.json"
if b0.exists() and not v21.exists():
    module(
        "tools.train_crnn_ctc_hi_csg_r_late_correction_v2",
        "train", "--config", "configs/htr_adapter_v2/v2_1_dev_p05_seed42.yaml",
        *resume_args(OUT / "v2_1_dev_p05_seed42"),
        log_name="v2_1_dev_p05",
    )
if v21.exists():
    show_json(v21, "V2-1 training")
    v21_history = pd.DataFrame(
        load_json(OUT / "v2_1_dev_p05_seed42/history.json")
    )
    display(v21_history.tail())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(v21_history["epoch"], v21_history["train_ctc_loss"], label="main CTC")
    axes[0, 0].plot(v21_history["epoch"], v21_history["train_auxiliary_ctc"], label="aux CTC")
    axes[0, 0].set_title("Training CTC objectives")
    axes[0, 0].legend()
    axes[0, 1].plot(v21_history["epoch"], v21_history["train_preservation_kl"])
    axes[0, 1].set_title("Baseline-preservation KL")
    axes[1, 0].plot(v21_history["epoch"], v21_history["val_CER"], label="CER")
    axes[1, 0].plot(v21_history["epoch"], v21_history["val_exact"], label="Exact")
    axes[1, 0].set_title("Development metrics")
    axes[1, 0].legend()
    axes[1, 1].plot(v21_history["epoch"], v21_history["val_alpha"], label="alpha")
    axes[1, 1].plot(
        v21_history["epoch"],
        [row["mean"] for row in v21_history["val_gate"]],
        label="gate mean",
    )
    axes[1, 1].set_title("Bounded intervention")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("epoch")
    plt.tight_layout()
    plt.show()
"""
        ),
        markdown("## 8. V2-1 correct/shuffle/zero и dev gate"),
        code(
            """
def evaluate_modes(run_dir, manifest, split_name):
    shuffle_map = DATA / f"shuffle_maps/{split_name}_seed42.json"
    if not shuffle_map.exists():
        module(
            "tools.build_hi_csg_r_adapter_v2_shuffle_map",
            "--manifest", manifest, "--seed", 42, "--out", shuffle_map,
            log_name=f"shuffle_{split_name}",
        )
    root = OUT / f"development/{run_dir.name}"
    for mode in ("correct", "shuffle", "zero"):
        destination = root / mode
        if not (destination / "summary.json").exists():
            extra = ("--shuffle_map", shuffle_map) if mode == "shuffle" else ()
            module(
                "tools.evaluate_crnn_ctc_hi_csg_r_late_correction_v2",
                "--checkpoint", run_dir / "best.pt",
                "--manifest", manifest,
                "--mode", mode,
                *extra,
                "--out_dir", destination,
                log_name=f"eval_{run_dir.name}_{mode}",
            )
    return root

v21_eval = None
v21_decision = None
if v21.exists():
    v21_eval = evaluate_modes(OUT / "v2_1_dev_p05_seed42", DATA / "features/dev.jsonl", "dev")
    decision_dir = OUT / "development/v2_1_dev_p05_seed42/decision"
    module(
        "tools.compare_hi_csg_r_adapter_v2_results",
        "--stage", "dev",
        "--correct", v21_eval / "correct",
        "--shuffle", v21_eval / "shuffle",
        "--zero", v21_eval / "zero",
        "--out_dir", decision_dir,
        log_name="v2_1_dev_gate",
        allowed=(0, 2),
    )
    v21_decision = show_json(decision_dir / "dev_decision.json", "V2-1 dev gate")
"""
        ),
        markdown("## 9. Условный V2-2 и максимум один p10 repeat"),
        code(
            """
allow_v22 = (
    preflight["decision"]["allow_v2_2"]
    and v21_decision is not None
    and not (
        v21_decision["relative_cer_improvement"] < -0.01
        and preflight["decision"]["d3_absolute_cer_gain"] < 0.0005
    )
)
v22 = OUT / "v2_2_dev_p05_seed42/train_summary.json"
if allow_v22 and not v22.exists():
    module(
        "tools.train_crnn_ctc_hi_csg_r_late_correction_v2",
        "train", "--config", "configs/htr_adapter_v2/v2_2_dev_p05_seed42.yaml",
        *resume_args(OUT / "v2_2_dev_p05_seed42"),
        log_name="v2_2_dev_p05",
    )
if not allow_v22:
    display(Markdown("V2-2 заблокирован frozen preflight/dev stopping rule."))
v22_eval = None
v22_decision = None
if v22.exists():
    v22_eval = evaluate_modes(
        OUT / "v2_2_dev_p05_seed42",
        DATA / "features/dev.jsonl",
        "dev",
    )
    v22_decision_dir = OUT / "development/v2_2_dev_p05_seed42/decision"
    if not (v22_decision_dir / "dev_decision.json").exists():
        module(
            "tools.compare_hi_csg_r_adapter_v2_results",
            "--stage", "dev",
            "--correct", v22_eval / "correct",
            "--shuffle", v22_eval / "shuffle",
            "--zero", v22_eval / "zero",
            "--out_dir", v22_decision_dir,
            log_name="v2_2_dev_gate",
            allowed=(0, 2),
        )
    v22_decision = show_json(
        v22_decision_dir / "dev_decision.json",
        "V2-2 dev gate",
    )

p05_selection_path = OUT / "development/p05_selection.json"
p05_candidates = []
if v21_eval:
    p05_candidates += ["--candidate", f"v2_1_p05={v21_eval}"]
if v22_eval:
    p05_candidates += ["--candidate", f"v2_2_p05={v22_eval}"]
if p05_candidates and not p05_selection_path.exists():
    module(
        "tools.select_hi_csg_r_adapter_v2_candidate",
        "select", *p05_candidates,
        "--purpose", "select_best_p05_before_single_p10_repeat",
        "--out", p05_selection_path,
        log_name="select_p05",
        allowed=(0, 2),
    )
p05_selection = (
    show_json(p05_selection_path, "P05 selection")
    if p05_selection_path.exists()
    else None
)

p10 = OUT / "v2_best_dev_p10_seed42/train_summary.json"
p10_eval = None
if p05_selection and p05_selection["status"] == "PASS":
    resolved_p10 = OUT / "development/resolved_v2_best_dev_p10_seed42.yaml"
    if not resolved_p10.exists():
        module(
            "tools.select_hi_csg_r_adapter_v2_candidate",
            "resolve-p10",
            "--selection", p05_selection_path,
            "--template", "configs/htr_adapter_v2/v2_best_dev_p10_seed42.yaml",
            "--out", resolved_p10,
            log_name="resolve_p10",
        )
    if not p10.exists():
        module(
            "tools.train_crnn_ctc_hi_csg_r_late_correction_v2",
            "train", "--config", resolved_p10,
            *resume_args(OUT / "v2_best_dev_p10_seed42"),
            log_name="v2_best_dev_p10",
        )
    if p10.exists():
        p10_eval = evaluate_modes(
            OUT / "v2_best_dev_p10_seed42",
            DATA / "features/dev.jsonl",
            "dev",
        )
        p10_decision_dir = OUT / "development/v2_best_dev_p10_seed42/decision"
        if not (p10_decision_dir / "dev_decision.json").exists():
            module(
                "tools.compare_hi_csg_r_adapter_v2_results",
                "--stage", "dev",
                "--correct", p10_eval / "correct",
                "--shuffle", p10_eval / "shuffle",
                "--zero", p10_eval / "zero",
                "--out_dir", p10_decision_dir,
                log_name="v2_best_p10_dev_gate",
                allowed=(0, 2),
            )
        show_json(p10_decision_dir / "dev_decision.json", "P10 dev gate")
"""
        ),
        markdown("## 10. Freeze кандидата и one-shot holdout"),
        code(
            """
selection_path = OUT / "development/selected_candidate.json"
final_candidates = []
if v21_eval:
    final_candidates += ["--candidate", f"v2_1_p05={v21_eval}"]
if v22_eval:
    final_candidates += ["--candidate", f"v2_2_p05={v22_eval}"]
if p10_eval:
    final_candidates += ["--candidate", f"best_p10={p10_eval}"]
if final_candidates and not selection_path.exists():
    module(
        "tools.select_hi_csg_r_adapter_v2_candidate",
        "select", *final_candidates,
        "--purpose", "freeze_candidate_before_one_shot_holdout",
        "--out", selection_path,
        log_name="freeze_dev_candidate",
        allowed=(0, 2),
    )
selected = (
    show_json(selection_path, "Frozen selected candidate")
    if selection_path.exists()
    else None
)
if selected:
    dev_rows = []
    for candidate in selected["candidates"]:
        decision = candidate["decision"]
        dev_rows.append({
            "candidate": candidate["name"],
            "gate": decision["status"],
            "baseline_CER": decision["baseline"]["cer"],
            "correct_CER": decision["correct"]["cer"],
            "shuffle_CER": decision["shuffle"]["cer"],
            "zero_CER": decision["zero"]["cer"],
            "relative_improvement": decision["relative_cer_improvement"],
            "Exact": decision["correct"]["exact"],
        })
    display(pd.DataFrame(dev_rows).sort_values("correct_CER"))

holdout_decision_path = OUT / "holdout/decision/holdout_decision.json"
if selected and selected["status"] == "PASS" and not holdout_decision_path.exists():
    selected_run = Path(selected["selected"]["checkpoint"]).parent
    holdout_manifest = DATA / "features/holdout.jsonl"
    holdout_map = DATA / "shuffle_maps/holdout_seed42.json"
    if not holdout_map.exists():
        module(
            "tools.build_hi_csg_r_adapter_v2_shuffle_map",
            "--manifest", holdout_manifest,
            "--seed", 42,
            "--out", holdout_map,
            log_name="shuffle_holdout",
        )
    holdout_root = OUT / "holdout/evaluation"
    for mode in ("correct", "shuffle", "zero"):
        destination = holdout_root / mode
        if not (destination / "summary.json").exists():
            extra = ("--shuffle_map", holdout_map) if mode == "shuffle" else ()
            module(
                "tools.evaluate_crnn_ctc_hi_csg_r_late_correction_v2",
                "--checkpoint", selected["selected"]["checkpoint"],
                "--manifest", holdout_manifest,
                "--mode", mode,
                *extra,
                "--out_dir", destination,
                log_name=f"holdout_{mode}",
            )
    module(
        "tools.compare_hi_csg_r_adapter_v2_results",
        "--stage", "holdout",
        "--correct", holdout_root / "correct",
        "--shuffle", holdout_root / "shuffle",
        "--zero", holdout_root / "zero",
        "--out_dir", holdout_decision_path.parent,
        log_name="holdout_gate",
        allowed=(0, 2),
    )

holdout = (
    show_json(holdout_decision_path, "One-shot holdout decision")
    if holdout_decision_path.exists()
    else None
)
if selected and selected["status"] == "STOP":
    display(Markdown("**STOP на development:** holdout и test не открыты."))
"""
        ),
        markdown("## 11. Intervention/failure analysis"),
        code(
            """
analysis_predictions = None
analysis_summary = None
analysis_manifest = None
analysis_name = None
if holdout:
    analysis_predictions = OUT / "holdout/evaluation/correct/predictions.jsonl"
    analysis_summary = OUT / "holdout/evaluation/correct/summary.json"
    analysis_manifest = DATA / "features/holdout.jsonl"
    analysis_name = "holdout"
elif selected and selected["status"] == "PASS":
    analysis_root = Path(selected["selected"]["evaluation_dir"]) / "correct"
    analysis_predictions = analysis_root / "predictions.jsonl"
    analysis_summary = analysis_root / "summary.json"
    analysis_manifest = DATA / "features/dev.jsonl"
    analysis_name = "development"
if analysis_predictions:
    failure_dir = OUT / f"failure_analysis/{analysis_name}"
    if not (failure_dir / "failure_analysis.json").exists():
        module(
            "tools.analyze_hi_csg_r_adapter_v2_failures",
            "--predictions", analysis_predictions,
            "--summary", analysis_summary,
            "--manifest", analysis_manifest,
            "--out_dir", failure_dir,
            "--limit", 20,
            log_name=f"failure_analysis_{analysis_name}",
        )
    show_json(failure_dir / "failure_analysis.json", "Intervention groups")
"""
        ),
        markdown("## 12. Условные final seeds"),
        code(
            """
final_allowed = bool(holdout and holdout["status"] == "PASS")
frozen_configs = OUT / "frozen_final_configs"
if final_allowed and not (frozen_configs / "final_seed42.yaml").exists():
    module(
        "tools.resolve_hi_csg_r_adapter_v2_final_configs",
        "--selection", selection_path,
        "--holdout_decision", holdout_decision_path,
        "--out_dir", frozen_configs,
        log_name="freeze_final_configs",
    )
if final_allowed:
    # Seed 42 matched M0-FT already exists and is reused without changing v1.
    for seed in (43, 44):
        m0_checkpoint = OUT / f"m0_ft_final_seed{seed}/best.pt"
        if not m0_checkpoint.exists():
            module(
                "tools.train_crnn_ctc_hi_csg_r_adapter_v1",
                "train",
                "--config", frozen_configs / f"m0_ft_final_seed{seed}.yaml",
                log_name=f"m0_ft_final_seed{seed}",
            )
    for seed in (42, 43, 44):
        summary_path = OUT / f"final_seed{seed}/train_summary.json"
        if not summary_path.exists():
            module(
                "tools.train_crnn_ctc_hi_csg_r_late_correction_v2",
                "train",
                "--config", frozen_configs / f"final_seed{seed}.yaml",
                *resume_args(OUT / f"final_seed{seed}"),
                log_name=f"v2_final_seed{seed}",
            )
else:
    display(Markdown(
        "**Final seeds/test заблокированы**: one-shot holdout не дал PASS."
    ))
"""
        ),
        markdown("## 13. Однократный main test и paired statistics"),
        code(
            """
final_summaries = [
    OUT / f"final_seed{seed}/train_summary.json" for seed in (42, 43, 44)
]
all_final_complete = final_allowed and all(path.exists() for path in final_summaries)
test_manifest = ROOT / "data/experiments/htr_adapter_v1/manifests/test.jsonl"
test_map = DATA / "shuffle_maps/final_test_seed42.json"
test_root = OUT / "final_evaluation/test"
if all_final_complete:
    if not test_map.exists():
        module(
            "tools.build_hi_csg_r_adapter_v2_shuffle_map",
            "--manifest", test_manifest,
            "--seed", 42,
            "--out", test_map,
            log_name="shuffle_final_test",
        )
    for seed in (42, 43, 44):
        for mode in ("correct", "shuffle"):
            destination = test_root / f"seed{seed}/{mode}"
            if not (destination / "summary.json").exists():
                extra = ("--shuffle_map", test_map) if mode == "shuffle" else ()
                module(
                    "tools.evaluate_crnn_ctc_hi_csg_r_late_correction_v2",
                    "--checkpoint", OUT / f"final_seed{seed}/best.pt",
                    "--manifest", test_manifest,
                    "--mode", mode,
                    *extra,
                    "--out_dir", destination,
                    log_name=f"final_test_seed{seed}_{mode}",
                )

    baseline_predictions = []
    correct_predictions = []
    shuffle_predictions = []
    for seed in (42, 43, 44):
        correct_path = test_root / f"seed{seed}/correct/predictions.jsonl"
        baseline_path = test_root / f"seed{seed}/baseline_predictions.jsonl"
        if not baseline_path.exists():
            module(
                "tools.materialize_hi_csg_r_adapter_v2_baseline_predictions",
                "--v2_predictions", correct_path,
                "--out", baseline_path,
                log_name=f"materialize_baseline_seed{seed}",
            )
        baseline_predictions.append(baseline_path)
        correct_predictions.append(correct_path)
        shuffle_predictions.append(
            test_root / f"seed{seed}/shuffle/predictions.jsonl"
        )
    stats_dir = OUT / "statistical_analysis"
    primary_stats = stats_dir / "m0_vs_v2_correct.json"
    shuffle_stats = stats_dir / "v2_correct_vs_shuffle.json"
    if not primary_stats.exists():
        module(
            "tools.paired_bootstrap_hi_csg_r_adapter_v2",
            "--baseline_predictions", *baseline_predictions,
            "--adapter_predictions", *correct_predictions,
            "--iterations", 10000,
            "--seed", 20260730,
            "--out", primary_stats,
            log_name="bootstrap_m0_vs_v2",
        )
    if not shuffle_stats.exists():
        module(
            "tools.paired_bootstrap_hi_csg_r_adapter_v2",
            "--baseline_predictions", *correct_predictions,
            "--adapter_predictions", *shuffle_predictions,
            "--iterations", 10000,
            "--seed", 20260730,
            "--out", shuffle_stats,
            log_name="bootstrap_correct_vs_shuffle",
        )
    if not (stats_dir / "final_statistics.json").exists():
        module(
            "tools.summarize_hi_csg_r_adapter_v2_final_statistics",
            "--comparison", f"m0_vs_v2={primary_stats}",
            "--comparison", f"correct_vs_shuffle={shuffle_stats}",
            "--primary", "m0_vs_v2",
            "--out_dir", stats_dir,
            log_name="final_statistics",
        )
    show_json(stats_dir / "final_statistics.json", "Final paired statistics")
else:
    display(Markdown("Main test остается закрытым до трех завершенных final seeds."))
"""
        ),
        markdown("## 14. Page-disjoint и robustness после main test"),
        code(
            """
main_test_reported = (OUT / "statistical_analysis/final_statistics.json").exists()
if main_test_reported:
    extra_manifests = {
        "page_disjoint": ROOT / "data/experiments/htr_adapter_v1/manifests/page_disjoint.jsonl",
    }
    for path in sorted(
        (ROOT / "data/experiments/htr_adapter_v1/manifests").glob("robustness_*.jsonl")
    ):
        if not path.name.endswith(".failures.jsonl"):
            extra_manifests[path.stem] = path
    for name, manifest in extra_manifests.items():
        destination = OUT / f"final_evaluation/additional/{name}/seed42_correct"
        if manifest.exists() and not (destination / "summary.json").exists():
            module(
                "tools.evaluate_crnn_ctc_hi_csg_r_late_correction_v2",
                "--checkpoint", OUT / "final_seed42/best.pt",
                "--manifest", manifest,
                "--mode", "correct",
                "--out_dir", destination,
                log_name=f"additional_{name}",
            )
else:
    display(Markdown("Дополнительные наборы не оцениваются до main test reporting."))
"""
        ),
        markdown("## 15. Итоговый отчет, таблицы и фигуры"),
        code(
            """
module(
    "tools.make_hi_csg_r_adapter_v2_final_report",
    "--root", OUT,
    "--out_dir", OUT / "final_report",
    log_name="final_report_v2",
)
full_report = OUT / "final_report/full_execution_report_ru.md"
display(Markdown(
    (
        full_report
        if full_report.exists()
        else OUT / "final_report/final_results.md"
    ).read_text(encoding="utf-8")
))
for name in (
    "figure_a_architecture.png",
    "figure_b_intervention.png",
    "figure_c_results.png",
    "figure_d_helps_hurts.png",
):
    path = OUT / "final_report" / name
    if path.exists():
        display(Image(filename=str(path)))
"""
        ),
        markdown(
            """
## Интерпретация

Научный статус определяется только dev/holdout/final gates. Технические
invariants, one-sample overfit, gradients, gate variability и zero-graph
dependency сами по себе не подтверждают H4-v2.
"""
        ),
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
