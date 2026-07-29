from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            """
# CRNN-CTC + x-aligned HI-CSG-R: полный эксперимент v1

Этот notebook исполняет зафиксированный протокол
`docs/crnn_ctc_hi_csg_r_adapter_protocol_v1.md`.

Принципы:

- основной baseline: canonical image-only CRNN-CTC `+10k`;
- seed: `42`, `43`, `44`;
- fixed blank penalty: `-0.4`;
- выбор checkpoint только по validation micro-CER;
- test недоступен до успешного validation gate и создания freeze registry;
- distorted images всегда получают заново построенный HI-CSG-R;
- после отрицательного seed-42 gate ветка останавливается.

Тяжёлые стадии выключены по умолчанию. Включайте флаги последовательно, не все сразу.
"""
        ),
        code(
            """
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image as DisplayImage, Markdown, display

START_DIR = Path.cwd().resolve()
ROOT = next(
    (
        candidate
        for candidate in (START_DIR, *START_DIR.parents)
        if (candidate / "src" / "htr").exists()
        and (candidate / "tools" / "train_crnn_ctc_hi_csg_r_adapter_v1.py").exists()
    ),
    None,
)
if ROOT is None:
    raise RuntimeError("Не найден корень репозитория hi-csg-r")
os.chdir(ROOT)

PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

SEEDS = (42, 43, 44)
BLANK_PENALTY = -0.4
RUN_ROOT = ROOT / "outputs" / "htr_adapter_v1"
DATA_ROOT = ROOT / "data" / "experiments" / "htr_adapter_v1"
LOG_ROOT = RUN_ROOT / "notebook" / "logs"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

EXECUTE = {
    "input_audit": True,
    "feature_build": False,
    "feature_audit": False,
    "tests": True,
    "visual_audit": False,
    "smoke": False,
    "seed42": False,
    "validation": False,
    "final_seeds": False,
    "final_test": False,
    "statistics": False,
    "final_report": False,
}

# Изменить на True разрешается только после PASS validation gate и freeze.
ALLOW_FINAL_TEST = False
"""
        ),
        markdown("## 1. Frozen paths и служебные функции"),
        code(
            """
CANONICAL = ROOT / "data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1"
VOCAB = CANONICAL / "vocab.json"
SOURCE_MANIFESTS = {
    "train": CANONICAL / "train.jsonl",
    "val": CANONICAL / "val.jsonl",
    "test": CANONICAL / "test.jsonl",
    "page_disjoint": ROOT / "data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1/test.jsonl",
    "clean_core": ROOT / "data/experiments/iter2_quality_manifests/school_notebooks_lineaware_v3/test.clean_core.jsonl",
    "hard_real": ROOT / "data/experiments/iter2_quality_manifests/school_notebooks_lineaware_v3/test.hard_real.jsonl",
}
ROBUSTNESS_SOURCE = ROOT / "data/experiments/htr_graph_v1/robustness_v2_recomputed/manifests"
CHECKPOINTS = {
    42: ROOT / "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1/best.pt",
    43: ROOT / "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1_seed43/best.pt",
    44: ROOT / "outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1_seed44/best.pt",
}
ENHANCED = {
    name: DATA_ROOT / "manifests" / f"{name}.jsonl"
    for name in ("train", "val", "test", "page_disjoint", "clean_core", "hard_real")
}
NORMALIZER = DATA_ROOT / "normalizer" / "train_stats.json"
FREEZE_REGISTRY = RUN_ROOT / "freeze_registry_v1.json"
GLOBAL_VECTOR_SUMMARY = (
    ROOT / "outputs/htr_graph_v1/"
    "eval_tri10k_graph_fusion_plus_school_lines_10k_context_v1_test_final/summary.json"
)
GLOBAL_VECTOR_PREDICTIONS = GLOBAL_VECTOR_SUMMARY.parent / "predictions.jsonl"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows),
        encoding="utf-8",
    )

def run_cmd(args, name: str, *, check: bool = True):
    args = [str(value) for value in args]
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (LOG_ROOT / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (LOG_ROOT / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    print("$", " ".join(args))
    print(result.stdout[-4000:])
    if result.stderr:
        print(result.stderr[-2000:], file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"{name} завершился с кодом {result.returncode}")
    return result

def module(name: str, *args, log_name: str, check: bool = True):
    return run_cmd([PYTHON, "-m", name, *args], log_name, check=check)

assert all(path.exists() for path in CHECKPOINTS.values())
assert all(path.exists() for path in SOURCE_MANIFESTS.values())
assert VOCAB.exists()
print("Frozen inputs: PASS")
"""
        ),
        markdown("## 2. WP0–WP1: protocol freeze и аудит входов"),
        code(
            """
PROTOCOL = ROOT / "docs/crnn_ctc_hi_csg_r_adapter_protocol_v1.md"
assert PROTOCOL.exists()
print("protocol_sha256:", sha256(PROTOCOL))

if EXECUTE["input_audit"]:
    module(
        "tools.audit_adapter_inputs_v1",
        "--train_manifest", SOURCE_MANIFESTS["train"],
        "--val_manifest", SOURCE_MANIFESTS["val"],
        "--test_manifest", SOURCE_MANIFESTS["test"],
        "--checkpoints", CHECKPOINTS[42], CHECKPOINTS[43], CHECKPOINTS[44],
        "--checkpoint_seeds", 42, 43, 44,
        "--vocab", VOCAB,
        "--expected_counts", 39998, 6000, 5563,
        "--out_dir", RUN_ROOT / "input_audit",
        log_name="input_audit",
    )
    audit = read_json(RUN_ROOT / "input_audit" / "report.json")
    assert audit["status"] == "PASS"
    display(Markdown((RUN_ROOT / "input_audit" / "report.md").read_text(encoding="utf-8")))
"""
        ),
        markdown("## 3. WP2: построение x-aligned features для всех manifests"),
        code(
            """
def build_features(name: str, source: Path, *, domain_split: bool = False):
    out_dir = DATA_ROOT / "xaligned" / name
    out_manifest = ENHANCED.get(name, DATA_ROOT / "manifests" / f"{name}.jsonl")
    summary = RUN_ROOT / "feature_build" / f"{name}_summary.json"
    args = [
        "--manifest", source,
        "--out_dir", out_dir,
        "--out_manifest", out_manifest,
        "--summary", summary,
        "--feature_version", "hi_csg_r_xaligned_v1",
        "--width_downsample", 4,
        "--smooth",
        "--workers", max((os.cpu_count() or 2) // 2, 1),
    ]
    if domain_split:
        args += ["--domain_manifest_dir", DATA_ROOT / "manifests"]
    module(
        "tools.build_xaligned_hi_csg_r_features_v1",
        *args,
        log_name=f"feature_build_{name}",
    )
    result = read_json(summary)
    assert result["status"] == "PASS"
    assert result["written_n"] == result["expected_n"]
    assert result["feature_dim"] == 20

if EXECUTE["feature_build"]:
    build_features("train", SOURCE_MANIFESTS["train"])
    build_features("val", SOURCE_MANIFESTS["val"])
    build_features("test", SOURCE_MANIFESTS["test"], domain_split=True)
    build_features("page_disjoint", SOURCE_MANIFESTS["page_disjoint"])
    build_features("clean_core", SOURCE_MANIFESTS["clean_core"])
    build_features("hard_real", SOURCE_MANIFESTS["hard_real"])
    for source in sorted(ROBUSTNESS_SOURCE.glob("*.jsonl")):
        # Каждый distorted manifest строит foreground/skeleton/graph заново.
        build_features(f"robustness_{source.stem}", source)
"""
        ),
        markdown("## 4. WP3: train-only normalizer, автоматический и визуальный audit"),
        code(
            """
if EXECUTE["feature_audit"]:
    module(
        "tools.audit_xaligned_features_v1",
        "fit-normalizer",
        "--manifest", ENHANCED["train"],
        "--out", NORMALIZER,
        log_name="fit_normalizer",
    )
    module(
        "tools.audit_xaligned_features_v1",
        "audit",
        "--manifests", ENHANCED["train"], ENHANCED["val"], ENHANCED["test"],
        "--normalizer", NORMALIZER,
        "--out_dir", RUN_ROOT / "feature_audit",
        log_name="feature_audit",
    )
    audit = read_json(RUN_ROOT / "feature_audit" / "feature_audit.json")
    assert audit["status"] == "PASS"
    display(Markdown((RUN_ROOT / "feature_audit" / "feature_audit.md").read_text(encoding="utf-8")))

if EXECUTE["visual_audit"]:
    module(
        "tools.visualize_xaligned_features_v1",
        "--manifest", ENHANCED["test"],
        "--out_dir", RUN_ROOT / "feature_audit" / "browser",
        log_name="feature_visual_audit",
    )
    selection = read_json(RUN_ROOT / "feature_audit" / "browser" / "selection.json")
    assert len(selection) == 30
    assert pd.Series([row["dataset"] for row in selection]).value_counts().to_dict() == {
        "cyrillic": 10, "hkr": 10, "school": 10
    }
    display(Markdown("[Открыть visual browser](feature_audit/browser/browser.html)"))
"""
        ),
        markdown("## 5. WP4–WP7: tests и initial equivalence"),
        code(
            """
if EXECUTE["tests"]:
    run_cmd(
        [
            ROOT / ".venv/bin/pytest",
            "-q",
            "tests/test_xaligned_hi_csg_r.py",
            "tests/test_hi_csg_r_adapter_data.py",
            "tests/test_hi_csg_r_adapter_model.py",
        ],
        "adapter_tests",
    )
    run_cmd(
        [
            ROOT / ".venv/bin/ruff",
            "check",
            "src/htr/xaligned_hi_csg_r.py",
            "src/htr/dataset_adapter.py",
            "src/htr/model_hi_csg_r_adapter.py",
            "src/htr/adapter_runtime.py",
            "tools/audit_adapter_inputs_v1.py",
            "tools/build_xaligned_hi_csg_r_features_v1.py",
            "tools/audit_xaligned_features_v1.py",
            "tools/visualize_xaligned_features_v1.py",
            "tools/train_crnn_ctc_hi_csg_r_adapter_v1.py",
            "tools/evaluate_crnn_ctc_hi_csg_r_adapter_v1.py",
            "tools/build_hi_csg_r_shuffle_map_v1.py",
            "tools/compare_hi_csg_r_adapter_results_v1.py",
            "tools/paired_bootstrap_hi_csg_r_adapter_v1.py",
            "tools/make_hi_csg_r_adapter_final_report_v1.py",
            "tests/test_xaligned_hi_csg_r.py",
            "tests/test_hi_csg_r_adapter_data.py",
            "tests/test_hi_csg_r_adapter_model.py",
        ],
        "adapter_ruff",
        check=True,
    )

if ENHANCED["train"].exists() and NORMALIZER.exists():
    from src.htr.dataset_adapter import HICSGRAdapterDataset, collate_adapter_batch
    from src.htr.model_hi_csg_r_adapter import (
        CRNNCTCHICSGRAdapter,
        baseline_model_config,
        load_canonical_image_model,
        load_canonical_visual_weights,
    )
    from src.htr.vocab import CTCVocab
    from src.htr.xaligned_hi_csg_r import XAlignedFeatureNormalizer
    import torch

    vocab = CTCVocab.from_path(VOCAB)
    normalizer = XAlignedFeatureNormalizer.from_path(NORMALIZER)
    dataset = HICSGRAdapterDataset(ENHANCED["train"], vocab, normalizer=normalizer)
    batch = collate_adapter_batch([dataset[0]])
    baseline, _ = load_canonical_image_model(
        CHECKPOINTS[42], num_classes=vocab.num_classes, blank_index=vocab.blank_index
    )
    raw = torch.load(CHECKPOINTS[42], map_location="cpu", weights_only=False)
    adapter = CRNNCTCHICSGRAdapter(
        num_classes=vocab.num_classes,
        blank_index=vocab.blank_index,
        **baseline_model_config(raw),
    )
    load_canonical_visual_weights(adapter, CHECKPOINTS[42])
    baseline.eval(); adapter.eval()
    with torch.no_grad():
        expected = baseline(batch["images"])
        actual = adapter(
            batch["images"], batch["widths"], batch["graph_features"],
            batch["graph_quality"], batch["graph_mask"]
        )["log_probs"]
    delta = float((expected - actual).abs().max())
    assert delta == 0.0
    print("real-checkpoint initial equivalence:", delta)
"""
        ),
        markdown("## 6. WP8: one-sample и 128-sample overfit gates"),
        code(
            """
def make_smoke_manifests():
    rows = read_jsonl(ENHANCED["train"])
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    rng = random.Random(42)
    selected = []
    for dataset in sorted(by_dataset):
        values = list(by_dataset[dataset])
        rng.shuffle(values)
        selected.extend(values[:32])
    assert 64 <= len(selected) <= 128
    one = DATA_ROOT / "manifests" / "smoke_one.jsonl"
    small = DATA_ROOT / "manifests" / "smoke_128.jsonl"
    write_jsonl(one, [selected[0]])
    write_jsonl(small, selected)
    return one, small

def train_direct(name: str, *args):
    module(
        "tools.train_crnn_ctc_hi_csg_r_adapter_v1",
        "train", *args,
        log_name=f"train_{name}",
    )

if EXECUTE["smoke"]:
    one_manifest, small_manifest = make_smoke_manifests()
    common = [
        "--mode", "m3_full",
        "--base_checkpoint", CHECKPOINTS[42],
        "--vocab", VOCAB,
        "--normalizer", NORMALIZER,
        "--seed", 42,
        "--batch_size", 16,
        "--num_workers", 4,
        "--blank_logit_penalty", BLANK_PENALTY,
    ]
    train_direct(
        "smoke_one",
        *common,
        "--train_manifest", one_manifest,
        "--val_manifest", one_manifest,
        "--warmup_epochs", 5,
        "--joint_epochs", 20,
        "--out_dir", RUN_ROOT / "smoke" / "one_sample",
    )
    train_direct(
        "smoke_aux_128",
        *common,
        "--train_manifest", small_manifest,
        "--val_manifest", small_manifest,
        "--warmup_epochs", 20,
        "--joint_epochs", 0,
        "--out_dir", RUN_ROOT / "smoke" / "aux_128",
    )
    train_direct(
        "smoke_full_128",
        *common,
        "--train_manifest", small_manifest,
        "--val_manifest", small_manifest,
        "--warmup_epochs", 5,
        "--joint_epochs", 15,
        "--out_dir", RUN_ROOT / "smoke" / "full_128",
    )
"""
        ),
        code(
            """
def smoke_gate():
    one = read_json(RUN_ROOT / "smoke" / "one_sample" / "val_summary.json")
    aux_history = read_jsonl(RUN_ROOT / "smoke" / "aux_128" / "history.jsonl")
    full_history = read_jsonl(RUN_ROOT / "smoke" / "full_128" / "history.jsonl")
    conditions = {
        "one_sample_near_zero_cer": one["cer"] <= 0.01,
        "aux_loss_decreases": aux_history[-1]["train_graph_aux_ctc_loss"]
        < aux_history[0]["train_graph_aux_ctc_loss"],
        "adapter_grad_nonzero": max(row["graph_adapter_grad_norm"] for row in full_history) > 0,
        "gate_not_constant": max((row.get("gate") or {}).get("std", 0) for row in full_history) > 0,
        "no_blank_collapse": min(row["blank_ratio"] for row in full_history) < 0.99,
        "finite_losses": all(np.isfinite(row["train_total_loss"]) for row in full_history),
    }
    result = {"status": "PASS" if all(conditions.values()) else "STOP", "conditions": conditions}
    path = RUN_ROOT / "smoke" / "smoke_gate.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if EXECUTE["smoke"]:
    SMOKE_GATE = smoke_gate()
    display(SMOKE_GATE)
    assert SMOKE_GATE["status"] == "PASS", "Full seed-42 experiment запрещён"
"""
        ),
        markdown("## 7. WP9: development seed 42, correct/shuffled/topology-off"),
        code(
            """
if EXECUTE["seed42"]:
    assert read_json(RUN_ROOT / "smoke" / "smoke_gate.json")["status"] == "PASS"
    for config_name in ("m0_ft_seed42.yaml", "m3_full_seed42.yaml"):
        module(
            "tools.train_crnn_ctc_hi_csg_r_adapter_v1",
            "train",
            "--config", ROOT / "configs/htr_adapter_v1" / config_name,
            log_name=f"train_{Path(config_name).stem}",
        )
"""
        ),
        code(
            """
VAL_EVAL = RUN_ROOT / "validation_seed42"

def evaluate(checkpoint: Path, manifest: Path, out_dir: Path, *, normalizer=True, shuffle=None):
    args = [
        "--checkpoint", checkpoint,
        "--manifest", manifest,
        "--vocab", VOCAB,
        "--blank_logit_penalty", BLANK_PENALTY,
        "--out_dir", out_dir,
    ]
    if normalizer:
        args += ["--normalizer", NORMALIZER]
    if shuffle:
        args += ["--shuffle_map", shuffle]
    module(
        "tools.evaluate_crnn_ctc_hi_csg_r_adapter_v1",
        *args,
        log_name=f"eval_{out_dir.name}",
    )

if EXECUTE["validation"]:
    evaluate(
        RUN_ROOT / "m0_ft_seed42" / "best.pt",
        ENHANCED["val"],
        VAL_EVAL / "m0_ft",
        normalizer=False,
    )
    evaluate(
        RUN_ROOT / "m3_full_seed42" / "best.pt",
        ENHANCED["val"],
        VAL_EVAL / "m3_correct",
    )
    shuffle_map = DATA_ROOT / "shuffle_maps" / "val_seed42.json"
    module(
        "tools.build_hi_csg_r_shuffle_map_v1",
        "--manifest", ENHANCED["val"],
        "--seed", 42,
        "--out", shuffle_map,
        log_name="shuffle_map_val",
    )
    evaluate(
        RUN_ROOT / "m3_full_seed42" / "best.pt",
        ENHANCED["val"],
        VAL_EVAL / "m3_shuffle",
        shuffle=shuffle_map,
    )
"""
        ),
        code(
            """
def preliminary_gate_without_m2():
    m0 = read_json(VAL_EVAL / "m0_ft" / "summary.json")
    m3 = read_json(VAL_EVAL / "m3_correct" / "summary.json")
    shuffled = read_json(VAL_EVAL / "m3_shuffle" / "summary.json")
    d0 = read_json(VAL_EVAL / "m0_ft" / "domain_summary.json")
    d3 = read_json(VAL_EVAL / "m3_correct" / "domain_summary.json")
    deltas = {key: d3[key]["cer"] - d0[key]["cer"] for key in sorted(set(d0) & set(d3))}
    history = read_jsonl(RUN_ROOT / "m3_full_seed42" / "history.jsonl")
    conditions = {
        "relative_improvement_2pct": (m0["cer"] - m3["cer"]) / max(m0["cer"], 1e-12) >= 0.02,
        "two_domains_not_worse": sum(value <= 0 for value in deltas.values()) >= 2,
        "max_domain_degradation": max(deltas.values(), default=0) <= 0.005,
        "correct_better_shuffle": m3["cer"] < shuffled["cer"],
        "gate_variable": (m3.get("gate") or {}).get("std", 0) > 0,
        "adapter_gradient": any(row["graph_adapter_grad_norm"] > 0 for row in history),
    }
    return {"status": "PASS" if all(conditions.values()) else "STOP", "conditions": conditions}

if EXECUTE["validation"]:
    PRE_GATE = preliminary_gate_without_m2()
    display(PRE_GATE)
    if PRE_GATE["status"] == "PASS":
        module(
            "tools.train_crnn_ctc_hi_csg_r_adapter_v1",
            "train",
            "--config", ROOT / "configs/htr_adapter_v1/m2_geometry_seed42.yaml",
            log_name="train_m2_geometry_seed42",
        )
        evaluate(
            RUN_ROOT / "m2_geometry_seed42" / "best.pt",
            ENHANCED["val"],
            VAL_EVAL / "m2",
        )
    else:
        raise RuntimeError("Seed-42 pre-gate STOP: M2 и seeds 43/44 запускать нельзя")
"""
        ),
        code(
            """
if EXECUTE["validation"]:
    gate_dir = RUN_ROOT / "statistical_analysis" / "validation_gate"
    result = module(
        "tools.compare_hi_csg_r_adapter_results_v1",
        "--run", f"M0-FT={VAL_EVAL / 'm0_ft' / 'summary.json'}",
        "--run", f"M2={VAL_EVAL / 'm2' / 'summary.json'}",
        "--run", f"M3={VAL_EVAL / 'm3_correct' / 'summary.json'}",
        "--run", f"M3-shuffle={VAL_EVAL / 'm3_shuffle' / 'summary.json'}",
        "--validation_gate",
        "--domain_summary", f"M0-FT={VAL_EVAL / 'm0_ft' / 'domain_summary.json'}",
        "--domain_summary", f"M3={VAL_EVAL / 'm3_correct' / 'domain_summary.json'}",
        "--m3_history", RUN_ROOT / "m3_full_seed42" / "history.jsonl",
        "--out_dir", gate_dir,
        log_name="validation_gate",
        check=False,
    )
    VALIDATION_GATE = read_json(gate_dir / "validation_gate.json")
    display(VALIDATION_GATE)
    assert VALIDATION_GATE["status"] == "PASS", "Seeds 43/44 и test запрещены"
"""
        ),
        markdown("## 8. WP10: final seeds и freeze registry"),
        code(
            """
if EXECUTE["final_seeds"]:
    gate = read_json(RUN_ROOT / "statistical_analysis/validation_gate/validation_gate.json")
    assert gate["status"] == "PASS"
    for seed in (43, 44):
        for prefix in ("m0_ft", "m3_full"):
            config = ROOT / "configs/htr_adapter_v1" / f"{prefix}_seed{seed}.yaml"
            module(
                "tools.train_crnn_ctc_hi_csg_r_adapter_v1",
                "train", "--config", config,
                log_name=f"train_{prefix}_seed{seed}",
            )

    frozen = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "blank_logit_penalty": BLANK_PENALTY,
        "validation_gate": gate,
        "checkpoints": {},
        "test_evaluated": False,
    }
    for model in ("m0_ft", "m3_full"):
        for seed in SEEDS:
            path = RUN_ROOT / f"{model}_seed{seed}" / "best.pt"
            frozen["checkpoints"][f"{model}_seed{seed}"] = {
                "path": str(path),
                "sha256": sha256(path),
            }
    FREEZE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_REGISTRY.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    display(frozen)
"""
        ),
        markdown(
            """
## 9. WP11: однократная final test evaluation

Эта секция намеренно имеет двойную блокировку:

1. `validation_gate.status == PASS`;
2. `ALLOW_FINAL_TEST = True`.

После первого успешного выполнения registry получает `test_evaluated=true`.
"""
        ),
        code(
            """
FINAL_MANIFESTS = {
    "mixed": ENHANCED["test"],
    "cyrillic": DATA_ROOT / "manifests/test_cyrillic.jsonl",
    "hkr": DATA_ROOT / "manifests/test_hkr.jsonl",
    "school": DATA_ROOT / "manifests/test_school.jsonl",
    "page_disjoint": ENHANCED["page_disjoint"],
    "clean_core": ENHANCED["clean_core"],
    "hard_real": ENHANCED["hard_real"],
}
for source in sorted(ROBUSTNESS_SOURCE.glob("*.jsonl")):
    FINAL_MANIFESTS[f"robustness_{source.stem}"] = (
        DATA_ROOT / "manifests" / f"robustness_{source.stem}.jsonl"
    )

def final_evaluate_model(model: str, seed: int, checkpoint: Path):
    for split, manifest in FINAL_MANIFESTS.items():
        out = RUN_ROOT / "final_evaluation" / model / f"seed{seed}" / split
        evaluate(
            checkpoint,
            manifest,
            out,
            normalizer=model != "M0",
        )
        if model == "M3":
            mapping = DATA_ROOT / "shuffle_maps" / f"{split}_seed42.json"
            if not mapping.exists():
                module(
                    "tools.build_hi_csg_r_shuffle_map_v1",
                    "--manifest", manifest,
                    "--seed", 42,
                    "--out", mapping,
                    log_name=f"shuffle_map_{split}",
                )
            evaluate(
                checkpoint,
                manifest,
                RUN_ROOT / "final_evaluation" / "M3-shuffle" / f"seed{seed}" / split,
                shuffle=mapping,
            )

if EXECUTE["final_test"]:
    assert ALLOW_FINAL_TEST, "Явно установите ALLOW_FINAL_TEST=True после freeze"
    registry = read_json(FREEZE_REGISTRY)
    assert registry["validation_gate"]["status"] == "PASS"
    assert not registry["test_evaluated"], "Final test уже был выполнен"
    for seed in SEEDS:
        final_evaluate_model("M0", seed, CHECKPOINTS[seed])
        final_evaluate_model("M0-FT", seed, RUN_ROOT / f"m0_ft_seed{seed}" / "best.pt")
        final_evaluate_model("M3", seed, RUN_ROOT / f"m3_full_seed{seed}" / "best.pt")
    final_evaluate_model("M2", 42, RUN_ROOT / "m2_geometry_seed42" / "best.pt")
    registry["test_evaluated"] = True
    registry["test_evaluated_at"] = datetime.now(timezone.utc).isoformat()
    FREEZE_REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
"""
        ),
        markdown("## 10. WP12: paired bootstrap, Holm correction и seed statistics"),
        code(
            """
STATS = RUN_ROOT / "statistical_analysis"

def prediction_paths(model: str, split: str = "mixed", seeds=SEEDS):
    return [
        RUN_ROOT / "final_evaluation" / model / f"seed{seed}" / split / "predictions.jsonl"
        for seed in seeds
    ]

def bootstrap(name: str, baseline, adapter):
    output = STATS / f"{name}.json"
    module(
        "tools.paired_bootstrap_hi_csg_r_adapter_v1",
        "--baseline_predictions", *baseline,
        "--adapter_predictions", *adapter,
        "--iterations", 10000,
        "--seed", 2026,
        "--out", output,
        log_name=f"bootstrap_{name}",
    )
    return output

if EXECUTE["statistics"]:
    primary = bootstrap("m3_vs_m0_ft", prediction_paths("M0-FT"), prediction_paths("M3"))
    shuffled = bootstrap(
        "m3_vs_shuffle",
        prediction_paths("M3-shuffle"),
        prediction_paths("M3"),
    )
    topology = bootstrap(
        "m3_vs_m2_seed42",
        prediction_paths("M2", seeds=(42,)),
        prediction_paths("M3", seeds=(42,)),
    )
    global_vector = bootstrap(
        "m3_vs_global_vector_seed42",
        [GLOBAL_VECTOR_PREDICTIONS],
        prediction_paths("M3", seeds=(42,)),
    )
    compare_args = []
    for model in ("M0", "M0-FT", "M3"):
        for seed in SEEDS:
            compare_args += [
                "--run",
                f"{model}={RUN_ROOT / 'final_evaluation' / model / f'seed{seed}' / 'mixed' / 'summary.json'}",
            ]
    compare_args += [
        "--run", f"M2={RUN_ROOT / 'final_evaluation/M2/seed42/mixed/summary.json'}",
        "--bootstrap", f"M3_vs_M0-FT={primary}",
        "--bootstrap", f"M3_vs_shuffle={shuffled}",
        "--bootstrap", f"M3_vs_M2={topology}",
        "--bootstrap", f"M3_vs_global_vector={global_vector}",
    ]
    module(
        "tools.compare_hi_csg_r_adapter_results_v1",
        *compare_args,
        "--out_dir", STATS,
        log_name="final_comparison",
    )
"""
        ),
        markdown("## 11. Итоговые таблицы"),
        code(
            """
def load_summary(model: str, seed: int, split: str):
    return read_json(
        RUN_ROOT / "final_evaluation" / model / f"seed{seed}" / split / "summary.json"
    )

if EXECUTE["statistics"]:
    main_rows = []
    for model, seeds in (("M0", SEEDS), ("M0-FT", SEEDS), ("M2", (42,)), ("M3", SEEDS)):
        summaries = [load_summary(model, seed, "mixed") for seed in seeds]
        domains = {
            name: np.mean([load_summary(model, seed, name)["cer"] for seed in seeds])
            for name in ("cyrillic", "hkr", "school")
        }
        main_rows.append({
            "Model": model,
            "Seeds": len(seeds),
            "CER overall": np.mean([row["cer"] for row in summaries]),
            "CER SD": np.std([row["cer"] for row in summaries], ddof=1) if len(seeds) > 1 else 0,
            "CER Cyrillic": domains["cyrillic"],
            "CER HKR": domains["hkr"],
            "CER School": domains["school"],
            "WER": np.mean([row["wer"] for row in summaries]),
            "Exact": np.mean([row["exact"] for row in summaries]),
        })
    main_table = pd.DataFrame(main_rows)
    historical = read_json(GLOBAL_VECTOR_SUMMARY)
    main_table = pd.concat(
        [
            main_table,
            pd.DataFrame(
                [
                    {
                        "Model": "M1 historical global-vector",
                        "Seeds": 1,
                        "CER overall": historical["cer"],
                        "CER SD": 0.0,
                        "CER Cyrillic": historical["grouped"]["cyrillic_handwriting"]["cer"],
                        "CER HKR": historical["grouped"]["hkr_words"]["cer"],
                        "CER School": historical["grouped"]["school_notebooks_clean"]["cer"],
                        "WER": historical["wer"],
                        "Exact": historical["exact"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    display(main_table.style.format(precision=5))
    main_table.to_csv(STATS / "main_table.csv", index=False)

    primary = read_json(STATS / "m3_vs_m0_ft.json")
    topology = read_json(STATS / "m3_vs_m2_seed42.json")
    shuffled = read_json(STATS / "m3_vs_shuffle.json")
    ablation = pd.DataFrame([
        {"Variant": "Full HI-CSG-R", "CER": primary["adapter_mean_cer"], "Delta": primary["delta_cer"]},
        {"Variant": "Topology-off", "CER": topology["baseline_mean_cer"], "Delta": -topology["delta_cer"]},
        {"Variant": "Shuffled graph", "CER": shuffled["baseline_mean_cer"], "Delta": -shuffled["delta_cer"]},
        {"Variant": "Image-only matched FT", "CER": primary["baseline_mean_cer"], "Delta": 0.0},
    ])
    display(ablation.style.format(precision=5))
    ablation.to_csv(STATS / "ablation_table.csv", index=False)

    seed_table = pd.DataFrame({
        "Seed": SEEDS,
        "M0-FT CER": primary["baseline_cer_by_seed"],
        "M3 CER": primary["adapter_cer_by_seed"],
    })
    seed_table["Absolute Delta"] = seed_table["M3 CER"] - seed_table["M0-FT CER"]
    seed_table["Relative Delta"] = seed_table["Absolute Delta"] / seed_table["M0-FT CER"]
    seed_table["Winner"] = np.where(seed_table["Absolute Delta"] < 0, "M3", "M0-FT")
    display(seed_table.style.format(precision=5))
    seed_table.to_csv(STATS / "seed_table.csv", index=False)
"""
        ),
        code(
            """
if EXECUTE["statistics"]:
    robustness_rows = []
    for model in ("M0-FT", "M3"):
        row = {
            "Model": model,
            "Clean CER": np.mean([load_summary(model, seed, "mixed")["cer"] for seed in SEEDS]),
            "Page-disjoint CER": np.mean([
                load_summary(model, seed, "page_disjoint")["cer"] for seed in SEEDS
            ]),
        }
        for family in ("noise", "low_contrast", "thin_strokes"):
            values = []
            for split in FINAL_MANIFESTS:
                if split.startswith(f"robustness_{family}"):
                    values.extend(load_summary(model, seed, split)["cer"] for seed in SEEDS)
            row[family] = np.mean(values)
        distorted = [
            load_summary(model, seed, split)["cer"]
            for split in FINAL_MANIFESTS if split.startswith("robustness_")
            for seed in SEEDS
        ]
        row["Mean distorted CER"] = np.mean(distorted)
        robustness_rows.append(row)
    robustness_table = pd.DataFrame(robustness_rows)
    display(robustness_table.style.format(precision=5))
    robustness_table.to_csv(STATS / "page_disjoint_robustness_table.csv", index=False)
"""
        ),
        markdown("## 12. Метрики и диагностические визуализации"),
        code(
            """
if EXECUTE["statistics"]:
    primary = read_json(STATS / "m3_vs_m0_ft.json")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    seed_df = pd.DataFrame({
        "seed": SEEDS,
        "M0-FT": primary["baseline_cer_by_seed"],
        "M3": primary["adapter_cer_by_seed"],
    }).set_index("seed")
    seed_df.plot(kind="bar", ax=axes[0], color=["#64748b", "#16a34a"])
    axes[0].set_ylabel("micro-CER")
    axes[0].set_title("Seed stability")
    domain_delta = pd.Series({
        key: value["delta_cer"] for key, value in primary["domains"].items()
    })
    domain_delta.plot(kind="bar", ax=axes[1], color=np.where(domain_delta < 0, "#16a34a", "#dc2626"))
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Delta CER (M3 - M0-FT)")
    axes[1].set_title("Domain effects")
    plt.tight_layout()
    display(figure)

    history = pd.DataFrame(read_jsonl(RUN_ROOT / "m3_full_seed42/history.jsonl"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    history.plot(x="global_epoch", y=["train_fused_ctc_loss", "train_graph_aux_ctc_loss"], ax=axes[0])
    history.plot(x="global_epoch", y=["graph_adapter_grad_norm", "gate_grad_norm"], ax=axes[1])
    axes[2].plot(history["global_epoch"], [row["std"] if row else 0 for row in history["gate"]])
    axes[2].set_title("Gate variability")
    axes[2].set_xlabel("epoch")
    plt.tight_layout()
    display(figure)
"""
        ),
        markdown("## 13. Figure A и Figure B"),
        code(
            """
if EXECUTE["final_report"]:
    figure_a = RUN_ROOT / "final_report" / "figure_a_architecture.png"
    module(
        "tools.make_hi_csg_r_adapter_architecture_figure_v1",
        "--out", figure_a,
        log_name="figure_a",
    )
    display(DisplayImage(filename=figure_a))

    figure_b_dir = RUN_ROOT / "final_report" / "figure_b_alignment"
    module(
        "tools.visualize_xaligned_features_v1",
        "--manifest", ENHANCED["test"],
        "--m0_predictions",
        RUN_ROOT / "final_evaluation/M0-FT/seed42/mixed/predictions.jsonl",
        "--m3_predictions",
        RUN_ROOT / "final_evaluation/M3/seed42/mixed/predictions.jsonl",
        "--out_dir", figure_b_dir,
        log_name="figure_b",
    )
    selection = read_json(figure_b_dir / "selection.json")
    display(DisplayImage(filename=ROOT / selection[0]["image"] if Path(selection[0]["image"]).is_absolute()
                         else figure_b_dir / selection[0]["image"]))
"""
        ),
        markdown("## 14. Final report и статус H4"),
        code(
            """
if EXECUTE["final_report"]:
    module(
        "tools.make_hi_csg_r_adapter_final_report_v1",
        "--comparison", STATS / "comparison.json",
        "--primary_bootstrap", STATS / "m3_vs_m0_ft.json",
        "--shuffle_bootstrap", STATS / "m3_vs_shuffle.json",
        "--topology_bootstrap", STATS / "m3_vs_m2_seed42.json",
        "--out_dir", RUN_ROOT / "final_report",
        log_name="final_report",
    )
    final = read_json(RUN_ROOT / "final_report/final_report.json")
    display(Markdown((RUN_ROOT / "final_report/final_report.md").read_text(encoding="utf-8")))

    method_text = '''## 2.X. Локальное выравнивание HI-CSG-R с временной осью CRNN-CTC

HI-CSG-R преобразуется в 20-мерную последовательность локальных признаков,
выровненную с временными шагами CRNN-CTC. Temporal adapter и quality-aware
residual gate добавляют структурное представление перед существующим BiLSTM.
Вспомогательная graph CTC objective используется только при обучении.

## 3.X. Сравнение image-only и локально структурно усиленной CRNN-CTC

Сравниваются canonical image-only +10k, matched M0-FT, topology-off M2,
full M3 и matched shuffled-graph control. Используются seeds 42/43/44,
blank penalty -0.4 и выбор checkpoint только по validation micro-CER.
'''
    conclusion = (
        "H4 получает подтверждение в ограниченной форме."
        if final["hypothesis_h4"] == "supported"
        else "H4 остаётся поисковой; устойчивого превосходства над matched image-only fine-tuning не показано."
    )
    result_text = f"## 4.X. Локальное слияние HI-CSG-R с CRNN-CTC\\n\\n{conclusion}\\n"
    (RUN_ROOT / "final_report/method_and_results_sections_ru.md").write_text(
        method_text + "\\n" + result_text,
        encoding="utf-8",
    )
"""
        ),
        markdown(
            """
## 15. Финальный checklist

После полного выполнения проверьте:

- `input_audit/report.json == PASS`;
- feature summaries для всех основных и дополнительных manifests имеют `PASS`;
- visual selection содержит 30 образцов;
- `smoke_gate.json == PASS`;
- `validation_gate.json == PASS` либо ветка остановлена;
- freeze registry создан до test;
- test выполнен ровно один раз;
- correct graph сравнён с matched shuffled graph;
- M3 сравнён с topology-off M2;
- primary paired CI использует одинаковые sample IDs;
- сформированы четыре таблицы, две figures и final report;
- H4 классифицирована без изменения архитектуры после seed 42.
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (hi-csg-r)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "hi_csg_r_protocol": {
                "version": "1.0",
                "blank_logit_penalty": -0.4,
                "seeds": [42, 43, 44],
                "test_requires_freeze": True,
            },
        }
    )
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="notebooks/htr_adapter_v1_full_experiment.ipynb",
    )
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output)
    print(output)


if __name__ == "__main__":
    main()
