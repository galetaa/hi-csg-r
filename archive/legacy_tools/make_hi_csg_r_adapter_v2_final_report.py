from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hi_csg_r_matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_decisions(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(root.glob("development/**/dev_decision.json")):
        value = read_json(path)
        if value:
            results.append({"path": str(path), **value})
    return results


def status_from_artifacts(
    preflight: dict[str, Any] | None,
    smoke: dict[str, Any] | None,
    selection: dict[str, Any] | None,
    holdout: dict[str, Any] | None,
    decisions: list[dict[str, Any]],
    final_seed_summaries: list[dict[str, Any]],
    final_statistics: dict[str, Any] | None,
) -> tuple[str, str]:
    if preflight and preflight.get("decision", {}).get("status") == "STOP":
        return "complete_negative_preflight", "not_supported"
    if smoke and smoke.get("status") != "PASS":
        return "blocked_technical_smoke", "not_evaluated"
    if holdout:
        if holdout.get("status") != "PASS":
            return "complete_negative_holdout", "not_supported"
        if len(final_seed_summaries) < 3:
            return "holdout_pass_final_pending", "partial"
        if final_statistics is None:
            return "final_training_complete_test_pending", "not_evaluated"
        hypothesis = {
            "clear_superiority": "supported",
            "minimal_or_partial_support": "partial",
            "not_supported": "not_supported",
        }.get(str(final_statistics.get("status")), "not_evaluated")
        return "final_evaluation_complete", hypothesis
    if selection:
        if selection.get("status") != "PASS":
            return "complete_negative_development", "not_supported"
        return "development_selected_holdout_pending", "not_evaluated"
    decision_runs = {Path(row["path"]).parents[1].name for row in decisions}
    v2_1_finished = any(name.startswith("v2_1_") for name in decision_runs)
    v2_2_finished = any(name.startswith("v2_2_") for name in decision_runs)
    v2_2_allowed = bool(
        preflight
        and preflight.get("decision", {}).get("allow_v2_2", False)
    )
    development_exhausted = v2_1_finished and (
        v2_2_finished or not v2_2_allowed
    )
    if (
        development_exhausted
        and decisions
        and all(row.get("status") == "STOP" for row in decisions)
    ):
        return "complete_negative_development", "not_supported"
    return "implementation_or_development_in_progress", "not_evaluated"


def metric_table(decisions: list[dict[str, Any]]) -> list[str]:
    if not decisions:
        return ["Development evaluations отсутствуют."]
    lines = [
        "| Run | Gate | Baseline CER | Correct CER | Shuffle CER | Zero CER | "
        "Relative improvement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decisions:
        name = Path(row["path"]).parents[1].name
        lines.append(
            f"| {name} | {row['status']} | {row['baseline']['cer']:.6f} | "
            f"{row['correct']['cer']:.6f} | {row['shuffle']['cer']:.6f} | "
            f"{row['zero']['cer']:.6f} | "
            f"{row['relative_cer_improvement']:.3%} |"
        )
    return lines


def architecture_figure(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 5)
    axis.axis("off")
    boxes = [
        (0.3, 3.2, 2.0, 1.0, "Image\n(grayscale)", "#e5e7eb"),
        (3.0, 3.2, 2.4, 1.0, "Frozen CRNN\nCNN + BiLSTM", "#dbeafe"),
        (6.1, 3.2, 2.1, 1.0, "Baseline\nCTC logits", "#dbeafe"),
        (0.3, 0.8, 2.0, 1.0, "HI-CSG-R\nx-aligned", "#dcfce7"),
        (3.0, 0.8, 2.4, 1.0, "Strict mask +\nmultiscale adapter", "#dcfce7"),
        (6.1, 0.8, 2.1, 1.0, "Uncertainty /\nrisk gate", "#fef3c7"),
        (9.0, 0.8, 2.0, 1.0, "Bounded\ncorrection", "#fee2e2"),
        (11.7, 2.0, 2.0, 1.0, "Final logits\n+ CTC", "#e9d5ff"),
    ]
    for x, y, width, height, label, color in boxes:
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=color,
                edgecolor="#334155",
                linewidth=1.2,
            )
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    arrows = [
        ((2.3, 3.7), (3.0, 3.7)),
        ((5.4, 3.7), (6.1, 3.7)),
        ((2.3, 1.3), (3.0, 1.3)),
        ((5.4, 1.3), (6.1, 1.3)),
        ((8.2, 1.3), (9.0, 1.3)),
        ((8.2, 3.7), (12.0, 3.0)),
        ((11.0, 1.3), (12.0, 2.0)),
        ((5.4, 3.4), (6.4, 1.8)),
    ]
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.2,
                color="#334155",
            )
        )
    axis.text(
        9.2,
        0.35,
        r"$Z_{final}=Z_{base}+\alpha\cdot gate\cdot\Delta Z$",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def results_figure(
    path: Path,
    decisions: list[dict[str, Any]],
    final_statistics: dict[str, Any] | None,
) -> None:
    if not decisions and not final_statistics:
        return
    if final_statistics:
        primary_name = str(final_statistics["primary"])
        primary = final_statistics["comparisons"][primary_name]
        baseline = np.asarray(primary["baseline_cer_by_seed"], dtype=float)
        adapter = np.asarray(primary["adapter_cer_by_seed"], dtype=float)
        seeds = np.arange(len(baseline))
        width = 0.36
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(seeds - width / 2, baseline, width, label="M0-FT", color="#64748b")
        axis.bar(seeds + width / 2, adapter, width, label="V2 correct", color="#0f766e")
        axis.set_xticks(seeds, ("42", "43", "44")[: len(seeds)])
        axis.set_xlabel("Seed")
        axis.set_ylabel("Final test micro-CER")
        axis.set_title(
            "Final seeds: mean "
            f"{baseline.mean():.4f} vs {adapter.mean():.4f}"
        )
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    names: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for row in decisions:
        short = Path(row["path"]).parents[1].name.replace("_dev_p05_seed42", "")
        for label, key, color in (
            ("B0", "baseline", "#64748b"),
            ("correct", "correct", "#0f766e"),
            ("shuffle", "shuffle", "#d97706"),
        ):
            names.append(f"{short}\n{label}")
            values.append(float(row[key]["cer"]))
            colors.append(color)
    figure, axis = plt.subplots(figsize=(max(8, len(values) * 1.25), 5))
    axis.bar(names, values, color=colors)
    axis.set_ylabel("Development micro-CER")
    axis.set_title("B0-dev-v2 vs late correction controls")
    axis.grid(axis="y", alpha=0.2)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def intervention_figure(path: Path, root: Path) -> None:
    candidates = sorted(root.glob("development/**/correct/frame_diagnostics.npz"))
    if not candidates:
        return
    with np.load(candidates[-1]) as frames:
        uncertainty = np.asarray(frames["uncertainty"])
        gate = np.asarray(frames["gate"])
        correction = np.asarray(frames["correction_norm"])
        nonempty = np.asarray(frames["nonempty"]).astype(bool)
        risk = np.asarray(frames["risk"])
        sample_lengths = (
            np.asarray(frames["sample_lengths"], dtype=np.int64)
            if "sample_lengths" in frames
            else np.asarray([])
        )
        sample_ids = (
            np.asarray(frames["sample_ids"])
            if "sample_ids" in frames
            else np.asarray([])
        )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(uncertainty, bins=40, color="#2563eb", alpha=0.85)
    axes[0].set_title("Frozen visual uncertainty")
    axes[1].scatter(
        uncertainty[nonempty][:: max(len(uncertainty[nonempty]) // 5000, 1)],
        correction[nonempty][:: max(len(correction[nonempty]) // 5000, 1)],
        s=3,
        alpha=0.25,
        color="#dc2626",
    )
    axes[1].set_xlabel("Uncertainty")
    axes[1].set_ylabel("Correction L2")
    axes[1].set_title("All non-empty frames")
    if sample_lengths.size:
        offsets = np.concatenate(([0], np.cumsum(sample_lengths)))
        scores = [
            float(correction[offsets[index] : offsets[index + 1]].mean())
            for index in range(len(sample_lengths))
        ]
        picked = int(np.argmax(scores))
        start, stop = int(offsets[picked]), int(offsets[picked + 1])
        x = np.arange(stop - start)
        axes[2].plot(x, uncertainty[start:stop], label="uncertainty")
        axes[2].plot(x, gate[start:stop], label="gate")
        axes[2].plot(x, risk[start:stop], label="risk")
        normalized_correction = correction[start:stop] / max(
            float(correction[start:stop].max()),
            1e-12,
        )
        axes[2].plot(x, normalized_correction, label="correction/max")
        axes[2].fill_between(
            x,
            0,
            nonempty[start:stop].astype(float),
            color="#94a3b8",
            alpha=0.15,
            label="graph presence",
        )
        sample_label = str(sample_ids[picked]) if sample_ids.size else str(picked)
        axes[2].set_title(f"Highest-intervention sample: {sample_label}")
        axes[2].legend(fontsize=7, ncol=2)
    else:
        axes[2].hist(gate[nonempty], bins=40, color="#0f766e", alpha=0.85)
        axes[2].set_title("Gate on non-empty bins")
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_human_reports(
    output: Path,
    report: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> None:
    status = report["status"]
    h4 = report["hypothesis_h4_v2"]
    final_lines = [
        "# HI-CSG-R Late Correction v2: результаты",
        "",
        f"**Execution status:** `{status}`",
        "",
        f"**H4-v2:** `{h4}`",
        "",
        "## Зафиксированные результаты v1",
        "",
        (
            "V1 остается завершенным отрицательным экспериментом: "
            "`M0-FT CER=0.079537`, `M3 CER=0.082196`, "
            "`M3-shuffle CER=0.083004`. V2 этот вывод не перезаписывает."
        ),
        "",
        "## V2 technical evidence",
        "",
        f"- preflight: `{report['preflight_status']}`",
        (
            "- fixed decode/correction: "
            f"`blank penalty={report['preflight_decision'].get('selected_blank_logit_penalty')}`, "
            f"`alpha_max={report['preflight_decision'].get('selected_alpha_max')}`"
        ),
        (
            "- preflight D2/D3 CER gains: "
            f"`{report['preflight_decision'].get('d2_absolute_cer_gain', 0.0):+.6f}` / "
            f"`{report['preflight_decision'].get('d3_absolute_cer_gain', 0.0):+.6f}`"
        ),
        f"- split audit: `{report['split_status']}`",
        (
            "- independent split: "
            f"`train={report['split_counts'].get('train')}`, "
            f"`dev={report['split_counts'].get('dev')}`, "
            f"`holdout={report['split_counts'].get('holdout')}`"
        ),
        (
            "- split overlap counts: "
            f"`{report['split_overlap_total']}` (sample/path/group/SHA1)"
        ),
        f"- smoke gate: `{report['smoke_status']}`",
        f"- fresh B0-dev: `{report['b0_status']}`",
        (
            "- B0 best dev micro-CER / epoch: "
            f"`{report['b0_summary'].get('best_dev_micro_cer', float('nan')):.6f}` / "
            f"`{report['b0_summary'].get('epochs_completed', 0)}` completed epochs"
        ),
        f"- holdout evaluated: `{report['holdout_evaluated']}`",
        f"- final test evaluated: `{report['test_evaluated']}`",
        (
            "- image-only parameters: "
            f"`{report['parameter_budget']['image_only_parameters']}`"
        ),
        (
            "- v2 trainable parameters: "
            f"`{report['parameter_budget']['v2_trainable_parameters']}` "
            f"(`{report['parameter_budget']['relative_increase_percent']:.3f}%`)"
        ),
        "",
        "## Development metrics",
        "",
        *metric_table(decisions),
    ]
    for row in decisions:
        name = Path(row["path"]).parents[1].name
        final_lines.extend(
            [
                "",
                f"### {name}",
                "",
                (
                    "- absolute / relative CER change vs B0: "
                    f"`{row['absolute_cer_delta']:+.6f}` / "
                    f"`{-row['relative_cer_improvement']:+.3%}` "
                    "(adapter minus baseline)"
                ),
                (
                    "- correct minus shuffle CER: "
                    f"`{row['correct_vs_shuffle_cer']:+.6f}`"
                ),
                (
                    "- domain CER deltas (Cyrillic/HKR/School): "
                    f"`{row['domain_cer_deltas'].get('cyrillic', 0.0):+.6f}` / "
                    f"`{row['domain_cer_deltas'].get('hkr', 0.0):+.6f}` / "
                    f"`{row['domain_cer_deltas'].get('school', 0.0):+.6f}`"
                ),
                (
                    "- alpha / empty correction max: "
                    f"`{row['alpha']:.9f}` / `{row['empty_correction_max']:.3g}`"
                ),
                (
                    "- gate conditions: "
                    + ", ".join(
                        f"`{key}={value}`"
                        for key, value in row["conditions"].items()
                    )
                ),
            ]
        )
    development_statistics = report.get("development_statistics", {})
    if development_statistics:
        final_lines.extend(["", "## Development paired statistics", ""])
        for name, values in development_statistics.items():
            final_lines.extend(
                [
                    f"### {name}",
                    "",
                    (
                        "- delta CER / relative delta: "
                        f"`{values['delta_cer']:+.6f}` / "
                        f"`{values['relative_delta']:+.3%}`"
                    ),
                    (
                        "- paired bootstrap CI95 / p: "
                        f"`[{values['ci95'][0]:+.6f}, {values['ci95'][1]:+.6f}]` / "
                        f"`{values['p_two_sided']:.6f}`"
                    ),
                    (
                        "- WER / Exact delta: "
                        f"`{values['delta_wer']:+.6f}` / "
                        f"`{values['delta_exact']:+.6f}`"
                    ),
                    (
                        "- wins/losses/ties: "
                        f"`{values['wins']}/{values['losses']}/{values['ties']}`"
                    ),
                    "",
                ]
            )
    diagnostics = report.get("development_diagnostics")
    if diagnostics:
        final_lines.extend(
            [
                "## Intervention diagnostics (best-CER development variant)",
                "",
                f"- alpha: `{diagnostics['alpha']:.9f}`",
                (
                    "- gate mean/std/P90/non-empty/empty: "
                    f"`{diagnostics['gate']['mean']:.6f}` / "
                    f"`{diagnostics['gate']['std']:.6f}` / "
                    f"`{diagnostics['gate']['p90']:.6f}` / "
                    f"`{diagnostics['gate']['nonempty']:.6f}` / "
                    f"`{diagnostics['gate']['empty']:.6f}`"
                ),
                (
                    "- intervention / strong intervention / changed prediction: "
                    f"`{diagnostics['intervention']['gate_gt_005_rate']:.3%}` / "
                    f"`{diagnostics['intervention']['gate_gt_015_rate']:.3%}` / "
                    f"`{diagnostics['intervention']['prediction_change_rate']:.3%}`"
                ),
                (
                    "- intervention precision (improves edit distance): "
                    f"`{diagnostics['intervention']['precision']:.3%}`"
                ),
                (
                    "- improved/hurt samples: "
                    f"`{diagnostics['intervention']['improved_samples']}` / "
                    f"`{diagnostics['intervention']['hurt_samples']}`"
                ),
                (
                    "- correction/base L2 ratio / empty correction max: "
                    f"`{diagnostics['correction']['l2_ratio']:.6f}` / "
                    f"`{diagnostics['correction']['empty_max']:.3g}`"
                ),
                (
                    "- visual uncertainty mean/P90: "
                    f"`{diagnostics['uncertainty']['mean']:.6f}` / "
                    f"`{diagnostics['uncertainty']['p90']:.6f}`"
                ),
                (
                    "- structural risk mean/P90: "
                    f"`{diagnostics['risk']['mean']:.6f}` / "
                    f"`{diagnostics['risk']['p90']:.6f}`"
                ),
                "",
            ]
        )
    holdout = report.get("holdout")
    if holdout:
        final_lines.extend(
            [
                "",
                "## One-shot holdout",
                "",
                f"- gate: `{holdout['status']}`",
                f"- baseline CER: `{holdout['baseline']['cer']:.6f}`",
                f"- correct CER: `{holdout['correct']['cer']:.6f}`",
                f"- shuffle CER: `{holdout['shuffle']['cer']:.6f}`",
                (
                    "- relative improvement: "
                    f"`{holdout['relative_cer_improvement']:.3%}`"
                ),
            ]
        )
    final_statistics = report.get("final_statistics")
    if final_statistics:
        primary = final_statistics["comparisons"][final_statistics["primary"]]
        final_lines.extend(
            [
                "",
                "## Final test and paired statistics",
                "",
                "| Seed | M0-FT CER | V2 CER | Delta | Winner |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for seed, baseline, adapter in zip(
            (42, 43, 44),
            primary["baseline_cer_by_seed"],
            primary["adapter_cer_by_seed"],
            strict=False,
        ):
            final_lines.append(
                f"| {seed} | {baseline:.6f} | {adapter:.6f} | "
                f"{adapter - baseline:+.6f} | "
                f"{'V2' if adapter < baseline else 'M0-FT'} |"
            )
        final_lines.extend(
            [
                "",
                f"- mean M0-FT CER: `{primary['baseline_mean_cer']:.6f}`",
                f"- mean V2 CER: `{primary['adapter_mean_cer']:.6f}`",
                f"- delta CER: `{primary['delta_cer']:+.6f}`",
                f"- relative delta: `{primary['relative_delta']:+.3%}`",
                (
                    "- paired bootstrap CI95: "
                    f"`[{primary['ci95'][0]:+.6f}, "
                    f"{primary['ci95'][1]:+.6f}]`"
                ),
                f"- WER delta: `{primary['delta_wer']:+.6f}`",
                f"- Exact delta: `{primary['delta_exact']:+.6f}`",
                (
                    "- wins/losses/ties: "
                    f"`{primary['wins']}/{primary['losses']}/{primary['ties']}`"
                ),
            ]
        )
    final_lines.extend(
        [
            "",
            "## Protected stages",
            "",
            "- `lambda_pres=0.10` repeat: **not run** (no dev PASS candidate).",
            "- independent holdout: **not opened**.",
            "- final seeds 42/43/44: **not run**.",
            "- canonical test/page-disjoint/robustness: **not opened**.",
            "",
            "## Решение",
            "",
            report["conclusion_ru"],
        ]
    )
    (output / "final_results.md").write_text(
        "\n".join(final_lines) + "\n",
        encoding="utf-8",
    )
    dev_lines = [
        "# HI-CSG-R Late Correction v2: development comparison",
        "",
        *metric_table(decisions),
        "",
        (
            "Кандидат выбирается только среди вариантов со статусом `PASS`, "
            "по минимальному development micro-CER. Holdout и test в выборе "
            "не участвуют."
        ),
    ]
    (output / "dev_comparison.md").write_text(
        "\n".join(dev_lines) + "\n",
        encoding="utf-8",
    )
    if report["holdout_evaluated"]:
        holdout_lines = [
            "# HI-CSG-R Late Correction v2: holdout decision",
            "",
            f"**Status:** `{report['holdout']['status']}`",
            "",
            "Holdout был оценён однократно после фиксации кандидата.",
        ]
    else:
        holdout_lines = [
            "# HI-CSG-R Late Correction v2: holdout decision",
            "",
            "**Status:** `NOT_EVALUATED_PROTOCOL_STOP`",
            "",
            (
                "Оба разрешённых development-варианта получили `STOP`. "
                "Независимый holdout не открывался и не использовался для "
                "обучения, выбора checkpoint или изменения конфигурации."
            ),
        ]
    (output / "holdout_decision.md").write_text(
        "\n".join(holdout_lines) + "\n",
        encoding="utf-8",
    )
    limitations = [
        "# Ограничения HI-CSG-R Late Correction v2",
        "",
        "- V1 validation уже использовалась исследовательски; V2 применяет новый "
        "group-aware train/dev/holdout split.",
        "- Structural risk attenuation является фиксированным proxy, а не "
        "вероятностью корректности графа.",
        "- Zero-graph является dependency control, но не fair image-only baseline.",
        "- Научный вывод запрещено делать по smoke, train loss, gate variability "
        "или одному development run.",
        "- Если holdout gate возвращает STOP, canonical test остается закрытым.",
        "- Результат относится к существующему extractor, 20 x-aligned признакам "
        "и зафиксированным русскоязычным доменам.",
    ]
    (output / "limitations.md").write_text(
        "\n".join(limitations) + "\n",
        encoding="utf-8",
    )
    method = [
        "# Фрагменты метода и результатов для научной работы",
        "",
        "## 2.X. Late correction HI-CSG-R",
        "",
        (
            "После отрицательного результата раннего x-aligned fusion v1 "
            "визуальная CRNN-CTC в v2 полностью замораживалась. Нормализованные "
            "локальные признаки HI-CSG-R маскировались после стандартизации, "
            "агрегировались masked pooling с окнами 1, 5 и 9 и преобразовывались "
            "в 128-мерное представление. Остаточная поправка добавлялась "
            "не к hidden sequence, а к baseline CTC logits."
        ),
        "",
        (
            "Вклад графа ограничивался произведением non-empty mask, визуальной "
            "неопределенности, learned gate и bounded alpha. Для V2-2 дополнительно "
            "применялось фиксированное structural-risk attenuation. CNN, BiLSTM "
            "и baseline classifier не получали градиентов."
        ),
        "",
        "## 4.X. Результаты",
        "",
        report["conclusion_ru"],
        "",
        (
            "Результат v1 сохраняется отдельно: раннее слияние было технически "
            "работоспособно и sample-specific, но уступило matched image-only "
            "fine-tuning."
        ),
    ]
    (output / "method_and_results_sections_ru.md").write_text(
        "\n".join(method) + "\n",
        encoding="utf-8",
    )
    failure_source = next(
        iter(sorted((output.parent / "failure_analysis").glob("**/failure_analysis.md"))),
        None,
    )
    if failure_source:
        (output / "failure_analysis.md").write_text(
            failure_source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    else:
        (output / "failure_analysis.md").write_text(
            "# Failure analysis\n\n"
            "Per-sample intervention analysis станет доступен после evaluation "
            "выбранного кандидата. Holdout/test не открываются ради заполнения "
            "этого раздела.\n",
            encoding="utf-8",
        )


def evidence_files(root: Path, output: Path) -> list[Path]:
    patterns = (
        "preflight/*",
        "split_audit/*",
        "feature_audit/*",
        "protocol_freeze/*",
        "smoke/smoke_gate.*",
        "b0_dev_seed42/*",
        "v2_1_dev_p05_seed42/*",
        "v2_2_dev_p05_seed42/*",
        "v2_best_dev_p10_seed42/*",
        "development/**/*.json",
        "holdout/**/*",
        "frozen_final_configs/*",
        "m0_ft_final_seed*/*",
        "final_seed*/*",
        "final_evaluation/**/*",
        "statistical_analysis/*",
        "failure_analysis/**/*",
        "failed_runs/**/invalidation_report.*",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    paths.update(path for path in Path("configs/htr_adapter_v2").glob("*.yaml"))
    paths.update(
        path
        for path in Path("data/experiments/htr_adapter_v2").rglob("*")
        if path.is_file()
    )
    paths.update(
        Path("docs").glob(
            "crnn_ctc_hi_csg_r_late_correction_protocol_v2*.md"
        )
    )
    paths.update(path for path in output.glob("*.md") if path.is_file())
    paths.update(path for path in output.glob("*.png") if path.is_file())
    if (output / "final_report.json").exists():
        paths.add(output / "final_report.json")
    return sorted(path.resolve() for path in paths if path.exists())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/htr_adapter_v2")
    parser.add_argument("--out_dir", default="outputs/htr_adapter_v2/final_report")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    preflight = read_json(root / "preflight/preflight_report.json")
    split = read_json(root / "split_audit/split_audit.json")
    smoke = read_json(root / "smoke/smoke_gate.json")
    smoke_runtime = read_json(root / "smoke/subset_128/runtime_metadata.json")
    b0 = read_json(root / "b0_dev_seed42/train_summary.json")
    selection = read_json(root / "development/selected_candidate.json")
    holdout = read_json(root / "holdout/decision/holdout_decision.json")
    decisions = collect_decisions(root)
    final_seed_summaries = [
        value
        for path in sorted(root.glob("final_seed*/train_summary.json"))
        if (value := read_json(path))
    ]
    final_statistics = read_json(
        root / "statistical_analysis/final_statistics.json"
    )
    status, hypothesis = status_from_artifacts(
        preflight,
        smoke,
        selection,
        holdout,
        decisions,
        final_seed_summaries,
        final_statistics,
    )
    test_evaluated = any(root.glob("final_evaluation/test/**/summary.json"))
    if status == "complete_negative_holdout":
        conclusion = (
            "H4-v2 не подтверждена: выбранная bounded late correction не прошла "
            "независимый holdout gate. Согласно frozen protocol final seeds и "
            "canonical test не запускались."
        )
    elif status == "complete_negative_development":
        conclusion = (
            "H4-v2 не подтверждена. Оба разрешенных development-варианта "
            "не прошли заранее установленный dev gate: снижение CER было "
            "меньше 1%, а correct graph не превзошел matched shuffle. "
            "Согласно frozen protocol p10, holdout, final seeds и test "
            "не запускались. Отрицательный вывод v1 сохранен отдельно."
        )
    elif status == "final_evaluation_complete":
        conclusion = (
            "Финальные метрики и paired statistics собраны. H4-v2 "
            f"классифицирована как `{hypothesis}` по заранее установленным "
            "критериям."
        )
    else:
        conclusion = (
            "Эксперимент v2 еще не достиг научного decision gate. Текущие "
            "технические результаты не подтверждают и не опровергают H4-v2."
        )
    split_overlaps = split.get("overlaps", {}) if split else {}
    split_overlap_total = sum(
        int(value)
        for category in split_overlaps.values()
        for value in category.values()
    )
    development_statistics: dict[str, dict[str, Any]] = {}
    v2_vs_b0 = read_json(
        root / "statistical_analysis/dev_v2_2_vs_b0_bootstrap.json"
    )
    if v2_vs_b0:
        development_statistics["V2-2 correct vs B0"] = v2_vs_b0
    correct_vs_shuffle = read_json(
        root
        / "statistical_analysis/dev_v2_2_correct_vs_shuffle_bootstrap.json"
    )
    if correct_vs_shuffle:
        development_statistics["V2-2 correct vs shuffle"] = correct_vs_shuffle
    development_diagnostics = read_json(
        root / "development/v2_2_dev_p05_seed42/correct/summary.json"
    )
    failure_analysis = read_json(root / "failure_analysis/failure_analysis.json")
    report = {
        "protocol": "crnn_ctc_hi_csg_r_late_correction_protocol_v2",
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "hypothesis_h4_v2": hypothesis,
        "preflight_status": (
            preflight.get("decision", {}).get("status") if preflight else "MISSING"
        ),
        "preflight_decision": preflight.get("decision", {}) if preflight else {},
        "split_status": split.get("status") if split else "MISSING",
        "split_counts": split.get("counts", {}) if split else {},
        "split_domain_counts": split.get("domain_counts", {}) if split else {},
        "split_overlap_total": split_overlap_total,
        "smoke_status": smoke.get("status") if smoke else "MISSING",
        "b0_status": "COMPLETE" if b0 else "PENDING",
        "b0_summary": b0 or {},
        "holdout_evaluated": holdout is not None,
        "test_evaluated": test_evaluated,
        "development_decisions": decisions,
        "development_statistics": development_statistics,
        "development_diagnostics": development_diagnostics,
        "failure_analysis": failure_analysis,
        "selection": selection,
        "holdout": holdout,
        "final_seed_summaries": final_seed_summaries,
        "final_statistics": final_statistics,
        "conclusion_ru": conclusion,
        "v1_preserved": True,
        "v1_conclusion": "negative",
        "parameter_budget": {
            "image_only_parameters": (
                int(smoke_runtime["total_parameters"])
                - int(smoke_runtime["v2_module_parameters"])
                if smoke_runtime
                else None
            ),
            "v2_trainable_parameters": (
                int(smoke_runtime["v2_module_parameters"])
                if smoke_runtime
                else None
            ),
            "relative_increase_percent": (
                100.0
                * int(smoke_runtime["v2_module_parameters"])
                / max(
                    int(smoke_runtime["total_parameters"])
                    - int(smoke_runtime["v2_module_parameters"]),
                    1,
                )
                if smoke_runtime
                else 0.0
            ),
        },
    }
    architecture_figure(output / "figure_a_architecture.png")
    intervention_figure(output / "figure_b_intervention.png", root)
    results_figure(
        output / "figure_c_results.png",
        decisions,
        final_statistics,
    )
    source_figure_d = next(
        iter(sorted((root / "failure_analysis").glob("**/figure_d_helps_hurts.png"))),
        None,
    )
    if source_figure_d:
        (output / "figure_d_helps_hurts.png").write_bytes(
            source_figure_d.read_bytes()
        )
    write_human_reports(output, report, decisions)
    (output / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files = evidence_files(root, output)
    evidence = [
        {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    (output / "final_evidence_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in evidence),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
