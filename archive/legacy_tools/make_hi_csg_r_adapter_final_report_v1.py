from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def model_by_name(comparison: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((row for row in comparison["models"] if row["model"] == name), None)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def negative_validation_report(args: argparse.Namespace) -> None:
    required = {
        name: getattr(args, name)
        for name in (
            "validation_gate",
            "m0_summary",
            "m3_summary",
            "shuffle_summary",
            "m0_domain_summary",
            "m3_domain_summary",
            "m3_history",
        )
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Validation-stop report is missing arguments: " + ", ".join(missing)
        )

    gate = read_json(args.validation_gate)
    if gate.get("status") != "STOP":
        raise ValueError("--validation_gate mode requires a recorded STOP")
    m0 = read_json(args.m0_summary)
    m3 = read_json(args.m3_summary)
    shuffled = read_json(args.shuffle_summary)
    m0_domains = read_json(args.m0_domain_summary)
    m3_domains = read_json(args.m3_domain_summary)
    history = [
        json.loads(line)
        for line in Path(args.m3_history).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    shared_domains = sorted(set(m0_domains) & set(m3_domains))
    domain_rows = [
        {
            "domain": domain,
            "m0_ft_cer": float(m0_domains[domain]["cer"]),
            "m3_cer": float(m3_domains[domain]["cer"]),
            "delta_cer": float(m3_domains[domain]["cer"])
            - float(m0_domains[domain]["cer"]),
            "m0_ft_exact": float(m0_domains[domain]["exact"]),
            "m3_exact": float(m3_domains[domain]["exact"]),
        }
        for domain in shared_domains
    ]
    overall_rows = [
        {
            "model": name,
            "cer": float(summary["cer"]),
            "wer": float(summary["wer"]),
            "exact": float(summary["exact"]),
            "macro_cer": float(summary["macro_cer"]),
        }
        for name, summary in (
            ("M0-FT", m0),
            ("M3", m3),
            ("M3-shuffle", shuffled),
        )
    ]
    joint = [row for row in history if row.get("stage") == "joint"]
    diagnostics = {
        "correct_vs_shuffle_delta_cer": float(shuffled["cer"]) - float(m3["cer"]),
        "gate": m3.get("gate"),
        "max_graph_adapter_grad_norm": max(
            (float(row.get("graph_adapter_grad_norm", 0.0)) for row in joint),
            default=0.0,
        ),
        "max_gate_grad_norm": max(
            (float(row.get("gate_grad_norm", 0.0)) for row in joint),
            default=0.0,
        ),
        "graph_aux_cer": m3.get("graph_aux_cer"),
    }
    report = {
        "report_type": "seed42_validation_stop",
        "status": "complete_negative_result",
        "hypothesis_h4": "exploratory",
        "test_evaluated": False,
        "created_at": datetime.now(UTC).isoformat(),
        "validation_gate": gate,
        "overall_validation": overall_rows,
        "domain_validation": domain_rows,
        "diagnostics": diagnostics,
        "not_run_by_protocol": [
            "M2 seed42",
            "M0-FT seeds 43 and 44",
            "M3 seeds 43 and 44",
            "final test/domain/page-disjoint/robustness evaluation",
            "paired test bootstrap and Holm analysis",
        ],
        "conclusion_ru": (
            "H4 остаётся поисковой: локальное слияние HI-CSG-R не "
            "продемонстрировало превосходства над matched image-only fine-tuning "
            "на seed-42 validation. Согласно заранее заданным stopping rules "
            "дальнейшие модели и test не запускались."
        ),
    }
    if args.figure_a:
        figure_a = Path(args.figure_a)
        if not figure_a.exists():
            raise FileNotFoundError(figure_a)
        report["figure_a"] = str(figure_a.resolve())

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "validation_overall_table.csv", overall_rows)
    write_csv(output / "validation_domain_table.csv", domain_rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    model_names = [row["model"] for row in overall_rows]
    cer_values = [row["cer"] for row in overall_rows]
    colors = ["#64748b", "#dc2626", "#f59e0b"]
    axes[0].bar(model_names, cer_values, color=colors)
    axes[0].set_ylabel("Validation micro-CER")
    axes[0].set_title("Seed-42 validation")
    axes[0].set_ylim(0, max(cer_values) * 1.15)
    for index, value in enumerate(cer_values):
        axes[0].text(index, value, f"{value:.4f}", ha="center", va="bottom")
    domain_names = [row["domain"] for row in domain_rows]
    domain_deltas = [row["delta_cer"] for row in domain_rows]
    axes[1].bar(
        domain_names,
        domain_deltas,
        color=["#16a34a" if value < 0 else "#dc2626" for value in domain_deltas],
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Delta CER (M3 - M0-FT)")
    axes[1].set_title("Core-domain effects")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure_path = output / "figure_b_seed42_validation.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report["figure_b"] = str(figure_path.resolve())

    (output / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# CRNN-CTC + x-aligned HI-CSG-R: финальный отрицательный отчёт",
        "",
        "Статус H4: **exploratory**",
        "",
        "## Решение протокола",
        "",
        (
            "Заранее зафиксированный validation gate на seed 42 вернул **STOP**. "
            "Это финальный результат ветки adapter; test set не использовался."
        ),
        "",
        "## Результаты validation",
        "",
        "| model | CER | WER | Exact | Macro-CER |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['model']} | {row['cer']:.6f} | {row['wer']:.6f} | "
        f"{row['exact']:.6f} | {row['macro_cer']:.6f} |"
        for row in overall_rows
    )
    lines.extend(
        [
            "",
            (
                f"Relative CER improvement M3 относительно M0-FT: "
                f"**{100 * float(gate['relative_improvement']):.3f}%** "
                f"(ухудшение на "
                f"**{-100 * float(gate['relative_improvement']):.3f}%**)."
            ),
            "",
            "## Основные домены",
            "",
            "| domain | M0-FT CER | M3 CER | Delta CER | M0-FT Exact | M3 Exact |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['domain']} | {row['m0_ft_cer']:.6f} | {row['m3_cer']:.6f} | "
        f"{row['delta_cer']:+.6f} | {row['m0_ft_exact']:.6f} | "
        f"{row['m3_exact']:.6f} |"
        for row in domain_rows
    )
    lines.extend(
        [
            "",
            "## Критерии остановки",
            "",
            "| criterion | passed |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | **{value}** |"
        for name, value in gate.get("conditions", {}).items()
    )
    lines.extend(
        [
            "",
            "## Контроли и диагностика",
            "",
            (
                f"- Правильный graph CER лучше shuffled на "
                f"`{diagnostics['correct_vs_shuffle_delta_cer']:.6f}`, однако M3 "
                "не превосходит M0-FT."
            ),
            f"- Стандартное отклонение gate: `{float((m3.get('gate') or {}).get('std', 0)):.6f}`.",
            (
                f"- Максимальная норма градиента graph adapter при joint training: "
                f"`{diagnostics['max_graph_adapter_grad_norm']:.6f}`."
            ),
            "- Topology-off M2 не обучался, поскольку pre-M2 gate не пройден.",
            "",
        ]
    )
    if args.figure_a:
        lines.extend(
            [
                f"![Adapter architecture]({Path(args.figure_a).name})",
                "",
            ]
        )
    lines.extend(
        [
            "![Seed-42 validation](figure_b_seed42_validation.png)",
            "",
            "## Намеренно не запускалось",
            "",
        ]
    )
    lines.extend(f"- {name}" for name in report["not_run_by_protocol"])
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            report["conclusion_ru"],
            "",
            "Для этого вывода не использовалась ни одна test-derived метрика.",
        ]
    )
    (output / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    method_and_results = """# Фрагменты для текста работы

## 2.X. Локальное выравнивание HI-CSG-R с временной осью CRNN-CTC

HI-CSG-R преобразуется в 20-мерную последовательность локальных признаков,
выровненную с временными шагами CRNN-CTC. Temporal adapter и quality-aware
residual gate добавляют структурное представление перед существующим BiLSTM.
Вспомогательная graph CTC objective используется только при обучении.

## 3.X. Сравнение image-only и локально структурно усиленной CRNN-CTC

На seed 42 сравнивались matched image-only fine-tuning (M0-FT), полный
x-aligned HI-CSG-R adapter (M3) и matched shuffled-graph control. Использовались
фиксированный blank penalty -0.4 и выбор checkpoint только по validation
micro-CER. Переход к M2 и дополнительным seeds был разрешён только при успешном
прохождении заранее зафиксированного validation gate.

## 4.X. Локальное слияние HI-CSG-R с CRNN-CTC

M0-FT получил validation CER 0.079537, а M3 — 0.082196, что соответствует
ухудшению на 3.342% relative. M3 ухудшил CER во всех трёх основных validation
доменах. Правильный граф был немного лучше matched shuffled graph, а gate и
градиенты не коллапсировали, однако основной критерий превосходства над M0-FT и
доменный критерий не были выполнены.

H4 остаётся поисковой; локальное структурное слияние не продемонстрировало
устойчивого превосходства над matched image-only fine-tuning. Согласно frozen
stopping rules M2, seeds 43/44 и test evaluation не запускались.
"""
    (output / "method_and_results_sections_ru.md").write_text(
        method_and_results, encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison")
    parser.add_argument("--primary_bootstrap")
    parser.add_argument("--shuffle_bootstrap")
    parser.add_argument("--topology_bootstrap")
    parser.add_argument("--validation_gate")
    parser.add_argument("--m0_summary")
    parser.add_argument("--m3_summary")
    parser.add_argument("--shuffle_summary")
    parser.add_argument("--m0_domain_summary")
    parser.add_argument("--m3_domain_summary")
    parser.add_argument("--m3_history")
    parser.add_argument("--figure_a")
    parser.add_argument("--m0_name", default="M0-FT")
    parser.add_argument("--m3_name", default="M3")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    if args.validation_gate:
        negative_validation_report(args)
        return
    required = (
        args.comparison,
        args.primary_bootstrap,
        args.shuffle_bootstrap,
        args.topology_bootstrap,
    )
    if not all(required):
        raise ValueError(
            "Positive report requires comparison and all three bootstrap inputs"
        )
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    bootstrap = json.loads(Path(args.primary_bootstrap).read_text(encoding="utf-8"))
    shuffle = json.loads(Path(args.shuffle_bootstrap).read_text(encoding="utf-8"))
    topology = json.loads(Path(args.topology_bootstrap).read_text(encoding="utf-8"))
    baseline = model_by_name(comparison, args.m0_name)
    adapter = model_by_name(comparison, args.m3_name)
    if baseline is None or adapter is None:
        raise ValueError("Comparison must contain M0-FT and M3 model rows")

    improved_seeds = sum(
        adapter_value < baseline_value
        for baseline_value, adapter_value in zip(
            bootstrap["baseline_cer_by_seed"],
            bootstrap["adapter_cer_by_seed"],
            strict=True,
        )
    )
    max_domain_degradation = max(
        (float(value["delta_cer"]) for value in bootstrap["domains"].values()),
        default=0.0,
    )
    minimally_positive = (
        adapter["cer_mean"] < baseline["cer_mean"]
        and improved_seeds >= 2
        and bootstrap["ci95"][1] < 0
        and max_domain_degradation <= 0.005
        and float(shuffle["delta_cer"]) < 0
        and float(topology["delta_cer"]) < 0
    )
    classification = "supported" if minimally_positive else "exploratory"
    report = {
        "hypothesis_h4": classification,
        "criteria": {
            "mean_m3_better": adapter["cer_mean"] < baseline["cer_mean"],
            "improved_seeds": improved_seeds,
            "paired_ci_below_zero": bootstrap["ci95"][1] < 0,
            "max_domain_degradation": max_domain_degradation,
            "correct_better_than_shuffled": float(shuffle["delta_cer"]) < 0,
            "full_better_than_topology_off": float(topology["delta_cer"]) < 0,
        },
        "baseline": baseline,
        "adapter": adapter,
        "primary_bootstrap": bootstrap,
        "shuffle_bootstrap": shuffle,
        "topology_bootstrap": topology,
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = [
        "# CRNN-CTC + x-aligned HI-CSG-R: Final Report",
        "",
        f"H4 status: **{classification}**",
        "",
        f"- M0-FT CER: {baseline['cer_mean']:.6f} ± {baseline['cer_sd']:.6f}",
        f"- M3 CER: {adapter['cer_mean']:.6f} ± {adapter['cer_sd']:.6f}",
        f"- Paired ΔCER: {bootstrap['delta_cer']:.6f}",
        f"- 95% CI: [{bootstrap['ci95'][0]:.6f}, {bootstrap['ci95'][1]:.6f}]",
        f"- Better seeds: {improved_seeds}/{bootstrap['seeds']}",
        "",
        f"- Correct vs shuffled ΔCER: {shuffle['delta_cer']:.6f}",
        f"- Full vs topology-off ΔCER: {topology['delta_cer']:.6f}",
        "",
        "The conclusion is generated only from the preregistered minimum criteria.",
    ]
    (output / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
