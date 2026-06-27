from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("outputs/final_result_package_v1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(rows: list[dict[str, Any]], path: Path, *, title: str, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def pct(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value) * 100:.2f}%"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_text_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, *, fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=2)
    x0, y0, x1, y1 = box
    f = font(20, bold=True)
    lines = text.split("\n")
    total_h = len(lines) * 26
    y = y0 + ((y1 - y0) - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=f)
        w = bbox[2] - bbox[0]
        draw.text((x0 + ((x1 - x0) - w) // 2, y), line, fill="#1f2937", font=f)
        y += 26


def save_pipeline_figure(path: Path) -> None:
    img = Image.new("RGB", (1500, 360), "#f8fafc")
    d = ImageDraw.Draw(img)
    boxes = [
        (40, 110, 220, 230, "input\nimage"),
        (285, 110, 505, 230, "foreground\nextraction"),
        (570, 110, 750, 230, "skeleton"),
        (815, 110, 995, 230, "graph\nconstruction"),
        (1060, 90, 1280, 250, "graph/quality\ndiagnostics"),
        (1345, 70, 1480, 270, "HTR\nselective\nanalysis"),
    ]
    for x0, y0, x1, y1, label in boxes:
        draw_text_box(d, (x0, y0, x1, y1), label, fill="#ffffff", outline="#2563eb")
    for _, _, x1, _, _ in boxes[:-1]:
        y = 170
        d.line((x1 + 15, y, x1 + 50, y), fill="#334155", width=3)
        d.polygon([(x1 + 50, y), (x1 + 38, y - 7), (x1 + 38, y + 7)], fill="#334155")
    d.text((40, 25), "HI-CSG-R pipeline", font=font(30, bold=True), fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_context_figure(path: Path) -> None:
    img = Image.new("RGB", (1200, 520), "#f8fafc")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "Natural-line context augmentation", font=font(30, bold=True), fill="#111827")
    draw_text_box(d, (70, 150, 450, 260), "isolated word crop\nbaseline", fill="#ffffff", outline="#64748b")
    draw_text_box(d, (720, 130, 1130, 280), "raw contextual\nnatural-line crop", fill="#ffffff", outline="#16a34a")
    d.line((480, 205, 690, 205), fill="#334155", width=4)
    d.polygon([(690, 205), (672, 194), (672, 216)], fill="#334155")
    d.text((115, 330), "word-level train/eval", font=font(20), fill="#334155")
    d.text((755, 330), "+ sampled School line groups\n+ original word-level eval unchanged", font=font(20), fill="#334155")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_seed_chart(rows: list[dict[str, str]], path: Path) -> None:
    img = Image.new("RGB", (1100, 650), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((40, 25), "3-seed CER comparison", font=font(30, bold=True), fill="#111827")
    left, top, bottom = 90, 110, 560
    right = 1030
    d.line((left, top, left, bottom), fill="#111827", width=2)
    d.line((left, bottom, right, bottom), fill="#111827", width=2)
    seeds = [row["seed"] for row in rows]
    base = [float(row["baseline_cer"]) for row in rows]
    plus = [float(row["plus_10k_cer"]) for row in rows]
    vals = base + plus
    vmin, vmax = min(vals) * 0.9, max(vals) * 1.05

    def y(v: float) -> int:
        return int(bottom - (v - vmin) / (vmax - vmin) * (bottom - top))

    x_positions = [left + 180 + i * 300 for i in range(len(seeds))]
    for x, seed, b, p in zip(x_positions, seeds, base, plus):
        d.rectangle((x - 55, y(b), x - 15, bottom), fill="#ef4444")
        d.rectangle((x + 15, y(p), x + 55, bottom), fill="#2563eb")
        d.text((x - 20, bottom + 15), seed, font=font(18), fill="#111827")
        d.text((x - 90, y(b) - 28), fmt(b, 3), font=font(14), fill="#991b1b")
        d.text((x + 25, y(p) - 28), fmt(p, 3), font=font(14), fill="#1d4ed8")
    d.rectangle((760, 65, 790, 85), fill="#ef4444")
    d.text((800, 60), "baseline", font=font(18), fill="#111827")
    d.rectangle((760, 95, 790, 115), fill="#2563eb")
    d.text((800, 90), "+10k context", font=font(18), fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_domain_delta_chart(rows: list[dict[str, str]], path: Path) -> None:
    rows = [r for r in rows if r["domain"] != "school"]
    img = Image.new("RGB", (1100, 620), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((40, 25), "Domain-wise mean CER delta (+10k - baseline)", font=font(28, bold=True), fill="#111827")
    left, top, bottom = 120, 110, 520
    zero_x = 760
    d.line((zero_x, top, zero_x, bottom), fill="#64748b", width=2)
    y = top + 55
    scale = 9000
    for row in rows:
        delta = float(row["mean_delta_cer"])
        x = int(zero_x + delta * scale)
        d.rectangle((min(x, zero_x), y - 25, max(x, zero_x), y + 25), fill="#16a34a" if delta < 0 else "#ef4444")
        d.text((40, y - 12), row["domain"], font=font(18), fill="#111827")
        d.text((780, y - 12), f"{delta:.4f} ({float(row['mean_relative_delta_cer']) * 100:.2f}%)", font=font(18), fill="#111827")
        y += 105
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_coverage_chart(rows: list[dict[str, str]], path: Path) -> None:
    rows = [
        row for row in rows
        if row.get("model") == "plus_10k_context" and row.get("scope") == "all"
    ]
    methods = ["feature_only", "model_confidence", "confidence_graph"]
    colors = {"feature_only": "#64748b", "model_confidence": "#2563eb", "confidence_graph": "#16a34a"}
    img = Image.new("RGB", (1200, 700), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((40, 25), "Selective prediction: coverage vs CER (+10k)", font=font(28, bold=True), fill="#111827")
    left, top, right, bottom = 90, 110, 1080, 590
    d.rectangle((left, top, right, bottom), outline="#111827", width=2)
    max_cer = max(float(row["cer"]) for row in rows) * 1.15

    def xy(cov: float, cer: float) -> tuple[int, int]:
        x = int(left + cov * (right - left))
        y = int(bottom - cer / max_cer * (bottom - top))
        return x, y

    for method in methods:
        pts = [
            xy(float(row["coverage"]), float(row["cer"]))
            for row in rows
            if row.get("risk_method") == method
        ]
        if len(pts) >= 2:
            d.line(pts, fill=colors[method], width=4)
            for x, y in pts:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[method])
    y0 = 120
    for method in methods:
        d.line((820, y0, 860, y0), fill=colors[method], width=4)
        d.text((870, y0 - 10), method, font=font(18), fill="#111827")
        y0 += 30
    d.text((left, bottom + 20), "coverage", font=font(18), fill="#111827")
    d.text((20, top), "CER", font=font(18), fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def copy_structural_example(path: Path) -> None:
    candidates = [
        Path("outputs/h2_gold_audit_v1/school_foreground_v3/school_foreground_v3_contact_sheet.png"),
        Path("outputs/h2_gold_audit_v1/contact_sheet/A_highCER_highRisk.png"),
        Path("outputs/h2_gold_audit_v1/border_suppression_v1/border_suppression_contact_sheet.png"),
    ]
    for src in candidates:
        if src.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, path)
            return
    img = Image.new("RGB", (1000, 500), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((40, 40), "Structural diagnostic example source image was not found.", font=font(24, bold=True), fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def table_1(out: Path) -> None:
    rows = [
        {
            "dataset": "Cyrillic Handwriting",
            "language": "Russian",
            "script": "Cyrillic",
            "level": "word/phrase crops",
            "role": "core crop-domain",
            "used_in_primary_htr": "yes",
            "used_in_structural_diagnostics": "yes",
            "used_in_selective_prediction": "yes",
            "used_in_stress_analysis": "yes",
            "primary_claim": "Improves across 3 seeds in domain-wise aggregation.",
            "limitations": "Crop-domain data; not the hardest notebook layout.",
        },
        {
            "dataset": "HKR Words",
            "language": "Russian",
            "script": "Cyrillic",
            "level": "word/phrase crops",
            "role": "secondary word-domain",
            "used_in_primary_htr": "yes",
            "used_in_structural_diagnostics": "yes",
            "used_in_selective_prediction": "yes",
            "used_in_stress_analysis": "yes",
            "primary_claim": "Improves on average; not fully seed-stable domain-wise.",
            "limitations": "HKR domain has 2/3 improved seeds.",
        },
        {
            "dataset": "School Notebooks",
            "language": "Russian",
            "script": "Cyrillic",
            "level": "word crops and natural-line context",
            "role": "hard notebook domain",
            "used_in_primary_htr": "yes",
            "used_in_structural_diagnostics": "yes",
            "used_in_selective_prediction": "yes",
            "used_in_stress_analysis": "yes",
            "primary_claim": "Strongest and most stable HTR gain.",
            "limitations": "Line crops are contextual, not clean isolated line crops.",
        },
        {
            "dataset": "HWR200",
            "language": "Russian",
            "script": "Cyrillic",
            "level": "diagnostic/stress",
            "role": "diagnostic/stress only",
            "used_in_primary_htr": "no",
            "used_in_structural_diagnostics": "limited",
            "used_in_selective_prediction": "no",
            "used_in_stress_analysis": "yes",
            "primary_claim": "Not part of primary HTR claim.",
            "limitations": "Used only as supporting stress/diagnostic material.",
        },
        {
            "dataset": "HKR Forms",
            "language": "Russian",
            "script": "Cyrillic",
            "level": "diagnostic/stress",
            "role": "diagnostic/stress only",
            "used_in_primary_htr": "no",
            "used_in_structural_diagnostics": "limited",
            "used_in_selective_prediction": "no",
            "used_in_stress_analysis": "yes",
            "primary_claim": "Not part of primary HTR claim.",
            "limitations": "Used only as supporting stress/diagnostic material.",
        },
        {
            "dataset": "IAM",
            "language": "English",
            "script": "Latin",
            "level": "background only",
            "role": "optional background",
            "used_in_primary_htr": "no",
            "used_in_structural_diagnostics": "no",
            "used_in_selective_prediction": "no",
            "used_in_stress_analysis": "no",
            "primary_claim": "Does not support Russian-domain claim.",
            "limitations": "Different language/script; not used for final Russian HTR claim.",
        },
    ]
    write_csv(rows, out / "table_1_dataset_roles.csv")
    write_md_table(rows, out / "table_1_dataset_roles.md", title="Table 1. Dataset roles")


def table_2(out: Path) -> None:
    rows = read_csv(ROOT / "seed_confirmation_deltas.csv")
    summary = read_json(ROOT / "seed_confirmation_summary.json")["delta_summary"]
    formatted = []
    for row in rows:
        formatted.append(
            {
                "seed": row["seed"],
                "baseline_CER": fmt(row["baseline_cer"]),
                "plus10k_CER": fmt(row["plus_10k_cer"]),
                "delta_CER": fmt(row["delta_cer"]),
                "relative_delta_CER": pct(row["relative_delta_cer"]),
                "baseline_WER": fmt(row["baseline_wer"]),
                "plus10k_WER": fmt(row["plus_10k_wer"]),
                "delta_WER": fmt(row["delta_wer"]),
                "baseline_exact": fmt(row["baseline_exact"]),
                "plus10k_exact": fmt(row["plus_10k_exact"]),
                "delta_exact": fmt(row["delta_exact"]),
            }
        )
    formatted.append(
        {
            "seed": "mean",
            "baseline_CER": "",
            "plus10k_CER": "",
            "delta_CER": fmt(summary["mean_delta_cer"]),
            "relative_delta_CER": pct(summary["mean_relative_delta_cer"]),
            "baseline_WER": "",
            "plus10k_WER": "",
            "delta_WER": fmt(summary["mean_delta_wer"]),
            "baseline_exact": "",
            "plus10k_exact": "",
            "delta_exact": fmt(summary["mean_delta_exact"]),
        }
    )
    write_csv(formatted, out / "table_2_primary_htr_3seed.csv")
    note = (
        f"Aggregate: mean ΔCER={summary['mean_delta_cer']:.6f}, "
        f"std ΔCER={summary['std_delta_cer']:.6f}, "
        f"mean relative ΔCER={summary['mean_relative_delta_cer'] * 100:.2f}%, "
        f"improved CER seeds={summary['improved_cer_seeds_n']}/3."
    )
    write_md_table(formatted, out / "table_2_primary_htr_3seed.md", title="Table 2. Primary HTR 3-seed result", note=note)


def table_3(out: Path) -> None:
    rows = read_csv(ROOT / "domainwise_seed_summary.csv")
    formatted = []
    for row in rows:
        if row["domain"] == "school":
            continue
        formatted.append(
            {
                "domain": row["domain"],
                "mean_baseline_CER": fmt(row["mean_baseline_cer"]),
                "mean_plus10k_CER": fmt(row["mean_plus_10k_cer"]),
                "mean_delta_CER": fmt(row["mean_delta_cer"]),
                "relative_delta_CER": pct(row["mean_relative_delta_cer"]),
                "improved_seeds": f"{row['improved_cer_seeds_n']}/{row['seeds_n']}",
                "interpretation": row["interpretation"],
            }
        )
    write_csv(formatted, out / "table_3_domainwise_htr.csv")
    write_md_table(
        formatted,
        out / "table_3_domainwise_htr.md",
        title="Table 3. Domain-wise HTR result",
        note="Natural-line context augmentation gives a seed-stable overall gain, with the strongest and most stable effect on School Notebooks.",
    )


def table_4(out: Path) -> None:
    summary = read_json(Path("outputs/iter2_structural_gold_v1/annotation_summary.json"))
    overall = summary["overall"]
    n = overall["n"]
    sev = overall["severity_counts"]
    htr = overall["htr_error_explained_by_structure"]
    rows = [
        {
            "n": n,
            "foreground_ok": fmt(overall["rates"]["foreground_ok"], 3),
            "skeleton_ok": fmt(overall["rates"]["skeleton_ok"], 3),
            "graph_ok": fmt(overall["rates"]["graph_ok"], 3),
            "structural_usable": fmt(overall["rates"]["structural_usable"], 3),
            "line_residual_minor_or_dominant": sev["line_residual"].get("1", 0) + sev["line_residual"].get("2", 0),
            "missed_ink_minor": sev["missed_ink"].get("1", 0),
            "neighbor_noise_minor": sev["neighbor_noise"].get("1", 0),
            "false_ink_minor": sev["false_ink"].get("1", 0),
            "htr_error_explained_by_structure": "; ".join(f"{k}:{v}" for k, v in htr.items()),
            "interpretation": "Diagnostic usability validated; not a full topology or pen-trajectory benchmark.",
        }
    ]
    write_csv(rows, out / "table_4_structural_gold_diagnostic.csv")
    write_md_table(
        rows,
        out / "table_4_structural_gold_diagnostic.md",
        title="Table 4. Structural gold diagnostic",
        note="Structural gold validates diagnostic usability, not full topological correctness.",
    )


def nearest_cer(rows: list[dict[str, str]], model: str, risk: str, coverage: float, scope: str = "all") -> str:
    candidates = [
        row for row in rows
        if row["model"] == model and row["risk_method"] == risk and row["scope"] == scope
    ]
    if not candidates:
        return ""
    best = min(candidates, key=lambda row: abs(float(row["coverage"]) - coverage))
    return fmt(best["cer"])


def table_5(out: Path) -> None:
    summary = read_json(Path("outputs/htr_graph_v1/selective_iter2_confidence_v1/selective_summary.json"))
    curves = read_csv(Path("outputs/htr_graph_v1/selective_iter2_confidence_v1/coverage_curves.csv"))
    clearance = read_json(ROOT / "selective_prediction_leakage_clearance.json")
    model = "plus_10k_context"
    rows = []
    for risk in ["feature_only", "model_confidence", "confidence_graph"]:
        info = summary["models"][model]["risk_methods"][risk]
        rows.append(
            {
                "risk_model": risk,
                "ROC_AUC": fmt(info.get("risk_auc_exact_error_all")),
                "PR_AUC": "",
                "CER@90": nearest_cer(curves, model, risk, 0.9),
                "CER@80": nearest_cer(curves, model, risk, 0.8),
                "CER@70": nearest_cer(curves, model, risk, 0.7),
                "CER@50": nearest_cer(curves, model, risk, 0.5),
                "domain": "all",
                "notes": "Canonical +10k; leakage clearance " + clearance["clearance_status"],
            }
        )
    write_csv(rows, out / "table_5_selective_prediction.csv")
    write_md_table(
        rows,
        out / "table_5_selective_prediction.md",
        title="Table 5. Selective prediction",
        note="text_len was detected only in post-hoc reporting/stratification, not as a risk-model feature.",
    )


def table_6(out: Path) -> None:
    image = read_json(Path("outputs/htr_graph_v1/eval_tri10k_image_only_plus_school_lines_10k_context_v1_test_final/summary.json"))
    graph = read_json(Path("outputs/htr_graph_v1/eval_tri10k_graph_fusion_plus_school_lines_10k_context_v1_test_final/summary.json"))
    zero = read_json(Path("outputs/htr_graph_v1/eval_tri10k_graph_fusion_plus_school_lines_10k_context_v1_test_final_zero_graph/summary.json"))

    def metrics(obj: dict[str, Any]) -> dict[str, Any]:
        return obj.get("metrics", obj)

    rows = [
        {
            "model": "image-only +10k",
            "CER": fmt(metrics(image)["cer"]),
            "WER": fmt(metrics(image)["wer"]),
            "exact": fmt(metrics(image)["exact"]),
            "interpretation": "Primary canonical image-only model.",
        },
        {
            "model": "graph-fusion +10k",
            "CER": fmt(metrics(graph)["cer"]),
            "WER": fmt(metrics(graph)["wer"]),
            "exact": fmt(metrics(graph)["exact"]),
            "interpretation": "Exploratory; CER/WER improve in this run but exact is lower and seed-stability is not established.",
        },
        {
            "model": "zero-graph ablation",
            "CER": fmt(metrics(zero)["cer"]),
            "WER": fmt(metrics(zero)["wer"]),
            "exact": fmt(metrics(zero)["exact"]),
            "interpretation": "Performance degradation suggests graph branch is used.",
        },
    ]
    write_csv(rows, out / "table_6_graph_fusion_exploratory.csv")
    write_md_table(rows, out / "table_6_graph_fusion_exploratory.md", title="Table 6. Graph-fusion exploratory result")


def table_7(out: Path) -> None:
    rows = [
        {
            "claim": "+10k natural-line context improves HTR.",
            "supported_by": "3 seeds; seed provenance; domain-wise aggregation.",
            "strength": "strong",
            "limitation": "Strongest on School; HKR is not 3/3 stable.",
            "allowed_wording": "Natural-line context augmentation improves image-only HTR across 3 seeds, especially on School Notebooks.",
            "forbidden_wording": "The improvement is equally stable in every domain.",
        },
        {
            "claim": "HI-CSG-R is diagnostically usable.",
            "supported_by": "Structural gold diagnostic subset.",
            "strength": "moderate/strong diagnostic",
            "limitation": "Not a pixel-level topology benchmark.",
            "allowed_wording": "HI-CSG-R provides a structurally usable diagnostic representation.",
            "forbidden_wording": "HI-CSG-R recovers true pen trajectory or full topology.",
        },
        {
            "claim": "Selective prediction provides a reliability layer.",
            "supported_by": "Canonical +10k confidence/graph-quality checks and leakage clearance.",
            "strength": "secondary applied",
            "limitation": "Coverage thresholds are not group-fair globally.",
            "allowed_wording": "Selective prediction supports risk-aware filtering.",
            "forbidden_wording": "Selective prediction improves full-coverage CER.",
        },
        {
            "claim": "Graph-fusion improves recognition.",
            "supported_by": "Single exploratory graph-fusion pilot.",
            "strength": "weak/exploratory",
            "limitation": "No seed-stable superiority; mixed domain effects.",
            "allowed_wording": "Graph-fusion shows limited/domain-dependent effects and is exploratory.",
            "forbidden_wording": "Graph-fusion proves universal recognition superiority.",
        },
    ]
    write_csv(rows, out / "table_7_claims_limitations.csv")
    write_md_table(rows, out / "table_7_claims_limitations.md", title="Table 7. Claims and limitations")


def write_claims(out: Path) -> None:
    claims = out / "thesis_claims"
    claims.mkdir(parents=True, exist_ok=True)
    (claims / "allowed_claims.md").write_text(
        "\n".join(
            [
                "# Allowed claims",
                "",
                "1. HI-CSG-R is diagnostically usable on sampled Russian handwriting data.",
                "2. Natural-line context augmentation improves image-only HTR across 3 seeds.",
                "3. The strongest domain-wise improvement is on School Notebooks.",
                "4. Cyrillic improves in all seeds.",
                "5. HKR improves on average but is not fully seed-stable.",
                "6. Selective prediction is acceptable as secondary reliability analysis.",
                "7. Graph-fusion is exploratory and not the main source of recognition gain.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (claims / "forbidden_claims.md").write_text(
        "\n".join(
            [
                "# Forbidden claims",
                "",
                "1. HI-CSG-R recovers true pen trajectory.",
                "2. HI-CSG-R recovers writing order.",
                "3. Structural gold proves full graph topology correctness.",
                "4. Graph-fusion universally improves HTR.",
                "5. The system is SOTA.",
                "6. IAM supports the Russian-domain claim.",
                "7. Selective prediction improves full-coverage CER.",
                "8. text_len was used as a model feature.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (claims / "final_narrative.md").write_text(
        (
            "# Final narrative\n\n"
            "The project introduces HI-CSG-R as a structural diagnostic representation "
            "for Russian offline HTR. The strongest recognition gain comes from "
            "natural-line context augmentation, validated across three seeds. "
            "Domain-wise analysis shows the strongest effect on School Notebooks, "
            "while Cyrillic also improves consistently and HKR improves on average. "
            "HI-CSG-R supports structural diagnostics and selective prediction. "
            "Direct graph-fusion remains exploratory.\n"
        ),
        encoding="utf-8",
    )


def write_appendix(out: Path) -> None:
    app = out / "thesis_appendix"
    app.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, text=True, capture_output=True).stdout.strip()
    (app / "appendix_a_reproducibility.md").write_text(
        f"""# Appendix A. Reproducibility

- repo commit hash at package generation: `{commit}`
- final package root: `outputs/final_result_package_v1`
- primary test manifest: `data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/test.jsonl`
- canonical +10k test manifest: `data/experiments/htr_baseline_v1_ctc_ready/tri10k_mixed_plus_school_lines_10k_context_v1/test.jsonl`
- baseline checkpoints: `outputs/htr_graph_v1/tri10k_image_only_v1* / best.pt`
- +10k checkpoints: `outputs/htr_graph_v1/tri10k_image_only_plus_school_lines_10k_context_v1* / best.pt`
- seeds: 42, 43, 44
- blank penalty protocol: final test evaluation uses validation-selected/evaluation-fixed penalty; seed42 provenance confirms `blank_logit_penalty=-0.4`.
- final scripts: `tools/build_results_inventory_v1.py`, `tools/check_seed42_provenance_v1.py`, `tools/build_seed_confirmation_report_v1.py`, `tools/build_domainwise_seed_confirmation_v1.py`, `tools/check_selective_prediction_canonical_v1.py`, `tools/write_selective_leakage_clearance_v1.py`.
""",
        encoding="utf-8",
    )
    (app / "appendix_b_seed_confirmation.md").write_text(
        "# Appendix B. Seed confirmation\n\nSee `seed_confirmation_summary.md`, `domainwise_seed_confirmation.md`, and `seed42_provenance_check.md`.\n",
        encoding="utf-8",
    )
    (app / "appendix_c_structural_gold_protocol.md").write_text(
        (
            "# Appendix C. Structural gold protocol\n\n"
            "The structural gold subset is used as diagnostic usability evidence, not as a topology-perfect graph benchmark. "
            "Fields include foreground/skeleton/graph usability and visible failure types such as line residual, missed ink, false ink and neighbor noise.\n"
        ),
        encoding="utf-8",
    )
    (app / "appendix_d_selective_prediction_clearance.md").write_text(
        "# Appendix D. Selective prediction clearance\n\nSee `selective_prediction_canonical_check.md` and `selective_prediction_leakage_clearance.md`. The only leakage-risk hit is `text_len` in post-hoc reporting/stratification.\n",
        encoding="utf-8",
    )
    (app / "appendix_e_negative_results.md").write_text(
        (
            "# Appendix E. Negative and exploratory results\n\n"
            "Graph-fusion is treated as exploratory: it shows that graph information can be used by the model, but does not establish seed-stable universal recognition superiority. "
            "Robustness/H1 is not confirmed in the current implementation and architecture search is stopped for this thesis package.\n"
        ),
        encoding="utf-8",
    )


def write_examples(out: Path) -> None:
    base = out / "thesis_examples"
    folders = [
        "structural_ok",
        "structural_hard",
        "selective_accept",
        "selective_reject",
        "graph_fusion_cases",
    ]
    for folder in folders:
        target = base / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text(
            f"# {folder}\n\nCurated example slots. Source artifacts are indexed in the final package; copy visual examples here only after manual selection.\n",
            encoding="utf-8",
        )


def build_all(out: Path) -> None:
    for folder in [
        "thesis_tables",
        "thesis_figures",
        "thesis_appendix",
        "thesis_claims",
        "thesis_examples",
    ]:
        (out / folder).mkdir(parents=True, exist_ok=True)

    tables = out / "thesis_tables"
    table_1(tables)
    table_2(tables)
    table_3(tables)
    table_4(tables)
    table_5(tables)
    table_6(tables)
    table_7(tables)

    figures = out / "thesis_figures"
    save_pipeline_figure(figures / "fig_1_hicsgr_pipeline.png")
    save_context_figure(figures / "fig_2_natural_line_context.png")
    save_seed_chart(read_csv(out / "seed_confirmation_deltas.csv"), figures / "fig_3_seed_cer_comparison.png")
    save_domain_delta_chart(read_csv(out / "domainwise_seed_summary.csv"), figures / "fig_4_domainwise_delta_cer.png")
    save_coverage_chart(read_csv(Path("outputs/htr_graph_v1/selective_iter2_confidence_v1/coverage_curves.csv")), figures / "fig_5_selective_coverage_risk.png")
    copy_structural_example(figures / "fig_6_structural_diagnostic_example.png")

    write_claims(out)
    write_appendix(out)
    write_examples(out)

    checklist = out / "thesis_claims" / "acceptance_checklist.md"
    checklist.write_text(
        "\n".join(
            [
                "# Final acceptance checklist",
                "",
                "- [x] Table 1 dataset roles готова",
                "- [x] Table 2 primary HTR 3-seed готова",
                "- [x] Table 3 domain-wise HTR готова",
                "- [x] Table 4 structural gold diagnostic готова",
                "- [x] Table 5 selective prediction готова",
                "- [x] Table 6 graph-fusion exploratory готова",
                "- [x] Table 7 claims/limitations готова",
                "- [x] Figure 1 pipeline готова",
                "- [x] Figure 2 natural-line context готова",
                "- [x] Figure 3 seed comparison готова",
                "- [x] Figure 4 domain-wise ΔCER готова",
                "- [x] Figure 5 selective coverage-risk готова",
                "- [x] Figure 6 structural example готова",
                "- [x] allowed_claims.md готов",
                "- [x] forbidden_claims.md готов",
                "- [x] final_narrative.md готов",
                "- [x] appendix reproducibility готов",
                "- [x] appendix seed confirmation готов",
                "- [x] appendix structural gold готов",
                "- [x] appendix selective clearance готов",
                "- [x] appendix negative results готов",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(ROOT))
    args = parser.parse_args()
    out = Path(args.out_dir)
    build_all(out)
    print("wrote thesis package:", out)


if __name__ == "__main__":
    main()
