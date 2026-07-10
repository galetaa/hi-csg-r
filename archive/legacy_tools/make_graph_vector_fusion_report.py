from __future__ import annotations

import json
from pathlib import Path


OUT = Path("outputs/htr_graph_v1")
REPORT_MD = OUT / "graph_vector_fusion_report.md"
REPORT_JSON = OUT / "graph_vector_fusion_report.json"

RUNS = {
    "image_only": {
        "title": "tri10k image-only v1",
        "prefix": "eval_tri10k_image_only_v1",
        "overall_sweep": OUT / "sweep_tri10k_image_only_v1_val",
    },
    "graph_v1": {
        "title": "tri10k graph-fusion v1 all-features",
        "prefix": "eval_tri10k_graph_fusion_v1",
        "overall_sweep": OUT / "sweep_tri10k_graph_fusion_v1_val",
    },
    "graph_v2": {
        "title": "tri10k graph-fusion v2 lowcap-all",
        "prefix": "eval_tri10k_graph_fusion_v2_lowcap_all",
        "overall_sweep": OUT / "sweep_tri10k_graph_fusion_v2_lowcap_all_val",
    },
    "graph_v3": {
        "title": "tri10k graph-fusion v3 normtopo",
        "prefix": "eval_tri10k_graph_fusion_v3_normtopo",
        "overall_sweep": OUT / "sweep_tri10k_graph_fusion_v3_normtopo_val",
    },
}

DATASETS = [
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks_clean",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def best_sweep(sweep_dir: Path):
    rows = []
    for p in sorted(sweep_dir.glob("penalty_*/summary.json")):
        s = load_json(p)
        if "metrics" in s:
            m = s["metrics"]
            rows.append({
                "penalty": s["blank_logit_penalty"],
                "cer": m["cer"],
                "wer": m["wer"],
                "exact": m["exact"],
                "pred_len": m["pred_len_mean"],
                "blank": m["argmax_blank_ratio"],
            })
        else:
            rows.append({
                "penalty": s["blank_logit_penalty"],
                "cer": s["cer"],
                "wer": s["wer"],
                "exact": s["exact"],
                "pred_len": s["pred_len_mean"],
                "blank": s["argmax_blank_ratio"],
            })

    return sorted(rows, key=lambda x: x["cer"])[0]


def load_eval(prefix: str, ds: str, split: str):
    p = OUT / f"{prefix}_{ds}_{split}_final" / "summary.json"
    s = load_json(p)

    if "metrics" in s:
        m = s["metrics"]
        return {
            "n": m["n"],
            "cer": m["cer"],
            "wer": m["wer"],
            "exact": m["exact"],
            "pred_len": m["pred_len_mean"],
            "blank": m["argmax_blank_ratio"],
            "penalty": s["blank_logit_penalty"],
        }

    return {
        "n": s["n"],
        "cer": s["cer"],
        "wer": s["wer"],
        "exact": s["exact"],
        "pred_len": s["pred_len_mean"],
        "blank": s["argmax_blank_ratio"],
        "penalty": s["blank_logit_penalty"],
    }


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def rel(base, new):
    return (base - new) / base if base else 0.0


def main():
    data = {"runs": {}}

    for key, cfg in RUNS.items():
        data["runs"][key] = {
            "title": cfg["title"],
            "best_sweep": best_sweep(cfg["overall_sweep"]),
            "datasets": {},
        }

        for ds in DATASETS:
            data["runs"][key]["datasets"][ds] = {}
            for split in ["val", "test"]:
                data["runs"][key]["datasets"][ds][split] = load_eval(cfg["prefix"], ds, split)

    lines = []
    lines.append("# Stage 4 graph-vector fusion report\n")
    lines.append("## 1. Purpose\n")
    lines.append(
        "This report compares the tri10k image-only control model against global graph-vector fusion variants. "
        "All graph-vector variants exclude `text_len` to avoid label-length leakage.\n"
    )

    lines.append("## 2. Overall mixed validation\n")
    lines.append("| run | best penalty | CER | WER | exact | pred_len | blank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, run in data["runs"].items():
        b = run["best_sweep"]
        lines.append(
            f"| {run['title']} | {fmt(b['penalty'], 1)} | {fmt(b['cer'])} | "
            f"{fmt(b['wer'])} | {fmt(b['exact'])} | {fmt(b['pred_len'], 2)} | {fmt(b['blank'], 3)} |"
        )

    lines.append("")
    lines.append("## 3. Test CER by dataset\n")
    lines.append("| run | Cyrillic test | HKR test | School test |")
    lines.append("|---|---:|---:|---:|")
    for key, run in data["runs"].items():
        lines.append(
            f"| {run['title']} | "
            f"{fmt(run['datasets']['cyrillic_handwriting']['test']['cer'])} | "
            f"{fmt(run['datasets']['hkr_words']['test']['cer'])} | "
            f"{fmt(run['datasets']['school_notebooks_clean']['test']['cer'])} |"
        )

    lines.append("")
    lines.append("## 4. Relative change vs image-only\n")
    lines.append("| run | Cyrillic | HKR | School |")
    lines.append("|---|---:|---:|---:|")

    base = data["runs"]["image_only"]["datasets"]
    for key, run in data["runs"].items():
        if key == "image_only":
            continue
        row = run["datasets"]
        lines.append(
            f"| {run['title']} | "
            f"{100 * rel(base['cyrillic_handwriting']['test']['cer'], row['cyrillic_handwriting']['test']['cer']):.1f}% | "
            f"{100 * rel(base['hkr_words']['test']['cer'], row['hkr_words']['test']['cer']):.1f}% | "
            f"{100 * rel(base['school_notebooks_clean']['test']['cer'], row['school_notebooks_clean']['test']['cer']):.1f}% |"
        )

    lines.append("")
    lines.append("## 5. Interpretation\n")
    lines.append(
        "Global graph-vector fusion improves mixed validation CER, with the best result from the low-capacity all-feature variant. "
        "However, the improvement is not robust across datasets: HKR Words improves, while Cyrillic Handwriting and School Notebooks do not consistently improve on test.\n"
    )
    lines.append(
        "The normalized-topology-only variant does not outperform low-capacity all-feature fusion, suggesting that raw geometry and domain/style cues contribute to the observed gain.\n"
    )
    lines.append(
        "Conclusion: global graph-vector fusion is useful as a diagnostic baseline but should not be treated as the final graph-aware model. "
        "The next stage should inject graph-derived structure locally, aligned with image coordinates.\n"
    )

    lines.append("## 6. Next step\n")
    lines.append(
        "Proceed to local graph-aware CRNN using additional foreground/skeleton/distance channels.\n"
    )

    REPORT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("wrote:", REPORT_MD)
    print("wrote:", REPORT_JSON)


if __name__ == "__main__":
    main()