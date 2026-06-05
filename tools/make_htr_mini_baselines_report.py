from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("outputs/htr_baseline_v1")
REPORT_MD = OUT_ROOT / "htr_mini_baselines_report.md"
REPORT_JSON = OUT_ROOT / "htr_mini_baselines_report.json"


RUNS = {
    "iam": {
        "title": "IAM mini10k v1",
        "dataset": "IAM",
        "level": "line",
        "language": "en",
        "checkpoint_dir": OUT_ROOT / "mini10k_iam_v1",
        "eval_dirs": {
            "train": OUT_ROOT / "eval_mini10k_iam_v1_train_final",
            "val": OUT_ROOT / "eval_mini10k_iam_v1_val_final",
            "test": OUT_ROOT / "eval_mini10k_iam_v1_test_final",
        },
        "notes": [
            "IAM is line-level and English.",
            "Exact match is expected to be much lower than word-level datasets because the target strings are long.",
            "The mini10k subset is close to full IAM train size.",
        ],
    },
    "cyrillic_handwriting": {
        "title": "Cyrillic Handwriting mini10k v1",
        "dataset": "Cyrillic Handwriting",
        "level": "word/phrase",
        "language": "ru",
        "checkpoint_dir": OUT_ROOT / "mini10k_cyrillic_v1",
        "eval_dirs": {
            "train": OUT_ROOT / "eval_mini10k_cyrillic_v1_train_final",
            "val": OUT_ROOT / "eval_mini10k_cyrillic_v1_val_final",
            "test": OUT_ROOT / "eval_mini10k_cyrillic_v1_test_final",
        },
        "notes": [
            "Cyrillic Handwriting is a Russian word/phrase crop dataset.",
            "The test split is smaller than 2000 because the available test split contains 1563 samples.",
            "The model strongly overfits train and generalizes moderately to val/test.",
        ],
    },
    "hkr_words": {
        "title": "HKR Words mini10k v1",
        "dataset": "HKR Words",
        "level": "word/phrase",
        "language": "ru_kk",
        "checkpoint_dir": OUT_ROOT / "mini10k_hkr_words_v1",
        "eval_dirs": {
            "train": OUT_ROOT / "eval_mini10k_hkr_words_v1_train_final_bestpen",
            "val": OUT_ROOT / "eval_mini10k_hkr_words_v1_val_final_bestpen",
            "test": OUT_ROOT / "eval_mini10k_hkr_words_v1_test_final_bestpen",
        },
        "notes": [
            "HKR Words uses a text-grouped split.",
            "The benchmark controls target-text leakage, not writer leakage, because writer_id is unavailable.",
            "Val and test are close, which suggests a stable split.",
        ],
    },
    "school_notebooks": {
        "title": "School Notebooks mini10k v1",
        "dataset": "School Notebooks Clean",
        "level": "word/phrase",
        "language": "ru",
        "checkpoint_dir": OUT_ROOT / "mini10k_school_notebooks_v1",
        "eval_dirs": {
            "train": OUT_ROOT / "eval_mini10k_school_notebooks_v1_train_final",
            "val": OUT_ROOT / "eval_mini10k_school_notebooks_v1_val_final",
            "test": OUT_ROOT / "eval_mini10k_school_notebooks_v1_test_final",
        },
        "notes": [
            "School Notebooks Clean excludes single_character_or_mark and occluded samples.",
            "Category-level metrics must be reported separately.",
            "pupil_text dominates the subset; pupil_comment and teacher_comment have smaller n and should be interpreted carefully.",
        ],
    },
}


def load_summary(eval_dir: Path) -> dict[str, Any]:
    path = eval_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(run_key: str, split: str, summary: dict[str, Any]) -> dict[str, Any]:
    m = summary["metrics"]
    return {
        "run": run_key,
        "split": split,
        "n": m["n"],
        "cer": m["cer"],
        "wer": m["wer"],
        "exact": m["exact"],
        "pred_len": m["pred_len_mean"],
        "empty": m["pred_empty_ratio"],
        "blank": m["argmax_blank_ratio"],
        "blank_logit_penalty": summary["blank_logit_penalty"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "checkpoint_val_cer": summary["checkpoint_val_cer"],
    }


def fmt(x: float | int | None, ndigits: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{ndigits}f}"


def main() -> None:
    report_data = {
        "runs": {},
        "overall_rows": [],
    }

    for run_key, cfg in RUNS.items():
        split_summaries = {}
        split_rows = {}

        for split, eval_dir in cfg["eval_dirs"].items():
            summary = load_summary(eval_dir)
            split_summaries[split] = summary
            row = metric_row(run_key, split, summary)
            split_rows[split] = row
            report_data["overall_rows"].append(row)

        report_data["runs"][run_key] = {
            "title": cfg["title"],
            "dataset": cfg["dataset"],
            "level": cfg["level"],
            "language": cfg["language"],
            "checkpoint_dir": str(cfg["checkpoint_dir"]),
            "notes": cfg["notes"],
            "splits": split_rows,
            "grouped": {
                split: split_summaries[split]["metrics"].get("grouped", {})
                for split in split_summaries
            },
        }

    lines = []
    lines.append("# HTR mini-baselines report — Stage 3\n")
    lines.append("## 1. Purpose\n")
    lines.append(
        "This report summarizes the first image-only HTR baselines for the HI-CSG-R project. "
        "All runs use OCR-preprocessed grayscale images and a height-preserving CRNN + BiLSTM + CTC model. "
        "Decode blank penalties are selected on validation splits and then applied consistently to train/val/test evaluation.\n"
    )

    lines.append("## 2. Overall metrics\n")
    lines.append("| dataset | split | n | CER | WER | exact | pred_len | empty | blank | penalty | epoch |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for run_key, cfg in RUNS.items():
        for split in ["train", "val", "test"]:
            r = report_data["runs"][run_key]["splits"][split]
            lines.append(
                f"| {cfg['title']} | {split} | {r['n']} | "
                f"{fmt(r['cer'])} | {fmt(r['wer'])} | {fmt(r['exact'])} | "
                f"{fmt(r['pred_len'], 2)} | {fmt(r['empty'], 3)} | {fmt(r['blank'], 3)} | "
                f"{fmt(r['blank_logit_penalty'], 3)} | {r['checkpoint_epoch']} |"
            )

    lines.append("")
    lines.append("## 3. Test-set comparison\n")
    lines.append("| dataset | level | language | test n | test CER | test WER | test exact |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    for run_key, cfg in RUNS.items():
        r = report_data["runs"][run_key]["splits"]["test"]
        lines.append(
            f"| {cfg['dataset']} | {cfg['level']} | {cfg['language']} | {r['n']} | "
            f"{fmt(r['cer'])} | {fmt(r['wer'])} | {fmt(r['exact'])} |"
        )

    lines.append("")
    lines.append("## 4. School Notebooks category breakdown\n")
    school = report_data["runs"]["school_notebooks"]

    for split in ["train", "val", "test"]:
        lines.append(f"### {split}")
        lines.append("| group | n | CER | WER | exact |")
        lines.append("|---|---:|---:|---:|---:|")

        grouped = school["grouped"].get(split, {})
        for group_key, vals in sorted(grouped.items()):
            lines.append(
                f"| `{group_key}` | {vals['n']} | {fmt(vals['cer'])} | "
                f"{fmt(vals['wer'])} | {fmt(vals['exact'])} |"
            )
        lines.append("")

    lines.append("## 5. Interpretation\n")
    lines.append("### 5.1 IAM\n")
    lines.append(
        "IAM achieves the lowest CER among the mini-baselines. This is not directly comparable to the Russian crop datasets, "
        "because IAM is English line-level recognition with longer targets. Exact match is naturally lower for line-level targets.\n"
    )

    lines.append("### 5.2 Cyrillic Handwriting\n")
    lines.append(
        "Cyrillic Handwriting shows strong train memorization and moderate test performance. "
        "The test split is relatively small and appears harder than validation.\n"
    )

    lines.append("### 5.3 HKR Words\n")
    lines.append(
        "HKR Words has stable validation and test CER. This is methodologically important because the split is text-grouped, "
        "so target-text leakage is controlled.\n"
    )

    lines.append("### 5.4 School Notebooks\n")
    lines.append(
        "School Notebooks performs strongly on the main pupil_text word category. "
        "pupil_comment and teacher_comment are harder and have smaller sample counts, so they should be reported separately.\n"
    )

    lines.append("## 6. Stage 3 status\n")
    lines.append("```text")
    lines.append("[x] HTR manifests created")
    lines.append("[x] CTC-ready manifests created")
    lines.append("[x] One-sample overfit passed")
    lines.append("[x] Tiny64 overfit passed")
    lines.append("[x] Cyrillic mini10k baseline passed")
    lines.append("[x] HKR Words mini10k baseline passed")
    lines.append("[x] School Notebooks mini10k baseline passed")
    lines.append("[x] IAM mini10k baseline passed")
    lines.append("```")
    lines.append("")

    lines.append("## 7. Next step\n")
    lines.append(
        "The next recommended step is full single-dataset baselines, starting with IAM and Cyrillic Handwriting, "
        "then HKR Words, then School Notebooks. After full image-only baselines are established, graph-aware experiments can begin.\n"
    )

    REPORT_JSON.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("wrote:", REPORT_MD)
    print("wrote:", REPORT_JSON)


if __name__ == "__main__":
    main()