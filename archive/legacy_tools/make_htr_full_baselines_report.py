from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("outputs/htr_baseline_v1")
REPORT_MD = OUT_ROOT / "htr_full_baselines_report.md"
REPORT_JSON = OUT_ROOT / "htr_full_baselines_report.json"


RUNS = {
    "iam": {
        "title": "IAM full v1",
        "dataset": "IAM",
        "level": "line",
        "language": "en",
        "eval_dirs": {
            "train": OUT_ROOT / "eval_full_iam_v1_train_final",
            "val": OUT_ROOT / "eval_full_iam_v1_val_final",
            "test": OUT_ROOT / "eval_full_iam_v1_test_final",
        },
        "note": "English line-level dataset. Exact match is less comparable with word-level datasets.",
    },
    "cyrillic_handwriting": {
        "title": "Cyrillic Handwriting full v1",
        "dataset": "Cyrillic Handwriting",
        "level": "word/phrase",
        "language": "ru",
        "eval_dirs": {
            "train": OUT_ROOT / "eval_full_cyrillic_v1_train_final",
            "val": OUT_ROOT / "eval_full_cyrillic_v1_val_final",
            "test": OUT_ROOT / "eval_full_cyrillic_v1_test_final",
        },
        "note": "Russian word/phrase crop dataset. Test remains harder than validation.",
    },
    "hkr_words": {
        "title": "HKR Words full v1",
        "dataset": "HKR Words",
        "level": "word/phrase",
        "language": "ru_kk",
        "eval_dirs": {
            "train": OUT_ROOT / "eval_full_hkr_words_v1_train_final",
            "val": OUT_ROOT / "eval_full_hkr_words_v1_val_final",
            "test": OUT_ROOT / "eval_full_hkr_words_v1_test_final",
        },
        "note": "Text-grouped split; val/test are stable and close.",
    },
    "school_notebooks": {
        "title": "School Notebooks full v1",
        "dataset": "School Notebooks Clean",
        "level": "word/phrase",
        "language": "ru",
        "eval_dirs": {
            "train": OUT_ROOT / "eval_full_school_notebooks_v1_train_final",
            "val": OUT_ROOT / "eval_full_school_notebooks_v1_val_final",
            "test": OUT_ROOT / "eval_full_school_notebooks_v1_test_final",
        },
        "note": "Largest Russian crop dataset. Category breakdown is mandatory.",
    },
}


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def fmt(x: float | int, nd: int = 4) -> str:
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def main() -> None:
    data: dict[str, Any] = {"runs": {}}

    for key, cfg in RUNS.items():
        splits = {}
        grouped = {}

        for split, eval_dir in cfg["eval_dirs"].items():
            s = load_summary(eval_dir)
            m = s["metrics"]

            splits[split] = {
                "n": m["n"],
                "cer": m["cer"],
                "wer": m["wer"],
                "exact": m["exact"],
                "pred_len": m["pred_len_mean"],
                "empty": m["pred_empty_ratio"],
                "blank": m["argmax_blank_ratio"],
                "penalty": s["blank_logit_penalty"],
                "checkpoint_epoch": s["checkpoint_epoch"],
                "checkpoint_val_cer": s["checkpoint_val_cer"],
            }

            grouped[split] = m.get("grouped", {})

        data["runs"][key] = {
            "title": cfg["title"],
            "dataset": cfg["dataset"],
            "level": cfg["level"],
            "language": cfg["language"],
            "note": cfg["note"],
            "splits": splits,
            "grouped": grouped,
        }

    lines = []
    lines.append("# HTR full baselines report — Stage 3\n")
    lines.append("## 1. Setup\n")
    lines.append("```text")
    lines.append("model: height-preserving CRNN + BiLSTM + CTC")
    lines.append("input: OCR-preprocessed grayscale images")
    lines.append("decode: greedy CTC, blank penalty selected on validation")
    lines.append("status: image-only HTR baseline stage completed")
    lines.append("```")
    lines.append("")

    lines.append("## 2. Overall metrics\n")
    lines.append("| dataset | split | n | CER | WER | exact | pred_len | empty | blank | penalty | epoch |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for key, run in data["runs"].items():
        for split in ["train", "val", "test"]:
            r = run["splits"][split]
            lines.append(
                f"| {run['title']} | {split} | {r['n']} | "
                f"{fmt(r['cer'])} | {fmt(r['wer'])} | {fmt(r['exact'])} | "
                f"{fmt(r['pred_len'], 2)} | {fmt(r['empty'], 3)} | {fmt(r['blank'], 3)} | "
                f"{fmt(r['penalty'], 3)} | {r['checkpoint_epoch']} |"
            )

    lines.append("")
    lines.append("## 3. Test-set comparison\n")
    lines.append("| dataset | level | language | test n | test CER | test WER | test exact |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    for key, run in data["runs"].items():
        r = run["splits"]["test"]
        lines.append(
            f"| {run['dataset']} | {run['level']} | {run['language']} | "
            f"{r['n']} | {fmt(r['cer'])} | {fmt(r['wer'])} | {fmt(r['exact'])} |"
        )

    lines.append("")
    lines.append("## 4. School Notebooks category breakdown\n")
    school = data["runs"]["school_notebooks"]

    for split in ["train", "val", "test"]:
        lines.append(f"### {split}")
        lines.append("| group | n | CER | WER | exact |")
        lines.append("|---|---:|---:|---:|---:|")
        for group, vals in sorted(school["grouped"][split].items()):
            lines.append(
                f"| `{group}` | {vals['n']} | {fmt(vals['cer'])} | "
                f"{fmt(vals['wer'])} | {fmt(vals['exact'])} |"
            )
        lines.append("")

    lines.append("## 5. Interpretation\n")
    lines.append(
        "IAM gives the lowest CER, but it is line-level English and should not be directly compared with word-level Russian crop datasets by exact match.\n"
    )
    lines.append(
        "Cyrillic Handwriting and HKR Words both improve strongly from mini10k to full training. HKR remains methodologically important because its split is text-grouped.\n"
    )
    lines.append(
        "School Notebooks full is the strongest Russian crop baseline by validation CER. The main `word|pupil_text` category is substantially easier than `word|pupil_comment`; phrase groups should be treated as secondary because of smaller sample counts.\n"
    )

    lines.append("## 6. Stage 3 conclusion\n")
    lines.append("```text")
    lines.append("[x] IAM full baseline")
    lines.append("[x] Cyrillic Handwriting full baseline")
    lines.append("[x] HKR Words full baseline")
    lines.append("[x] School Notebooks full baseline")
    lines.append("[x] blank-collapse resolved")
    lines.append("[x] image-only HTR baseline stage completed")
    lines.append("```")
    lines.append("")

    lines.append("## 7. Next recommended stage\n")
    lines.append(
        "Next: build mixed-dataset Cyrillic baselines, then add graph-aware features and compare image-only versus image+graph models.\n"
    )

    REPORT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("wrote:", REPORT_MD)
    print("wrote:", REPORT_JSON)


if __name__ == "__main__":
    main()