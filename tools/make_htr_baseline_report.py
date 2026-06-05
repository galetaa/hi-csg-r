from __future__ import annotations

import json
from pathlib import Path


RUN_NAME = "mini10k_cyrillic_v1"
OUT_DIR = Path("outputs/htr_baseline_v1")
REPORT = OUT_DIR / f"{RUN_NAME}_report.md"


def load_summary(split: str) -> dict:
    p = OUT_DIR / f"eval_{RUN_NAME}_{split}_final" / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    rows = {}
    for split in ["train", "val", "test"]:
        rows[split] = load_summary(split)

    lines = []
    lines.append(f"# HTR baseline report — {RUN_NAME}\n")
    lines.append("## Setup\n")
    lines.append("```text")
    lines.append("decode blank penalty selected on validation: -0.9")
    lines.append("model: height-preserving CRNN + BiLSTM + CTC")
    lines.append("dataset: Cyrillic Handwriting")
    lines.append("subset: mini10k")
    lines.append("input: OCR-preprocessed images")
    lines.append("target: transcription_modes.ctc_default")
    lines.append("blank penalty: scheduled during training")
    lines.append("```")
    lines.append("")

    lines.append("## Metrics\n")
    lines.append("| split | n | CER | WER | exact | pred_len | empty | blank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for split, s in rows.items():
        m = s["metrics"]
        lines.append(
            f"| {split} | {m['n']} | {m['cer']:.4f} | {m['wer']:.4f} | "
            f"{m['exact']:.4f} | {m['pred_len_mean']:.2f} | "
            f"{m['pred_empty_ratio']:.3f} | {m['argmax_blank_ratio']:.3f} |"
        )

    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "The mini10k run confirms that the CRNN-CTC pipeline is viable. "
        "The model strongly overfits the 10k training subset, but validation performance is already usable "
        "for a first image-only baseline. This run should be treated as a development baseline, not the final full-dataset result.\n"
    )

    lines.append("## Example predictions\n")
    test_preds = OUT_DIR / f"eval_{RUN_NAME}_test" / "predictions.jsonl"
    if test_preds.exists():
        examples = []
        with test_preds.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
                if len(examples) >= 30:
                    break

        lines.append("| target | pred | CER |")
        lines.append("|---|---|---:|")
        for e in examples:
            lines.append(f"| `{e['target']}` | `{e['pred']}` | {e['cer']:.3f} |")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote:", REPORT)


if __name__ == "__main__":
    main()