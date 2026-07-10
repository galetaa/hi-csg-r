from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("outputs/htr_baseline_v1")
REPORT_MD = OUT_ROOT / "htr_mixed_cyrillic_report.md"
REPORT_JSON = OUT_ROOT / "htr_mixed_cyrillic_report.json"


SINGLE = {
    "cyrillic_handwriting": {
        "title": "Cyrillic Handwriting",
        "eval": {
            "val": OUT_ROOT / "eval_full_cyrillic_v1_val_final",
            "test": OUT_ROOT / "eval_full_cyrillic_v1_test_final",
        },
    },
    "hkr_words": {
        "title": "HKR Words",
        "eval": {
            "val": OUT_ROOT / "eval_full_hkr_words_v1_val_final",
            "test": OUT_ROOT / "eval_full_hkr_words_v1_test_final",
        },
    },
    "school_notebooks_clean": {
        "title": "School Notebooks Clean",
        "eval": {
            "val": OUT_ROOT / "eval_full_school_notebooks_v1_val_final",
            "test": OUT_ROOT / "eval_full_school_notebooks_v1_test_final",
        },
    },
}


MIXED = {
    "balanced50k": {
        "title": "Mixed Cyrillic balanced50k v1",
        "prefix": "eval_mixed_cyrillic_balanced50k_v1",
        "description": "Balanced training: 50k samples from each Cyrillic dataset.",
        "penalty": -0.2,
    },
    "natural_full": {
        "title": "Mixed Cyrillic natural-full v1",
        "prefix": "eval_mixed_cyrillic_natural_full_v1",
        "description": "Natural full training: all available train samples from each Cyrillic dataset.",
        "penalty": -0.4,
    },
}


def load_summary(eval_dir: Path) -> dict[str, Any]:
    return json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))


def metrics(eval_dir: Path) -> dict[str, Any]:
    s = load_summary(eval_dir)
    m = s["metrics"]
    return {
        "n": m["n"],
        "cer": m["cer"],
        "wer": m["wer"],
        "exact": m["exact"],
        "pred_len": m["pred_len_mean"],
        "empty": m["pred_empty_ratio"],
        "blank": m["argmax_blank_ratio"],
        "penalty": s["blank_logit_penalty"],
        "epoch": s["checkpoint_epoch"],
        "checkpoint_val_cer": s["checkpoint_val_cer"],
    }


def rel_improvement(old: float, new: float) -> float:
    return (old - new) / old if old else 0.0


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    data: dict[str, Any] = {
        "single": {},
        "mixed": {},
        "comparison": {},
    }

    for ds_key, ds in SINGLE.items():
        data["single"][ds_key] = {
            "title": ds["title"],
            "splits": {
                split: metrics(eval_dir)
                for split, eval_dir in ds["eval"].items()
            },
        }

    for mixed_key, cfg in MIXED.items():
        data["mixed"][mixed_key] = {
            "title": cfg["title"],
            "description": cfg["description"],
            "splits": {},
        }

        for ds_key in SINGLE:
            data["mixed"][mixed_key]["splits"][ds_key] = {}
            for split in ["val", "test"]:
                eval_dir = OUT_ROOT / f"{cfg['prefix']}_{ds_key}_{split}_final"
                data["mixed"][mixed_key]["splits"][ds_key][split] = metrics(eval_dir)

    for ds_key, ds in SINGLE.items():
        single_test = data["single"][ds_key]["splits"]["test"]["cer"]
        balanced_test = data["mixed"]["balanced50k"]["splits"][ds_key]["test"]["cer"]
        natural_test = data["mixed"]["natural_full"]["splits"][ds_key]["test"]["cer"]

        data["comparison"][ds_key] = {
            "title": ds["title"],
            "single_test_cer": single_test,
            "balanced50k_test_cer": balanced_test,
            "natural_full_test_cer": natural_test,
            "balanced_vs_single_rel_improvement": rel_improvement(single_test, balanced_test),
            "natural_vs_single_rel_improvement": rel_improvement(single_test, natural_test),
            "natural_vs_balanced_rel_improvement": rel_improvement(balanced_test, natural_test),
        }

    lines = []
    lines.append("# Mixed Cyrillic image-only baselines report — Stage 3.3\n")

    lines.append("## 1. Purpose\n")
    lines.append(
        "This report compares single-dataset Cyrillic HTR baselines with mixed-dataset image-only baselines. "
        "The goal is to determine whether a universal Cyrillic CRNN-CTC model improves cross-domain recognition before graph-aware experiments.\n"
    )

    lines.append("## 2. Mixed runs\n")
    lines.append("| run | training composition | selected penalty |")
    lines.append("|---|---|---:|")
    for mixed_key, cfg in MIXED.items():
        lines.append(
            f"| {cfg['title']} | {cfg['description']} | {fmt(cfg['penalty'], 1)} |"
        )
    lines.append("")

    lines.append("## 3. Test CER comparison\n")
    lines.append(
        "| dataset | single full CER | mixed balanced50k CER | mixed natural-full CER | natural-full vs single |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    for ds_key, row in data["comparison"].items():
        lines.append(
            f"| {row['title']} | "
            f"{fmt(row['single_test_cer'])} | "
            f"{fmt(row['balanced50k_test_cer'])} | "
            f"{fmt(row['natural_full_test_cer'])} | "
            f"{fmt(100 * row['natural_vs_single_rel_improvement'], 1)}% |"
        )
    lines.append("")

    lines.append("## 4. Full per-dataset metrics for mixed natural-full\n")
    lines.append("| dataset | split | n | CER | WER | exact | pred_len | blank | penalty | epoch |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    natural = data["mixed"]["natural_full"]
    for ds_key, ds in SINGLE.items():
        for split in ["val", "test"]:
            r = natural["splits"][ds_key][split]
            lines.append(
                f"| {ds['title']} | {split} | {r['n']} | "
                f"{fmt(r['cer'])} | {fmt(r['wer'])} | {fmt(r['exact'])} | "
                f"{fmt(r['pred_len'], 2)} | {fmt(r['blank'], 3)} | "
                f"{fmt(r['penalty'], 1)} | {r['epoch']} |"
            )
    lines.append("")

    lines.append("## 5. Interpretation\n")
    lines.append(
        "Mixed Cyrillic natural-full improves all three Cyrillic test sets relative to their single-dataset full baselines. "
        "The largest gain is on HKR Words, suggesting that additional Cyrillic domains provide useful regularization and character-shape coverage.\n"
    )
    lines.append(
        "The balanced50k run already improves Cyrillic Handwriting and HKR Words but hurts School Notebooks. "
        "The natural-full run fixes this by restoring the full School Notebooks training mass while preserving the gains on the other datasets.\n"
    )
    lines.append(
        "Therefore, `mixed_cyrillic_natural_full_v1` should be treated as the primary Cyrillic image-only baseline before graph-aware experiments.\n"
    )

    lines.append("## 6. Stage 3.3 conclusion\n")
    lines.append("```text")
    lines.append("[x] mixed balanced50k baseline")
    lines.append("[x] mixed natural-full baseline")
    lines.append("[x] per-dataset validation/test evaluation")
    lines.append("[x] universal Cyrillic image-only baseline selected")
    lines.append("primary baseline: mixed_cyrillic_natural_full_v1")
    lines.append("```")
    lines.append("")

    lines.append("## 7. Next stage\n")
    lines.append(
        "Next: Stage 4 graph-aware experiments. Start with a lightweight graph-feature fusion baseline before moving to full graph neural models.\n"
    )

    REPORT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("wrote:", REPORT_MD)
    print("wrote:", REPORT_JSON)


if __name__ == "__main__":
    main()