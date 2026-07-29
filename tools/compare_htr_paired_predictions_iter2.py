from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TEXT_LEN_BUCKETS = [
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-10", 7, 10),
    ("11+", 11, 10**9),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def text_len_bucket(text: str) -> str:
    n = len(text)
    for name, low, high in TEXT_LEN_BUCKETS:
        if low <= n <= high:
            return name
    return "0"


def dataset_from_prediction(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", ""))
    if sample_id.startswith("hkr_") or sample_id.startswith("hkr_words"):
        return "hkr_words"
    if (
        sample_id.startswith("cyr_")
        or sample_id.startswith("cyrillic_")
        or sample_id.startswith("cyrillic_handwriting")
    ):
        return "cyrillic_handwriting"
    if sample_id.startswith("school_") or sample_id.startswith("school_notebooks"):
        return "school_notebooks_clean"
    return str(row.get("dataset", "unknown"))


def read_predictions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row["dataset"] = dataset_from_prediction(row)
            out[str(row["sample_id"])] = row
    return out


def read_school_quality(quality_root: Path | None) -> dict[str, str]:
    if quality_root is None:
        return {}

    out: dict[str, str] = {}
    for split in ["test"]:
        for bucket in ["clean_core", "hard_real", "invalid_or_review"]:
            path = quality_root / f"{split}.{bucket}.jsonl"
            if not path.exists():
                continue
            for row in read_jsonl(path):
                out[str(row["sample_id"])] = bucket
    return out


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "mean_baseline_cer": None,
            "mean_augmented_cer": None,
            "mean_delta_cer": None,
            "median_delta_cer": None,
            "p05_delta_cer": None,
            "p95_delta_cer": None,
            "baseline_wer": None,
            "augmented_wer": None,
            "delta_wer": None,
            "baseline_exact": None,
            "augmented_exact": None,
            "delta_exact": None,
        }

    delta = np.asarray([row["delta_cer"] for row in rows], dtype=np.float64)
    base_cer = np.asarray([row["baseline_cer"] for row in rows], dtype=np.float64)
    aug_cer = np.asarray([row["augmented_cer"] for row in rows], dtype=np.float64)
    base_wer = np.asarray([row["baseline_wer"] for row in rows], dtype=np.float64)
    aug_wer = np.asarray([row["augmented_wer"] for row in rows], dtype=np.float64)
    base_exact = np.asarray([row["baseline_exact"] for row in rows], dtype=np.float64)
    aug_exact = np.asarray([row["augmented_exact"] for row in rows], dtype=np.float64)

    wins = int(np.sum(delta < 0))
    losses = int(np.sum(delta > 0))
    ties = int(np.sum(delta == 0))

    return {
        "n": len(rows),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(rows),
        "loss_rate": losses / len(rows),
        "tie_rate": ties / len(rows),
        "mean_baseline_cer": float(base_cer.mean()),
        "mean_augmented_cer": float(aug_cer.mean()),
        "mean_delta_cer": float(delta.mean()),
        "median_delta_cer": float(np.quantile(delta, 0.50)),
        "p05_delta_cer": float(np.quantile(delta, 0.05)),
        "p95_delta_cer": float(np.quantile(delta, 0.95)),
        "baseline_wer": float(base_wer.mean()),
        "augmented_wer": float(aug_wer.mean()),
        "delta_wer": float(aug_wer.mean() - base_wer.mean()),
        "baseline_exact": float(base_exact.mean()),
        "augmented_exact": float(aug_exact.mean()),
        "delta_exact": float(aug_exact.mean() - base_exact.mean()),
    }


def bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, float | None]:
    if not rows:
        return {
            "mean_delta_cer": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_bootstrap": n_bootstrap,
        }

    rng = np.random.default_rng(seed)
    values = np.asarray([row["delta_cer"] for row in rows], dtype=np.float64)
    n = values.size
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()

    return {
        "mean_delta_cer": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_bootstrap": n_bootstrap,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_predictions", nargs="+", required=True)
    parser.add_argument("--augmented_predictions", required=True)
    parser.add_argument("--quality_root", default=None)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    args = parser.parse_args()

    baseline = read_predictions([Path(p) for p in args.baseline_predictions])
    augmented = read_predictions([Path(args.augmented_predictions)])
    school_quality = read_school_quality(Path(args.quality_root) if args.quality_root else None)

    common = sorted(set(baseline) & set(augmented))
    if not common:
        raise RuntimeError("No common sample_id between baseline and augmented predictions")

    paired = []
    for sample_id in common:
        b = baseline[sample_id]
        a = augmented[sample_id]
        target = str(a.get("target", b.get("target", "")))
        dataset = str(a.get("dataset") or b.get("dataset") or dataset_from_prediction(a))
        row = {
            "sample_id": sample_id,
            "dataset": dataset,
            "target": target,
            "text_len": len(target),
            "text_len_bucket": text_len_bucket(target),
            "school_quality_bucket": school_quality.get(sample_id, ""),
            "baseline_pred": b.get("pred", ""),
            "augmented_pred": a.get("pred", ""),
            "baseline_cer": float(b["cer"]),
            "augmented_cer": float(a["cer"]),
            "delta_cer": float(a["cer"]) - float(b["cer"]),
            "baseline_wer": float(b["wer"]),
            "augmented_wer": float(a["wer"]),
            "baseline_exact": float(b.get("exact", 0.0)),
            "augmented_exact": float(a.get("exact", 0.0)),
        }
        paired.append(row)

    by_dataset = {
        dataset: summarize_pairs([row for row in paired if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in paired})
    }
    by_text_len = {
        name: summarize_pairs([row for row in paired if row["text_len_bucket"] == name])
        for name, _, _ in TEXT_LEN_BUCKETS
    }
    by_school_quality = {
        bucket: summarize_pairs([
            row for row in paired
            if row["dataset"] == "school_notebooks_clean"
            and row["school_quality_bucket"] == bucket
        ])
        for bucket in ["clean_core", "hard_real", "invalid_or_review", ""]
    }

    result = {
        "n_baseline": len(baseline),
        "n_augmented": len(augmented),
        "n_common": len(common),
        "baseline_only": len(set(baseline) - set(augmented)),
        "augmented_only": len(set(augmented) - set(baseline)),
        "overall": summarize_pairs(paired),
        "by_dataset": by_dataset,
        "by_text_len_bucket": by_text_len,
        "by_school_quality_bucket": by_school_quality,
        "bootstrap": {
            "overall": bootstrap_ci(
                paired,
                seed=args.seed,
                n_bootstrap=args.n_bootstrap,
            ),
            "school_notebooks_clean": bootstrap_ci(
                [row for row in paired if row["dataset"] == "school_notebooks_clean"],
                seed=args.seed + 1,
                n_bootstrap=args.n_bootstrap,
            ),
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    write_json(out_json, result)

    lines = [
        "# Paired HTR Comparison - Iteration 2",
        "",
        f"Common samples: {result['n_common']}",
        "",
        "## Overall",
        "",
        "| n | wins | losses | ties | baseline CER | augmented CER | delta CER | CER CI95 | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = result["overall"]
    ci = result["bootstrap"]["overall"]
    lines.append(
        f"| {overall['n']} | {overall['wins']} | {overall['losses']} | {overall['ties']} | "
        f"{fmt(overall['mean_baseline_cer'])} | {fmt(overall['mean_augmented_cer'])} | {fmt(overall['mean_delta_cer'])} | "
        f"[{fmt(ci['ci95_low'])}, {fmt(ci['ci95_high'])}] | "
        f"{fmt(overall['baseline_wer'])} | {fmt(overall['augmented_wer'])} | {fmt(overall['delta_wer'])} | "
        f"{fmt(overall['baseline_exact'])} | {fmt(overall['augmented_exact'])} | {fmt(overall['delta_exact'])} |"
    )

    lines.extend([
        "",
        "## By Dataset",
        "",
        "| dataset | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline WER | augmented WER | delta WER | baseline exact | augmented exact | delta exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, row in result["by_dataset"].items():
        lines.append(
            f"| `{dataset}` | {row['n']} | {row['wins']} | {row['losses']} | {row['ties']} | "
            f"{fmt(row['mean_baseline_cer'])} | {fmt(row['mean_augmented_cer'])} | {fmt(row['mean_delta_cer'])} | "
            f"{fmt(row['baseline_wer'])} | {fmt(row['augmented_wer'])} | {fmt(row['delta_wer'])} | "
            f"{fmt(row['baseline_exact'])} | {fmt(row['augmented_exact'])} | {fmt(row['delta_exact'])} |"
        )

    lines.extend([
        "",
        "## By Text Length",
        "",
        "| text_len | n | wins | losses | ties | baseline CER | augmented CER | delta CER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for bucket, row in result["by_text_len_bucket"].items():
        lines.append(
            f"| `{bucket}` | {row['n']} | {row['wins']} | {row['losses']} | {row['ties']} | "
            f"{fmt(row['mean_baseline_cer'])} | {fmt(row['mean_augmented_cer'])} | {fmt(row['mean_delta_cer'])} |"
        )

    school_ci = result["bootstrap"]["school_notebooks_clean"]
    lines.extend([
        "",
        "## School Quality Buckets",
        "",
        f"School CER delta bootstrap CI95: [{fmt(school_ci['ci95_low'])}, {fmt(school_ci['ci95_high'])}]",
        "",
        "| bucket | n | wins | losses | ties | baseline CER | augmented CER | delta CER | baseline exact | augmented exact | delta exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for bucket, row in result["by_school_quality_bucket"].items():
        label = bucket or "unknown"
        lines.append(
            f"| `{label}` | {row['n']} | {row['wins']} | {row['losses']} | {row['ties']} | "
            f"{fmt(row['mean_baseline_cer'])} | {fmt(row['mean_augmented_cer'])} | {fmt(row['mean_delta_cer'])} | "
            f"{fmt(row['baseline_exact'])} | {fmt(row['augmented_exact'])} | {fmt(row['delta_exact'])} |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "overall": result["overall"],
        "bootstrap": result["bootstrap"],
        "dataset_delta_cer": {
            key: value["mean_delta_cer"]
            for key, value in by_dataset.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
