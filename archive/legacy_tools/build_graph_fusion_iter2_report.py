from __future__ import annotations

import argparse
import json
import re
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

DATASETS = [
    "hkr_words",
    "cyrillic_handwriting",
    "school_notebooks_clean",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_summary(obj: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(obj.get("metrics") or obj)
    grouped = metrics.get("grouped", {})
    if grouped and not any(key in grouped for key in DATASETS):
        collapsed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for key, value in grouped.items():
            dataset = str(key).split("|", 1)[0]
            collapsed[dataset].append(value)

        merged = {}
        for dataset, rows in collapsed.items():
            n_total = sum(int(row.get("n", 0)) for row in rows)
            merged[dataset] = {
                "n": n_total,
                "cer": sum(float(row.get("cer", 0.0)) * int(row.get("n", 0)) for row in rows) / max(n_total, 1),
                "wer": sum(float(row.get("wer", 0.0)) * int(row.get("n", 0)) for row in rows) / max(n_total, 1),
                "exact": sum(float(row.get("exact", 0.0)) * int(row.get("n", 0)) for row in rows) / max(n_total, 1),
            }
        metrics["grouped"] = merged
    return metrics


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def edit_distance(a: list[Any], b: list[Any]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def prediction_metrics(target: str, pred: str) -> dict[str, float]:
    target_words = target.split()
    pred_words = pred.split()
    return {
        "cer": edit_distance(list(target), list(pred)) / max(len(target), 1),
        "wer": edit_distance(target_words, pred_words) / max(len(target_words), 1),
        "exact": float(target == pred),
    }


def dataset_from_sample_id(sample_id: str, fallback: str = "unknown") -> str:
    if sample_id.startswith(("hkr_", "hkr_words")):
        return "hkr_words"
    if sample_id.startswith(("cyr_", "cyr_word_", "cyrillic_", "cyrillic_handwriting")):
        return "cyrillic_handwriting"
    if sample_id.startswith(("school_", "school_notebooks")):
        return "school_notebooks_clean"
    return fallback


def dataset_from_path(path: Path) -> str:
    name = str(path)
    for dataset in DATASETS:
        if dataset in name:
            return dataset
    return Path(path).parent.name


def read_predictions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            sample_id = str(row["sample_id"])
            target = str(row.get("target", ""))
            pred = str(row.get("pred", ""))
            if not {"cer", "wer", "exact"} <= set(row):
                row.update(prediction_metrics(target, pred))
            row["dataset"] = dataset_from_sample_id(
                sample_id,
                str(row.get("dataset", "unknown")),
            )
            out[sample_id] = row
    return out


def read_school_quality(quality_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for bucket in ["clean_core", "hard_real", "invalid_or_review"]:
        path = quality_root / f"test.{bucket}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            out[str(row["sample_id"])] = bucket
    return out


def read_graph_valid(manifest: Path) -> dict[str, bool]:
    out = {}
    for row in read_jsonl(manifest):
        out[str(row["sample_id"])] = bool(row.get("graph_valid", True))
    return out


def text_len_bucket(text: str) -> str:
    n = len(text)
    for name, low, high in TEXT_LEN_BUCKETS:
        if low <= n <= high:
            return name
    return "0"


def token_type(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= 3:
        return "short_1_3"
    has_alpha = bool(re.search(r"[A-Za-zА-Яа-яЁё]", stripped))
    has_digit = bool(re.search(r"\d", stripped))
    has_other = bool(re.search(r"[^A-Za-zА-Яа-яЁё\d\s]", stripped))
    if has_digit and not has_alpha and not has_other:
        return "numeric"
    if has_alpha and not has_digit and not has_other:
        return "alpha"
    if has_alpha or has_digit:
        return "mixed"
    return "punctuation"


def aggregate_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, float | int | None]:
    if not rows:
        return {"n": 0, "cer": None, "wer": None, "exact": None}
    char_dist = 0
    char_total = 0
    word_dist = 0
    word_total = 0
    exact = 0
    for row in rows:
        target = str(row["target"])
        pred = str(row[pred_key])
        char_dist += edit_distance(list(target), list(pred))
        char_total += max(len(target), 1)
        word_dist += edit_distance(target.split(), pred.split())
        word_total += max(len(target.split()), 1)
        exact += int(target == pred)
    return {
        "n": len(rows),
        "cer": char_dist / max(char_total, 1),
        "wer": word_dist / max(word_total, 1),
        "exact": exact / max(len(rows), 1),
    }


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "baseline": aggregate_metrics([], "baseline_pred"),
            "candidate": aggregate_metrics([], "candidate_pred"),
            "delta_cer": None,
            "delta_wer": None,
            "delta_exact": None,
            "mean_delta_cer_per_sample": None,
        }

    baseline = aggregate_metrics(rows, "baseline_pred")
    candidate = aggregate_metrics(rows, "candidate_pred")
    deltas = np.asarray([row["delta_cer"] for row in rows], dtype=np.float64)
    wins = int(np.sum(deltas < 0))
    losses = int(np.sum(deltas > 0))
    ties = int(np.sum(deltas == 0))
    return {
        "n": len(rows),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "baseline": baseline,
        "candidate": candidate,
        "delta_cer": (
            float(candidate["cer"] - baseline["cer"])
            if candidate["cer"] is not None and baseline["cer"] is not None
            else None
        ),
        "delta_wer": (
            float(candidate["wer"] - baseline["wer"])
            if candidate["wer"] is not None and baseline["wer"] is not None
            else None
        ),
        "delta_exact": (
            float(candidate["exact"] - baseline["exact"])
            if candidate["exact"] is not None and baseline["exact"] is not None
            else None
        ),
        "mean_delta_cer_per_sample": float(deltas.mean()),
        "median_delta_cer_per_sample": float(np.quantile(deltas, 0.50)),
        "p05_delta_cer_per_sample": float(np.quantile(deltas, 0.05)),
        "p95_delta_cer_per_sample": float(np.quantile(deltas, 0.95)),
    }


def bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean_delta_cer_per_sample": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_bootstrap": n_bootstrap,
        }
    values = np.asarray([row["delta_cer"] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n = len(values)
    for i in range(n_bootstrap):
        means[i] = rng.choice(values, size=n, replace=True).mean()
    return {
        "n": n,
        "mean_delta_cer_per_sample": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_bootstrap": n_bootstrap,
    }


def build_pairs(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    school_quality: dict[str, str],
    graph_valid: dict[str, bool],
) -> list[dict[str, Any]]:
    common = sorted(set(baseline) & set(candidate))
    if not common:
        raise RuntimeError("No common predictions")
    rows = []
    for sample_id in common:
        b = baseline[sample_id]
        c = candidate[sample_id]
        target = str(c.get("target") or b.get("target") or "")
        b_pred = str(b.get("pred", ""))
        c_pred = str(c.get("pred", ""))
        b_metrics = prediction_metrics(target, b_pred)
        c_metrics = prediction_metrics(target, c_pred)
        dataset = dataset_from_sample_id(
            sample_id,
            str(c.get("dataset") or b.get("dataset") or "unknown"),
        )
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "target": target,
                "baseline_pred": b_pred,
                "candidate_pred": c_pred,
                "baseline_cer": b_metrics["cer"],
                "candidate_cer": c_metrics["cer"],
                "delta_cer": c_metrics["cer"] - b_metrics["cer"],
                "baseline_wer": b_metrics["wer"],
                "candidate_wer": c_metrics["wer"],
                "baseline_exact": b_metrics["exact"],
                "candidate_exact": c_metrics["exact"],
                "text_len": len(target),
                "text_len_bucket": text_len_bucket(target),
                "token_type": token_type(target),
                "school_quality_bucket": (
                    school_quality.get(sample_id, "")
                    if dataset == "school_notebooks_clean"
                    else ""
                ),
                "graph_valid": graph_valid.get(sample_id),
            }
        )
    return rows


def grouped_summary(
    pairs: list[dict[str, Any]],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    by_dataset = {
        dataset: summarize_pairs([row for row in pairs if row["dataset"] == dataset])
        for dataset in DATASETS
    }
    by_text_len = {
        bucket: summarize_pairs([row for row in pairs if row["text_len_bucket"] == bucket])
        for bucket, _, _ in TEXT_LEN_BUCKETS
    }
    token_buckets = ["alpha", "mixed", "numeric", "punctuation", "short_1_3"]
    by_token_type = {
        bucket: summarize_pairs([row for row in pairs if row["token_type"] == bucket])
        for bucket in token_buckets
    }
    by_school_quality = {
        bucket: summarize_pairs(
            [
                row
                for row in pairs
                if row["dataset"] == "school_notebooks_clean"
                and row["school_quality_bucket"] == bucket
            ]
        )
        for bucket in ["clean_core", "hard_real", "invalid_or_review", ""]
    }
    by_graph_valid = {
        str(value): summarize_pairs([row for row in pairs if row["graph_valid"] is value])
        for value in [True, False]
    }

    bootstrap = {
        "overall": bootstrap_ci(pairs, seed=seed, n_bootstrap=n_bootstrap),
    }
    for i, dataset in enumerate(DATASETS, start=1):
        bootstrap[dataset] = bootstrap_ci(
            [row for row in pairs if row["dataset"] == dataset],
            seed=seed + i,
            n_bootstrap=n_bootstrap,
        )

    return {
        "n_common": len(pairs),
        "overall": summarize_pairs(pairs),
        "by_dataset": by_dataset,
        "by_school_quality_bucket": by_school_quality,
        "by_text_len_bucket": by_text_len,
        "by_token_type": by_token_type,
        "by_graph_valid": by_graph_valid,
        "bootstrap": bootstrap,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def ci_fmt(row: dict[str, Any]) -> str:
    return f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]"


def metric_row(label: str, metrics: dict[str, Any]) -> str:
    grouped = metrics.get("grouped", {})
    school = grouped.get("school_notebooks_clean", {})
    return (
        f"| {label} | {fmt(metrics.get('cer'))} | {fmt(metrics.get('wer'))} | "
        f"{fmt(metrics.get('exact'))} | {fmt(grouped.get('hkr_words', {}).get('cer'))} | "
        f"{fmt(grouped.get('cyrillic_handwriting', {}).get('cer'))} | "
        f"{fmt(school.get('cer'))} | {fmt(school.get('wer'))} | {fmt(school.get('exact'))} |"
    )


def comparison_md(title: str, result: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        f"Common samples: {result['n_common']}",
        "",
        "## Paired Bootstrap",
        "",
        "| scope | n | mean per-sample ΔCER | 95% CI | aggregate ΔCER | wins | losses | ties |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ["overall", *DATASETS]:
        summary = result["overall"] if scope == "overall" else result["by_dataset"][scope]
        boot = result["bootstrap"][scope]
        lines.append(
            f"| `{scope}` | {summary['n']} | {fmt(boot['mean_delta_cer_per_sample'])} | "
            f"{ci_fmt(boot)} | {fmt(summary['delta_cer'])} | "
            f"{summary['wins']} | {summary['losses']} | {summary['ties']} |"
        )

    def add_group(title: str, key: str) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| bucket | n | baseline CER | candidate CER | aggregate ΔCER | baseline WER | candidate WER | baseline exact | candidate exact | wins | losses | ties |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for bucket, row in result[key].items():
            lines.append(
                f"| `{bucket or 'unknown'}` | {row['n']} | "
                f"{fmt(row['baseline']['cer'])} | {fmt(row['candidate']['cer'])} | {fmt(row['delta_cer'])} | "
                f"{fmt(row['baseline']['wer'])} | {fmt(row['candidate']['wer'])} | "
                f"{fmt(row['baseline']['exact'])} | {fmt(row['candidate']['exact'])} | "
                f"{row['wins']} | {row['losses']} | {row['ties']} |"
            )

    add_group("By Dataset", "by_dataset")
    add_group("School Quality", "by_school_quality_bucket")
    add_group("By Text Length", "by_text_len_bucket")
    add_group("By Token Type", "by_token_type")
    add_group("By Graph Valid", "by_graph_valid")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image10k_predictions", required=True)
    parser.add_argument("--baseline_predictions", nargs="+", required=True)
    parser.add_argument("--graph_predictions", required=True)
    parser.add_argument("--zero_graph_summary", required=True)
    parser.add_argument("--image10k_summary", required=True)
    parser.add_argument("--graph_summary", required=True)
    parser.add_argument("--baseline_summary", nargs="+", required=True)
    parser.add_argument("--quality_root", required=True)
    parser.add_argument("--fusion_manifest_test", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    args = parser.parse_args()

    image10k = read_predictions([Path(args.image10k_predictions)])
    baseline = read_predictions([Path(path) for path in args.baseline_predictions])
    graph = read_predictions([Path(args.graph_predictions)])
    school_quality = read_school_quality(Path(args.quality_root))
    graph_valid = read_graph_valid(Path(args.fusion_manifest_test))

    paired_vs_image = grouped_summary(
        build_pairs(
            image10k,
            graph,
            school_quality=school_quality,
            graph_valid=graph_valid,
        ),
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    paired_vs_baseline = grouped_summary(
        build_pairs(
            baseline,
            graph,
            school_quality=school_quality,
            graph_valid=graph_valid,
        ),
        seed=args.seed + 100,
        n_bootstrap=args.n_bootstrap,
    )

    image10k_summary = normalize_summary(read_json(Path(args.image10k_summary)))
    graph_summary = normalize_summary(read_json(Path(args.graph_summary)))
    zero_graph_summary = normalize_summary(read_json(Path(args.zero_graph_summary)))

    baseline_grouped: dict[str, Any] = {}
    baseline_n = 0
    for path in args.baseline_summary:
        summary = normalize_summary(read_json(Path(path)))
        grouped = summary.get("grouped", {})
        if grouped:
            baseline_grouped.update(grouped)
            baseline_n += int(summary.get("n", 0))
        else:
            dataset = str(summary.get("dataset") or dataset_from_path(Path(path)))
            baseline_grouped[dataset] = summary
            baseline_n += int(summary.get("n", 0))

    baseline_overall = aggregate_metrics(
        [
            {
                "target": row["target"],
                "baseline_pred": row["pred"],
            }
            for row in baseline.values()
        ],
        "baseline_pred",
    )
    baseline_summary = {
        **baseline_overall,
        "grouped": baseline_grouped,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "models": {
            "baseline_image_only": baseline_summary,
            "image_only_10k": image10k_summary,
            "graph_fusion": graph_summary,
            "zero_graph": zero_graph_summary,
        },
        "paired_vs_image10k": paired_vs_image,
        "paired_vs_baseline": paired_vs_baseline,
        "notes": {
            "bootstrap_delta": "candidate per-sample CER minus baseline per-sample CER",
            "aggregate_delta": "candidate aggregate CER minus baseline aggregate CER",
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
    }

    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "paired_vs_image10k.json", paired_vs_image)
    write_json(out_dir / "paired_vs_baseline.json", paired_vs_baseline)

    (out_dir / "paired_vs_image10k.md").write_text(
        comparison_md("Graph Fusion vs Image-only +10k Context", paired_vs_image),
        encoding="utf-8",
    )
    (out_dir / "paired_vs_baseline.md").write_text(
        comparison_md("Graph Fusion vs Baseline Image-only", paired_vs_baseline),
        encoding="utf-8",
    )

    lines = [
        "# Graph Fusion Iteration 2 Context-10k Pilot",
        "",
        "## Metrics",
        "",
        "| model | overall CER | overall WER | exact | HKR CER | Cyrillic CER | School CER | School WER | School exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("image-only +10k", image10k_summary),
        metric_row("graph-fusion", graph_summary),
        metric_row("zero-graph", zero_graph_summary),
        "",
        "## Paired Bootstrap vs Image-only +10k",
        "",
        "| scope | mean per-sample ΔCER | 95% CI | aggregate ΔCER |",
        "|---|---:|---:|---:|",
    ]
    for scope in ["overall", *DATASETS]:
        row = paired_vs_image["bootstrap"][scope]
        summary_row = paired_vs_image["overall"] if scope == "overall" else paired_vs_image["by_dataset"][scope]
        lines.append(
            f"| `{scope}` | {fmt(row['mean_delta_cer_per_sample'])} | {ci_fmt(row)} | {fmt(summary_row['delta_cer'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired Bootstrap vs Baseline Image-only",
            "",
            "| scope | mean per-sample ΔCER | 95% CI | aggregate ΔCER |",
            "|---|---:|---:|---:|",
        ]
    )
    for scope in ["overall", *DATASETS]:
        row = paired_vs_baseline["bootstrap"][scope]
        summary_row = paired_vs_baseline["overall"] if scope == "overall" else paired_vs_baseline["by_dataset"][scope]
        lines.append(
            f"| `{scope}` | {fmt(row['mean_delta_cer_per_sample'])} | {ci_fmt(row)} | {fmt(summary_row['delta_cer'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Graph-fusion is compared against the strongest image-only +10k contextual-line model and the original image-only baseline. "
            "The zero-graph ablation checks whether the trained model actually uses graph inputs.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    card_lines = [
        "# Result Card: Graph Fusion Iteration 2 Context-10k Pilot",
        "",
        "## Setup",
        "",
        "- Control: image-only +10k contextual School line augmentation.",
        "- Candidate: image + 39 lineaware graph features with invalid graph rows masked.",
        "- Contextual line train samples use `graph_valid=false`; word-level samples use graph features.",
        "- Test split is unchanged word-level tri10k mixed.",
        "",
        "## Main Table",
        "",
        "| model | overall CER | overall WER | exact | HKR CER | Cyrillic CER | School CER | School WER | School exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("image-only +10k", image10k_summary),
        metric_row("graph-fusion", graph_summary),
        metric_row("zero-graph", zero_graph_summary),
        "",
        "## Key Paired CI vs Image-only +10k",
        "",
        "| scope | mean per-sample ΔCER | 95% CI | aggregate ΔCER |",
        "|---|---:|---:|---:|",
    ]
    for scope in ["overall", *DATASETS]:
        row = paired_vs_image["bootstrap"][scope]
        summary_row = paired_vs_image["overall"] if scope == "overall" else paired_vs_image["by_dataset"][scope]
        card_lines.append(
            f"| `{scope}` | {fmt(row['mean_delta_cer_per_sample'])} | {ci_fmt(row)} | {fmt(summary_row['delta_cer'])} |"
        )
    card_lines.extend(
        [
            "",
            "## Preliminary Conclusion",
            "",
            "The graph branch is not ignored: zero-graph inference degrades strongly. "
            "The final interpretation depends on the paired CIs: the pilot should be treated as targeted if School improves but Cyrillic degrades.",
        ]
    )
    (out_dir / "result_card.md").write_text(
        "\n".join(card_lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "out_dir": str(out_dir),
        "vs_image10k": {
            scope: paired_vs_image["bootstrap"][scope]
            for scope in ["overall", *DATASETS]
        },
        "vs_baseline": {
            scope: paired_vs_baseline["bootstrap"][scope]
            for scope in ["overall", *DATASETS]
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
