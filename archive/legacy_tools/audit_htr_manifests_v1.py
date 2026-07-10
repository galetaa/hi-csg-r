from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_ROOT = Path("data/experiments/htr_baseline_v1")

DATASETS = [
    "iam",
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks_clean",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stats(xs: list[float | int]) -> dict[str, Any]:
    if not xs:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    ys = sorted(float(x) for x in xs)

    def q(p: float) -> float:
        if not ys:
            return None
        idx = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * p))))
        return ys[idx]

    return {
        "count": len(ys),
        "min": ys[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": ys[-1],
        "mean": sum(ys) / len(ys),
    }


def ctc_min_timesteps(text: str) -> int:
    """
    Conservative CTC alignment lower bound.

    Basic target length is len(text).
    Consecutive equal characters need extra separation steps.
    Example: 'лл' usually needs l blank l, so +1.
    """
    if not text:
        return 0

    repeats = 0
    for a, b in zip(text, text[1:]):
        if a == b:
            repeats += 1

    return len(text) + repeats


def estimate_timesteps(width: int, time_downsample: int) -> int:
    """
    Approximate CRNN time steps after CNN width downsampling.

    We use floor because it is conservative for CTC feasibility.
    """
    return max(1, width // max(1, time_downsample))


def inspect_image(path: str | Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as img:
            w, h = img.size
            return int(w), int(h), img.mode
    except Exception:
        return None, None, None


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    time_downsample: int,
    max_bad_examples: int,
) -> dict[str, Any]:
    widths = []
    heights = []
    text_lens = []
    ctc_min_lens = []
    estimated_ts = []
    ratios_text_to_t = []
    ratios_ctcmin_to_t = []

    missing_images = 0
    unreadable_images = 0
    empty_text = 0

    bad_target_gt_t = []
    bad_ctcmin_gt_t = []
    suspicious_short_t = []
    suspicious_long_text = []

    by_split = Counter()
    by_level = Counter()
    by_category = Counter()
    chars = Counter()
    char_by_split = defaultdict(Counter)

    image_modes = Counter()

    for r in rows:
        split = r.get("split")
        level = r.get("level")
        category = r.get("category")
        text = str(r.get("text") or "")
        image_path = r.get("image_path")

        by_split[split] += 1
        by_level[level] += 1
        by_category[category] += 1

        if not text:
            empty_text += 1

        chars.update(text)
        char_by_split[split].update(text)

        if not image_path or not Path(image_path).exists():
            missing_images += 1
            continue

        w, h, mode = inspect_image(image_path)

        if w is None or h is None:
            unreadable_images += 1
            continue

        widths.append(w)
        heights.append(h)
        image_modes[mode] += 1

        target_len = len(text)
        ctc_min = ctc_min_timesteps(text)
        t = estimate_timesteps(w, time_downsample)

        text_lens.append(target_len)
        ctc_min_lens.append(ctc_min)
        estimated_ts.append(t)

        ratios_text_to_t.append(target_len / max(t, 1))
        ratios_ctcmin_to_t.append(ctc_min / max(t, 1))

        if target_len > t:
            bad_target_gt_t.append({
                "sample_id": r.get("sample_id"),
                "split": split,
                "image_path": image_path,
                "width": w,
                "height": h,
                "text": text,
                "text_len": target_len,
                "estimated_timesteps": t,
            })

        if ctc_min > t:
            bad_ctcmin_gt_t.append({
                "sample_id": r.get("sample_id"),
                "split": split,
                "image_path": image_path,
                "width": w,
                "height": h,
                "text": text,
                "text_len": target_len,
                "ctc_min_timesteps": ctc_min,
                "estimated_timesteps": t,
            })

        if t < 4:
            suspicious_short_t.append({
                "sample_id": r.get("sample_id"),
                "split": split,
                "image_path": image_path,
                "width": w,
                "height": h,
                "text": text,
                "text_len": target_len,
                "estimated_timesteps": t,
            })

        if target_len >= 80:
            suspicious_long_text.append({
                "sample_id": r.get("sample_id"),
                "split": split,
                "image_path": image_path,
                "width": w,
                "height": h,
                "text": text,
                "text_len": target_len,
                "estimated_timesteps": t,
            })

    train_chars = set(char_by_split["train"])
    val_chars = set(char_by_split["val"])
    test_chars = set(char_by_split["test"])

    unseen_val = sorted(val_chars - train_chars)
    unseen_test = sorted(test_chars - train_chars)

    return {
        "dataset": dataset,
        "num_records": len(rows),
        "splits": dict(by_split),
        "levels": dict(by_level),
        "categories": dict(by_category),
        "image_modes": dict(image_modes),
        "missing_images": missing_images,
        "unreadable_images": unreadable_images,
        "empty_text": empty_text,
        "time_downsample": time_downsample,
        "image_width": stats(widths),
        "image_height": stats(heights),
        "text_len": stats(text_lens),
        "ctc_min_timesteps": stats(ctc_min_lens),
        "estimated_timesteps": stats(estimated_ts),
        "text_len_to_timesteps_ratio": stats(ratios_text_to_t),
        "ctc_min_to_timesteps_ratio": stats(ratios_ctcmin_to_t),
        "num_chars": len(chars),
        "top_chars": chars.most_common(80),
        "unseen_val_chars_vs_train": unseen_val,
        "unseen_test_chars_vs_train": unseen_test,
        "num_bad_target_len_gt_timesteps": len(bad_target_gt_t),
        "num_bad_ctc_min_gt_timesteps": len(bad_ctcmin_gt_t),
        "num_suspicious_short_timesteps": len(suspicious_short_t),
        "num_suspicious_long_text": len(suspicious_long_text),
        "bad_target_len_gt_timesteps_examples": bad_target_gt_t[:max_bad_examples],
        "bad_ctc_min_gt_timesteps_examples": bad_ctcmin_gt_t[:max_bad_examples],
        "suspicious_short_timesteps_examples": suspicious_short_t[:max_bad_examples],
        "suspicious_long_text_examples": suspicious_long_text[:max_bad_examples],
        "is_ctc_feasible_with_current_downsample": len(bad_ctcmin_gt_t) == 0,
    }


def audit_dataset(
    root: Path,
    dataset: str,
    *,
    use_smoke: bool,
    time_downsample: int,
    max_bad_examples: int,
) -> dict[str, Any]:
    manifest_root = root / dataset

    if use_smoke:
        manifest_root = manifest_root / "smoke"

    path = manifest_root / "all.jsonl"

    if not path.exists():
        raise FileNotFoundError(path)

    rows = read_jsonl(path)

    return audit_rows(
        rows,
        dataset=dataset,
        time_downsample=time_downsample,
        max_bad_examples=max_bad_examples,
    )


def print_short_report(report: dict[str, Any]) -> None:
    print("\n===", report["dataset"], "===")
    print("records:", report["num_records"])
    print("splits:", report["splits"])
    print("levels:", report["levels"])
    print("categories:", report["categories"])
    print("missing images:", report["missing_images"])
    print("unreadable images:", report["unreadable_images"])
    print("empty text:", report["empty_text"])
    print("width:", report["image_width"])
    print("height:", report["image_height"])
    print("text_len:", report["text_len"])
    print("estimated_timesteps:", report["estimated_timesteps"])
    print("ctc_min_to_timesteps_ratio:", report["ctc_min_to_timesteps_ratio"])
    print("bad target_len > T:", report["num_bad_target_len_gt_timesteps"])
    print("bad ctc_min > T:", report["num_bad_ctc_min_gt_timesteps"])
    print("unseen val chars:", report["unseen_val_chars_vs_train"])
    print("unseen test chars:", report["unseen_test_chars_vs_train"])
    print("ctc feasible:", report["is_ctc_feasible_with_current_downsample"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default="data/experiments/htr_baseline_v1/manifest_audit.json")
    parser.add_argument("--time_downsample", type=int, default=4)
    parser.add_argument("--use_smoke", action="store_true")
    parser.add_argument("--max_bad_examples", type=int, default=20)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()

    root = Path(args.manifest_root)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    reports = {}

    for dataset in datasets:
        report = audit_dataset(
            root,
            dataset,
            use_smoke=args.use_smoke,
            time_downsample=args.time_downsample,
            max_bad_examples=args.max_bad_examples,
        )
        reports[dataset] = report
        print_short_report(report)

    global_report = {
        "manifest_root": str(root),
        "use_smoke": args.use_smoke,
        "time_downsample": args.time_downsample,
        "datasets": reports,
        "global_decision": {
            "ctc_feasible_all": all(r["is_ctc_feasible_with_current_downsample"] for r in reports.values()),
            "datasets_with_unseen_val_chars": [
                d for d, r in reports.items() if r["unseen_val_chars_vs_train"]
            ],
            "datasets_with_unseen_test_chars": [
                d for d, r in reports.items() if r["unseen_test_chars_vs_train"]
            ],
            "datasets_with_bad_ctc_samples": [
                d for d, r in reports.items() if r["num_bad_ctc_min_gt_timesteps"] > 0
            ],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(global_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nwrote:", out)
    print("global decision:")
    print(json.dumps(global_report["global_decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()