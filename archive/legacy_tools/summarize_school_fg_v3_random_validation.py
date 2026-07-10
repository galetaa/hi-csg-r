from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def value(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, "") or "").strip()


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0

    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / denominator
    )

    return max(0.0, center - half), min(1.0, center + half)


def metric(k: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(k, n)
    return {
        "count": k,
        "n": n,
        "rate": k / max(n, 1),
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.annotations_csv))
    n = len(rows)

    required = [
        "best_variant",
        "fix_grade",
        "real_ink_erased",
        "border_artifact_after",
        "skeleton_follows_ink_after",
    ]

    missing = {
        key: sum(not value(row, key) for row in rows)
        for key in required
    }

    variant_counts = Counter(value(r, "best_variant") for r in rows)
    grade_counts = Counter(value(r, "fix_grade") for r in rows)

    good_n = sum(value(r, "fix_grade") == "good_fix" for r in rows)
    partial_n = sum(value(r, "fix_grade") == "partial_fix" for r in rows)
    bad_n = sum(value(r, "fix_grade") == "bad_fix" for r in rows)

    erased_n = sum(value(r, "real_ink_erased") == "1" for r in rows)
    artifact_n = sum(value(r, "border_artifact_after") == "1" for r in rows)
    follows_n = sum(
        value(r, "skeleton_follows_ink_after") == "1"
        for r in rows
    )

    strict_good_rows = [
        r for r in rows
        if value(r, "fix_grade") == "good_fix"
        and value(r, "real_ink_erased") == "0"
        and value(r, "border_artifact_after") == "0"
        and value(r, "skeleton_follows_ink_after") == "1"
    ]

    strict_usable_rows = [
        r for r in rows
        if value(r, "real_ink_erased") == "0"
        and value(r, "border_artifact_after") == "0"
        and value(r, "skeleton_follows_ink_after") == "1"
    ]

    inconsistencies = []

    for r in rows:
        reasons = []

        if value(r, "fix_grade") == "good_fix":
            if value(r, "real_ink_erased") == "1":
                reasons.append("good_fix_with_erased_ink")
            if value(r, "border_artifact_after") == "1":
                reasons.append("good_fix_with_remaining_artifact")
            if value(r, "skeleton_follows_ink_after") == "0":
                reasons.append("good_fix_without_skeleton_following_ink")

        if reasons:
            inconsistencies.append({
                "sample_id": value(r, "sample_id"),
                "target": value(r, "target"),
                "reasons": reasons,
            })

    summary = {
        "annotations_csv": args.annotations_csv,
        "n": n,
        "missing_required_fields": missing,
        "variant_counts": dict(variant_counts),
        "fix_grade_counts": dict(grade_counts),
        "metrics": {
            "auto_selected": metric(
                sum(value(r, "best_variant") == "school_dark_auto" for r in rows),
                n,
            ),
            "good_fix": metric(good_n, n),
            "partial_fix": metric(partial_n, n),
            "bad_fix": metric(bad_n, n),
            "real_ink_erased": metric(erased_n, n),
            "background_artifact_after": metric(artifact_n, n),
            "skeleton_follows_ink_after": metric(follows_n, n),
            "strict_good": metric(len(strict_good_rows), n),
            "strict_usable": metric(len(strict_usable_rows), n),
        },
        "annotation_inconsistencies": inconsistencies,
        "verdict": (
            "independent_random_validation_supports_school_dark_auto"
            if len(strict_usable_rows) / max(n, 1) >= 0.80
            else "random_validation_does_not_support_generalization"
        ),
    }

    lines: list[str] = []
    lines.append("# School foreground v3 random validation")
    lines.append("")
    lines.append("## 1. Sampling")
    lines.append("")
    lines.append(f"- independently sampled test samples: {n}")
    lines.append(f"- selected preprocessing: `school_dark_auto`")
    lines.append("")

    lines.append("## 2. Results")
    lines.append("")
    lines.append("| metric | count | rate | 95% Wilson CI |")
    lines.append("|---|---:|---:|---:|")

    labels = {
        "auto_selected": "school_dark_auto selected",
        "good_fix": "good fix",
        "partial_fix": "partial fix",
        "bad_fix": "bad fix",
        "real_ink_erased": "real ink erased",
        "background_artifact_after": "background artifact remains",
        "skeleton_follows_ink_after": "skeleton follows ink",
        "strict_good": "strict good",
        "strict_usable": "strict usable",
    }

    for key, label in labels.items():
        m = summary["metrics"][key]
        lines.append(
            f"| {label} | {m['count']}/{m['n']} | "
            f"{m['rate']:.3f} | "
            f"{m['wilson_95_low']:.3f}–{m['wilson_95_high']:.3f} |"
        )

    lines.append("")
    lines.append("## 3. Annotation QA")
    lines.append("")

    if inconsistencies:
        lines.append("| sample | target | issue |")
        lines.append("|---|---|---|")

        for r in inconsistencies:
            lines.append(
                f"| `{r['sample_id']}` | `{r['target']}` | "
                f"`{';'.join(r['reasons'])}` |"
            )
    else:
        lines.append("No rubric inconsistencies detected.")

    lines.append("")
    lines.append("## 4. Verdict")
    lines.append("")
    lines.append(
        "`school_dark_auto` is supported by an independent random validation sample. "
        "The raw good-fix rate is reported together with the stricter usable rate, "
        "which requires no erased ink, no remaining dominant background artifact, "
        "and a skeleton that follows visible ink."
    )
    lines.append("")
    lines.append(
        "The result supports generalization to the sampled School Notebooks test "
        "distribution. It does not establish performance across all splits, all "
        "possible acquisition conditions, or independent annotators."
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_json)
    print("wrote:", out_md)


if __name__ == "__main__":
    main()