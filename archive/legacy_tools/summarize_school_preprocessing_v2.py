from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean(x: Any) -> str:
    s = str(x or "").strip()
    return s if s else "missing"


def rate(rows: list[dict[str, Any]], key: str, positive: str) -> float:
    vals = [clean(r.get(key)) for r in rows if clean(r.get(key)) != "missing"]
    if not vals:
        return 0.0
    return sum(v == positive for v in vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.annotations_csv))

    summary = {
        "annotations_csv": args.annotations_csv,
        "n": len(rows),
        "best_variant_counts": dict(Counter(clean(r.get("best_variant")) for r in rows)),
        "fix_grade_counts": dict(Counter(clean(r.get("fix_grade")) for r in rows)),
        "real_ink_erased_counts": dict(Counter(clean(r.get("real_ink_erased")) for r in rows)),
        "border_artifact_after_counts": dict(Counter(clean(r.get("border_artifact_after")) for r in rows)),
        "skeleton_follows_ink_after_counts": dict(Counter(clean(r.get("skeleton_follows_ink_after")) for r in rows)),
        "good_fix_rate": rate(rows, "fix_grade", "good_fix"),
        "partial_fix_rate": rate(rows, "fix_grade", "partial_fix"),
        "bad_fix_rate": rate(rows, "fix_grade", "bad_fix"),
        "real_ink_erased_rate": rate(rows, "real_ink_erased", "1"),
        "border_artifact_after_rate": rate(rows, "border_artifact_after", "1"),
        "skeleton_follows_ink_after_rate": rate(rows, "skeleton_follows_ink_after", "1"),
    }

    lines = []
    lines.append("# School preprocessing v2 summary")
    lines.append("")
    lines.append("## 1. Aggregate")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| n | {summary['n']} |")
    lines.append(f"| good fix rate | {summary['good_fix_rate']:.3f} |")
    lines.append(f"| partial fix rate | {summary['partial_fix_rate']:.3f} |")
    lines.append(f"| bad fix rate | {summary['bad_fix_rate']:.3f} |")
    lines.append(f"| real ink erased rate | {summary['real_ink_erased_rate']:.3f} |")
    lines.append(f"| border artifact after rate | {summary['border_artifact_after_rate']:.3f} |")
    lines.append(f"| skeleton follows ink after rate | {summary['skeleton_follows_ink_after_rate']:.3f} |")
    lines.append("")
    lines.append("## 2. Best variant counts")
    lines.append("")
    lines.append("| variant | n |")
    lines.append("|---|---:|")
    for k, v in sorted(summary["best_variant_counts"].items()):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## 3. Fix grade counts")
    lines.append("")
    lines.append("| grade | n |")
    lines.append("|---|---:|")
    for k, v in sorted(summary["fix_grade_counts"].items()):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## 4. Verdict")
    lines.append("")

    if summary["good_fix_rate"] >= 0.6 and summary["real_ink_erased_rate"] <= 0.15:
        verdict = "candidate_preprocessing_fix_found"
        lines.append(
            "A candidate preprocessing fix appears viable. The next step is to implement the winning variant "
            "as a deterministic preprocessing function and rerun H2 audit metrics on school-notebooks."
        )
    elif summary["good_fix_rate"] + summary["partial_fix_rate"] >= 0.6:
        verdict = "partial_fix_only"
        lines.append(
            "The tested variants provide only partial improvement. Do not integrate automatically yet. "
            "Use the annotations to design a more targeted preprocessing rule."
        )
    else:
        verdict = "no_reliable_fix_found"
        lines.append(
            "No reliable preprocessing fix was found among tested variants. School-notebooks should remain "
            "reported as a preprocessing limitation."
        )

    summary["verdict"] = verdict

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_json)
    print("wrote:", out_md)


if __name__ == "__main__":
    main()