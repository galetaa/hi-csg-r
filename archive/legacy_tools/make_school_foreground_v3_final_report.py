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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "n/a"


def fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "n/a"


def clean(x: Any) -> str:
    s = str(x or "").strip()
    return s if s else "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--global145_comparison_json", required=True)
    parser.add_argument("--auto_comparison_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    annotations = read_csv(Path(args.annotations_csv))
    selector_summary = load_json(Path(args.summary_json))
    global145 = load_json(Path(args.global145_comparison_json))
    auto = load_json(Path(args.auto_comparison_json))

    best_variant_counts = Counter(clean(r.get("best_variant")) for r in annotations)

    final = {
        "verdict": "school_foreground_v3_auto_selected",
        "reason": [
            "Manual browser audit found a viable foreground extraction fix.",
            "global_dark_145 and global_dark_120 were both selected manually.",
            "school_dark_auto approximates this by using global_dark_145 unless foreground is too large, then falling back to global_dark_120.",
            "auto has lower high-foreground warning rate than global145 on train/val/test.",
        ],
        "manual_selector_summary": selector_summary,
        "manual_best_variant_counts": dict(best_variant_counts),
        "selected_method": "school_dark_auto",
        "global145_comparison": global145,
        "auto_comparison": auto,
    }

    lines = []
    lines.append("# School foreground v3 final report")
    lines.append("")
    lines.append("## 1. Verdict")
    lines.append("")
    lines.append("```text")
    lines.append("Selected method: school_dark_auto")
    lines.append("Status: candidate preprocessing fix found")
    lines.append("```")
    lines.append("")
    lines.append(
        "Manual browser audit showed that replacing Sauvola with simple dark-threshold foreground extraction "
        "substantially improves school-notebooks foreground masks on the audited samples."
    )
    lines.append("")
    lines.append("## 2. Manual audit summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| n | {selector_summary['n']} |")
    lines.append(f"| good fix rate | {pct(selector_summary['good_fix_rate'])} |")
    lines.append(f"| partial fix rate | {pct(selector_summary['partial_fix_rate'])} |")
    lines.append(f"| bad fix rate | {pct(selector_summary['bad_fix_rate'])} |")
    lines.append(f"| real ink erased rate | {pct(selector_summary['real_ink_erased_rate'])} |")
    lines.append(f"| background/blob artifact after rate | {pct(selector_summary['border_artifact_after_rate'])} |")
    lines.append(f"| skeleton follows ink after rate | {pct(selector_summary['skeleton_follows_ink_after_rate'])} |")
    lines.append("")
    lines.append("## 3. Manual best variant counts")
    lines.append("")
    lines.append("| variant | n |")
    lines.append("|---|---:|")
    for k, v in sorted(best_variant_counts.items()):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## 4. Feature-level comparison")
    lines.append("")
    lines.append("### `global_dark_145`")
    lines.append("")
    lines.append("| split | old fg | new fg | old skel | new skel | old high-fg | new high-fg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split, s in global145["splits"].items():
        old = s["old"]
        new = s["new"]
        lines.append(
            f"| `{split}` | {fmt(old['fg_fraction_mean'])} | {fmt(new['fg_fraction_mean'])} | "
            f"{fmt(old['skel_fraction_mean'])} | {fmt(new['skel_fraction_mean'])} | "
            f"{pct(old['very_high_foreground_rate'])} | {pct(new['very_high_foreground_rate'])} |"
        )
    lines.append("")
    lines.append("### `school_dark_auto`")
    lines.append("")
    lines.append("| split | old fg | new fg | old skel | new skel | old high-fg | new high-fg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split, s in auto["splits"].items():
        old = s["old"]
        new = s["new"]
        lines.append(
            f"| `{split}` | {fmt(old['fg_fraction_mean'])} | {fmt(new['fg_fraction_mean'])} | "
            f"{fmt(old['skel_fraction_mean'])} | {fmt(new['skel_fraction_mean'])} | "
            f"{pct(old['very_high_foreground_rate'])} | {pct(new['very_high_foreground_rate'])} |"
        )
    lines.append("")
    lines.append("## 5. Decision")
    lines.append("")
    lines.append(
        "`school_dark_auto` is selected over fixed `global_dark_145` because it reduces excessive foreground "
        "more consistently while preserving nonzero skeleton structure. It is a deterministic preprocessing fix "
        "for school-notebooks graph extraction, not an HTR architecture change."
    )
    lines.append("")
    lines.append("## 6. Limits")
    lines.append("")
    lines.append(
        "This fix is validated on the H2 school-notebooks audit subset and feature-level split summaries. "
        "It should be used for graph extraction repair and H2 follow-up, not yet as evidence of improved HTR accuracy."
    )

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()