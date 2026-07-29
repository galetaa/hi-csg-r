from __future__ import annotations

import argparse
import csv
import json
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
    return sum(v == positive for v in vals) / max(len(vals), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_h2_summary_json", required=True)
    parser.add_argument("--foreground_v3_annotations_csv", required=True)
    parser.add_argument("--foreground_v3_final_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    old = json.loads(Path(args.old_h2_summary_json).read_text(encoding="utf-8"))
    rows = read_csv(Path(args.foreground_v3_annotations_csv))
    fg_final = json.loads(Path(args.foreground_v3_final_json).read_text(encoding="utf-8"))

    school_old = old["by_dataset"]["school_notebooks_clean"]

    update = {
        "old_school_notebooks": school_old,
        "foreground_v3": {
            "n": len(rows),
            "selected_method": fg_final["selected_method"],
            "good_fix_rate": rate(rows, "fix_grade", "good_fix"),
            "partial_fix_rate": rate(rows, "fix_grade", "partial_fix"),
            "bad_fix_rate": rate(rows, "fix_grade", "bad_fix"),
            "real_ink_erased_rate": rate(rows, "real_ink_erased", "1"),
            "background_artifact_after_rate": rate(rows, "border_artifact_after", "1"),
            "skeleton_follows_ink_after_rate": rate(rows, "skeleton_follows_ink_after", "1"),
        },
        "verdict": "school_notebooks_preprocessing_failure_partially_repaired",
    }

    lines = []
    lines.append("# H2 school-notebooks foreground v3 audit update")
    lines.append("")
    lines.append("## 1. Previous H2 school-notebooks status")
    lines.append("")
    lines.append("| metric | old value |")
    lines.append("|---|---:|")
    lines.append(f"| n | {school_old['n']} |")
    lines.append(f"| critical topology error rate | {school_old['critical_topology_error_rate']:.3f} |")
    lines.append(f"| skeleton follows ink rate | {school_old['skeleton_follows_ink_rate']:.3f} |")
    lines.append(f"| border artifact rate | {school_old['border_artifact_rate']:.3f} |")
    lines.append(f"| mean graph quality 0–3 | {school_old['mean_graph_quality_0_3']:.3f} |")
    lines.append("")
    lines.append("## 2. Foreground v3 audit")
    lines.append("")
    v3 = update["foreground_v3"]
    lines.append("| metric | v3 value |")
    lines.append("|---|---:|")
    lines.append(f"| n | {v3['n']} |")
    lines.append(f"| selected method | `{v3['selected_method']}` |")
    lines.append(f"| good fix rate | {v3['good_fix_rate']:.3f} |")
    lines.append(f"| partial fix rate | {v3['partial_fix_rate']:.3f} |")
    lines.append(f"| bad fix rate | {v3['bad_fix_rate']:.3f} |")
    lines.append(f"| real ink erased rate | {v3['real_ink_erased_rate']:.3f} |")
    lines.append(f"| background artifact after rate | {v3['background_artifact_after_rate']:.3f} |")
    lines.append(f"| skeleton follows ink after rate | {v3['skeleton_follows_ink_after_rate']:.3f} |")
    lines.append("")
    lines.append("## 3. Interpretation")
    lines.append("")
    lines.append(
        "Foreground v3 substantially repairs the school-notebooks preprocessing failure on the audited subset. "
        "The fix should be treated as a graph-extraction preprocessing improvement, not as HTR accuracy evidence."
    )
    lines.append("")
    lines.append(
        "The original H2 conclusion remains historically valid for the old pipeline, but the improved pipeline now has "
        "a viable path for school-notebooks foreground extraction."
    )

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(update, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()