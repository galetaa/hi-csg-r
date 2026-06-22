from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


YES_VALUES = {"1", "yes", "true", "y"}
NO_VALUES = {"0", "no", "false", "n"}


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_yes(value: Any) -> bool:
    return norm(value) in YES_VALUES


def is_no(value: Any) -> bool:
    return norm(value) in NO_VALUES


def rate(rows: list[dict[str, str]], field: str, *, yes: bool = True) -> float:
    if not rows:
        return 0.0

    if yes:
        return sum(is_yes(row.get(field)) for row in rows) / len(rows)

    return sum(is_no(row.get(field)) for row in rows) / len(rows)


def completion_rate(rows: list[dict[str, str]]) -> float:
    required = [
        "usable",
        "ink_loss",
        "line_residual",
        "neighbor_text_removed",
        "skeleton_follows_ink",
    ]

    if not rows:
        return 0.0

    complete = 0

    for row in rows:
        if all(norm(row.get(field)) for field in required):
            complete += 1

    return complete / len(rows)


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "completion_rate": completion_rate(rows),
        "usable_rate": rate(rows, "usable"),
        "ink_loss_rate": rate(rows, "ink_loss"),
        "line_residual_rate": rate(rows, "line_residual"),
        "neighbor_text_removed_rate": rate(rows, "neighbor_text_removed"),
        "skeleton_follows_ink_rate": rate(rows, "skeleton_follows_ink"),
        "counts": {
            field: dict(Counter(norm(row.get(field)) for row in rows))
            for field in [
                "usable",
                "ink_loss",
                "line_residual",
                "neighbor_text_removed",
                "skeleton_follows_ink",
            ]
        },
    }


def verdict(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "usable_ge_90": summary["usable_rate"] >= 0.90,
        "skeleton_follows_ink_ge_90": summary["skeleton_follows_ink_rate"] >= 0.90,
        "ink_loss_le_07": summary["ink_loss_rate"] <= 0.07,
        "line_residual_le_10": summary["line_residual_rate"] <= 0.10,
        "neighbor_text_removed_ge_90": summary["neighbor_text_removed_rate"] >= 0.90,
    }

    return {
        "accepted": all(checks.values()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.annotations_csv))
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_group[row.get("validation_group", "unknown")].append(row)

    overall = summarize_rows(rows)

    result = {
        "annotations_csv": args.annotations_csv,
        "overall": {
            **overall,
            "verdict": verdict(overall),
        },
        "by_group": {
            group: {
                **summarize_rows(group_rows),
                "verdict": verdict(summarize_rows(group_rows)),
            }
            for group, group_rows in sorted(by_group.items())
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("# School lineaware v3 quality gate")
    lines.append("")
    lines.append(f"- annotations: `{args.annotations_csv}`")
    lines.append(f"- accepted: `{result['overall']['verdict']['accepted']}`")
    lines.append("")
    lines.append("| group | n | complete | usable | ink loss | line residual | neighbor removed | skeleton follows | accepted |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    def row_line(group: str, summary: dict[str, Any]) -> str:
        return (
            f"| `{group}` | "
            f"{summary['n']} | "
            f"{summary['completion_rate']:.3f} | "
            f"{summary['usable_rate']:.3f} | "
            f"{summary['ink_loss_rate']:.3f} | "
            f"{summary['line_residual_rate']:.3f} | "
            f"{summary['neighbor_text_removed_rate']:.3f} | "
            f"{summary['skeleton_follows_ink_rate']:.3f} | "
            f"{summary['verdict']['accepted']} |"
        )

    lines.append(row_line("overall", result["overall"]))

    for group, summary in result["by_group"].items():
        lines.append(row_line(group, summary))

    out_md.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    print("wrote:", out_json)
    print("wrote:", out_md)


if __name__ == "__main__":
    main()
