from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def text_len_bucket(text: str) -> str:
    n = len(text)
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11+"


def token_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"

    chars = [ch for ch in stripped if not ch.isspace()]
    if not chars:
        return "empty"

    alpha = [ch for ch in chars if ch.isalpha()]
    numeric = [ch for ch in chars if ch.isdigit()]
    punct = [
        ch for ch in chars
        if unicodedata.category(ch).startswith("P")
        or unicodedata.category(ch).startswith("S")
    ]
    other = [
        ch for ch in chars
        if ch not in alpha
        and ch not in numeric
        and ch not in punct
    ]

    if len(alpha) == len(chars):
        return "alpha"
    if len(numeric) == len(chars):
        return "numeric"
    if len(punct) == len(chars):
        return "punctuation"
    if other:
        return "mixed"
    return "mixed"


def metrics(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    accepted = [
        row for row in rows
        if float(row["risk"]) <= threshold
    ]
    rejected = len(rows) - len(accepted)
    return {
        "n_total": len(rows),
        "n_accepted": len(accepted),
        "n_rejected": rejected,
        "coverage": len(accepted) / max(len(rows), 1),
        "cer_accepted": (
            sum(float(row["cer"]) for row in accepted) / len(accepted)
            if accepted else None
        ),
        "exact_accepted": (
            sum(float(row["exact"]) for row in accepted) / len(accepted)
            if accepted else None
        ),
        "mean_risk_all": (
            sum(float(row["risk"]) for row in rows) / len(rows)
            if rows else None
        ),
        "mean_risk_accepted": (
            sum(float(row["risk"]) for row in accepted) / len(accepted)
            if accepted else None
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    operating = json.loads((out_root / "operating_points.json").read_text(encoding="utf-8"))
    prediction_rows = read_jsonl(
        out_root / "confidence_predictions" / "plus_10k_context.test.jsonl"
    )

    # Reconstruct risk from audit files plus accepted/rejected threshold membership is not enough.
    # The operating-points builder writes full risk only to audit subsets, so read the cache
    # through operating_points.json is insufficient. Recompute rows by joining the two audit
    # files for strict and using calibration output is intentionally avoided here. Instead,
    # consume accepted/rejected audit files when present for strict, and fail loudly otherwise.
    #
    # To keep all operating points available, use the richer JSONL rows emitted by
    # build_confidence_operating_points_iter2 if it exists.
    enriched_path = out_root / "operating_point_test_rows.jsonl"
    if not enriched_path.exists():
        raise FileNotFoundError(
            f"{enriched_path} missing; rerun build_confidence_operating_points_iter2 "
            "after the update that writes full test rows."
        )

    rows = read_jsonl(enriched_path)
    for row in rows:
        text = str(row.get("target") or row.get("text") or "")
        row["text_len_bucket"] = text_len_bucket(text)
        row["token_type"] = token_type(text)
        row["short_1_3"] = "short_1_3" if len(text) <= 3 else "not_short_1_3"

    strata_rows = []
    for point, point_data in operating["operating_points"].items():
        threshold = float(point_data["threshold"])

        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            strata[("dataset", str(row.get("dataset", "unknown")))].append(row)
            strata[("text_len", str(row["text_len_bucket"]))].append(row)
            strata[("token_type", str(row["token_type"]))].append(row)
            strata[("short_flag", str(row["short_1_3"]))].append(row)

        for (stratum_type, stratum), stratum_rows in sorted(strata.items()):
            m = metrics(stratum_rows, threshold=threshold)
            strata_rows.append({
                "operating_point": point,
                "threshold": threshold,
                "stratum_type": stratum_type,
                "stratum": stratum,
                **m,
            })

    csv_path = out_root / "operating_point_strata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(strata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(strata_rows)

    lines = [
        "# Operating Point Stratification",
        "",
        "Rows are test samples for `plus_10k_context` with `confidence_graph` risk.",
        "",
    ]
    for point in ["strict", "balanced", "broad"]:
        lines.extend([
            f"## {point}",
            "",
            "| stratum type | stratum | n | accepted | coverage | accepted CER | accepted exact | mean risk |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in strata_rows:
            if row["operating_point"] != point:
                continue
            lines.append(
                f"| `{row['stratum_type']}` | `{row['stratum']}` | {row['n_total']} | "
                f"{row['n_accepted']} | {fmt(row['coverage'])} | "
                f"{fmt(row['cer_accepted'])} | {fmt(row['exact_accepted'])} | "
                f"{fmt(row['mean_risk_all'])} |"
            )
        lines.append("")

    (out_root / "operating_point_strata.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(json.dumps({
        "csv": str(csv_path),
        "md": str(out_root / "operating_point_strata.md"),
        "n_rows": len(rows),
        "n_strata_rows": len(strata_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
