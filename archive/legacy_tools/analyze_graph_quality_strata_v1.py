from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = cur
    return prev[-1]


def norm_text(x: Any) -> str:
    return "" if x is None else str(x)


def row_text(row: dict[str, Any]) -> str:
    for key in ("text", "target", "label", "transcription", "normalized_transcription"):
        if key in row:
            return norm_text(row[key])
    raise KeyError(f"No text key in row. Keys: {sorted(row.keys())}")


def row_pred(row: dict[str, Any]) -> str:
    for key in ("pred", "prediction", "decoded", "hyp"):
        if key in row:
            return norm_text(row[key])
    raise KeyError(f"No prediction key in row. Keys: {sorted(row.keys())}")


def graph_score(row: dict[str, Any]) -> float:
    for key in (
        "graph_quality_score",
        "graph_confidence",
        "graph_quality",
        "quality_score",
    ):
        if key in row:
            try:
                return float(row[key])
            except Exception:
                pass

    if "graph_warning_count" in row:
        return -float(row["graph_warning_count"])

    if "warning_count" in row:
        return -float(row["warning_count"])

    feats = row.get("graph_features")
    names = row.get("graph_feature_names")

    if isinstance(feats, list) and isinstance(names, list):
        name_to_value = {str(k): float(v) for k, v in zip(names, feats)}
        warning = name_to_value.get("warning_count", 0.0)
        short_branch = name_to_value.get("short_branch_ratio", 0.0)
        return -(warning + short_branch)

    raise KeyError("No graph quality proxy found.")


def cer_for_rows(rows: list[dict[str, Any]]) -> float:
    edits = 0
    total = 0
    for row in rows:
        ref = row_text(row)
        pred = row_pred(row)
        edits += levenshtein(ref, pred)
        total += max(1, len(ref))
    return edits / max(1, total)


def assign_strata(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    scored = [(graph_score(r), r) for r in rows]
    scored.sort(key=lambda x: x[0])

    n = len(scored)
    lo = n // 3
    hi = 2 * n // 3

    return {
        "low": [r for _, r in scored[:lo]],
        "medium": [r for _, r in scored[lo:hi]],
        "high": [r for _, r in scored[hi:]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Manifest with graph quality fields.")
    parser.add_argument("--predictions", required=True, help="Model predictions jsonl.")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()

    manifest_rows = read_jsonl(Path(args.manifest))
    pred_rows = read_jsonl(Path(args.predictions))

    pred_by_id = {str(r.get("sample_id")): r for r in pred_rows}

    joined: list[dict[str, Any]] = []
    missing = 0

    for m in manifest_rows:
        sid = str(m.get("sample_id"))
        p = pred_by_id.get(sid)
        if p is None:
            missing += 1
            continue

        row = dict(m)
        row["pred"] = row_pred(p)

        try:
            row["text"] = row_text(p)
        except Exception:
            row["text"] = row_text(m)

        joined.append(row)

    strata = assign_strata(joined)

    out = {
        "model": args.model_name,
        "condition": args.condition,
        "n_manifest": len(manifest_rows),
        "n_predictions": len(pred_rows),
        "n_joined": len(joined),
        "missing_predictions": missing,
        "strata": {},
    }

    for name, group in strata.items():
        scores = [graph_score(r) for r in group]
        out["strata"][name] = {
            "n": len(group),
            "cer": cer_for_rows(group),
            "graph_score_min": min(scores) if scores else None,
            "graph_score_mean": sum(scores) / len(scores) if scores else None,
            "graph_score_max": max(scores) if scores else None,
        }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
