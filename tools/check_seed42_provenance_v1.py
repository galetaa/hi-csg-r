from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRED_KEYS = [
    "prediction",
    "pred",
    "hyp",
    "hypothesis",
    "decoded",
    "text_pred",
]

REF_KEYS = [
    "reference",
    "ref",
    "target",
    "text",
    "gt",
    "label",
    "transcription",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_metric(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]

    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]

    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            insert = cur[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            cur.append(min(insert, delete, replace))
        prev = cur

    return prev[-1]


def get_first(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def recompute_from_predictions(predictions_path: Path) -> dict[str, float]:
    rows = read_jsonl(predictions_path)

    row_cers = []
    row_wers = []
    row_exacts = []

    missing = []

    for i, row in enumerate(rows):
        pred = get_first(row, PRED_KEYS)
        ref = get_first(row, REF_KEYS)

        if pred is None or ref is None:
            missing.append(
                {
                    "idx": i,
                    "keys": sorted(row.keys()),
                    "pred_found": pred is not None,
                    "ref_found": ref is not None,
                }
            )
            continue

        if "cer" in row:
            row_cers.append(float(row["cer"]))
        else:
            row_cers.append(levenshtein(pred, ref) / max(len(ref), 1))

        if "wer" in row:
            row_wers.append(float(row["wer"]))
        else:
            pred_words = pred.split()
            ref_words = ref.split()
            row_wers.append(levenshtein(pred_words, ref_words) / max(len(ref_words), 1))

        if "exact" in row:
            row_exacts.append(float(row["exact"]))
        else:
            row_exacts.append(1.0 if pred == ref else 0.0)

    valid_n = len(rows) - len(missing)

    if missing:
        raise ValueError(
            "Cannot recompute metrics: missing prediction/reference keys. "
            f"First missing examples: {missing[:3]}"
        )

    return {
        "n": float(len(rows)),
        "valid_n": float(valid_n),
        "cer": sum(row_cers) / max(len(row_cers), 1),
        "wer": sum(row_wers) / max(len(row_wers), 1),
        "exact": sum(row_exacts) / max(len(row_exacts), 1),
    }


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/htr_graph_v1/eval_tri10k_image_only_v1_test_final/summary.json",
    )
    parser.add_argument(
        "--manifest",
        default="data/experiments/htr_graph_v1/graph_ready/tri10k_mixed/test.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/htr_graph_v1/tri10k_image_only_v1/best.pt",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/htr_graph_v1/eval_tri10k_image_only_v1_test_final/predictions.jsonl",
    )
    parser.add_argument("--expected_n", type=int, default=5563)
    parser.add_argument("--expected_penalty", type=float, default=-0.4)
    parser.add_argument("--metric_tol", type=float, default=1e-8)
    parser.add_argument(
        "--out_dir",
        default="outputs/final_result_package_v1",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    manifest_path = Path(args.manifest)
    checkpoint_path = Path(args.checkpoint)
    predictions_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, status: str, details: dict[str, Any] | None = None) -> None:
        checks.append(
            {
                "name": name,
                "status": status,
                "details": details or {},
            }
        )

    if summary_path.exists():
        summary = read_json(summary_path)
        add_check("summary_exists", "PASS", {"path": str(summary_path)})
    else:
        summary = {}
        add_check("summary_exists", "FAIL", {"path": str(summary_path)})

    if manifest_path.exists():
        manifest_n = count_jsonl(manifest_path)
        add_check(
            "manifest_exists",
            "PASS",
            {"path": str(manifest_path), "manifest_n": manifest_n},
        )
    else:
        manifest_n = None
        add_check("manifest_exists", "FAIL", {"path": str(manifest_path)})

    if checkpoint_path.exists():
        add_check(
            "checkpoint_exists",
            "PASS",
            {
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            },
        )
    else:
        add_check("checkpoint_exists", "FAIL", {"path": str(checkpoint_path)})

    summary_n_value = get_metric(summary, "n")
    summary_n = int(summary_n_value) if summary_n_value is not None else None
    if summary_n == args.expected_n:
        add_check("summary_n_expected", "PASS", {"summary_n": summary_n})
    else:
        add_check(
            "summary_n_expected",
            "FAIL",
            {"summary_n": summary_n, "expected_n": args.expected_n},
        )

    if manifest_n == args.expected_n:
        add_check("manifest_n_expected", "PASS", {"manifest_n": manifest_n})
    else:
        add_check(
            "manifest_n_expected",
            "FAIL",
            {"manifest_n": manifest_n, "expected_n": args.expected_n},
        )

    if manifest_n is not None and summary_n is not None and manifest_n == summary_n:
        add_check("manifest_n_matches_summary_n", "PASS", {"n": summary_n})
    else:
        add_check(
            "manifest_n_matches_summary_n",
            "FAIL",
            {"manifest_n": manifest_n, "summary_n": summary_n},
        )

    penalty = summary.get("blank_logit_penalty")
    if penalty is not None and close(float(penalty), args.expected_penalty, 1e-12):
        add_check(
            "blank_logit_penalty_expected",
            "PASS",
            {"blank_logit_penalty": penalty},
        )
    else:
        add_check(
            "blank_logit_penalty_expected",
            "FAIL",
            {"blank_logit_penalty": penalty, "expected": args.expected_penalty},
        )

    recomputed = None
    if predictions_path.exists():
        try:
            recomputed = recompute_from_predictions(predictions_path)
            add_check(
                "predictions_recomputed",
                "PASS",
                {"path": str(predictions_path), **recomputed},
            )

            for metric in ["cer", "wer", "exact"]:
                summary_metric = get_metric(summary, metric)
                if summary_metric is not None:
                    ok = close(
                        float(summary_metric),
                        float(recomputed[metric]),
                        args.metric_tol,
                    )
                    add_check(
                        f"{metric}_matches_recomputed",
                        "PASS" if ok else "FAIL",
                        {
                            "summary": summary_metric,
                            "recomputed": recomputed[metric],
                            "tol": args.metric_tol,
                        },
                    )
                else:
                    add_check(
                        f"{metric}_matches_recomputed",
                        "FAIL",
                        {"reason": f"{metric} missing from summary"},
                    )

        except Exception as exc:
            add_check(
                "predictions_recomputed",
                "FAIL",
                {
                    "path": str(predictions_path),
                    "error": repr(exc),
                },
            )
    else:
        add_check(
            "predictions_recomputed",
            "WEAK",
            {
                "reason": "predictions.jsonl not found; metrics could not be recomputed",
                "path": str(predictions_path),
            },
        )

    fail_n = sum(1 for check in checks if check["status"] == "FAIL")
    weak_n = sum(1 for check in checks if check["status"] == "WEAK")

    if fail_n == 0 and weak_n == 0:
        verdict = "PASS"
        interpretation = (
            "Seed42 combined summary is reproducible enough for primary 3-seed reporting."
        )
    elif fail_n == 0 and weak_n > 0:
        verdict = "WEAK_PASS"
        interpretation = (
            "Seed42 combined summary passes structural checks, but metrics were not "
            "independently recomputed from predictions. Prefer rerunning evaluation with predictions."
        )
    else:
        verdict = "FAIL"
        interpretation = (
            "Seed42 combined summary is not acceptable for primary reporting until failed checks are fixed."
        )

    report = {
        "verdict": verdict,
        "interpretation": interpretation,
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path),
        "summary": summary,
        "recomputed_from_predictions": recomputed,
        "checks": checks,
    }

    json_path = out_dir / "seed42_provenance_check.json"
    md_path = out_dir / "seed42_provenance_check.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Seed42 provenance check\n")
    lines.append(f"Verdict: **{verdict}**\n")
    lines.append(f"{interpretation}\n")
    lines.append("## Checks\n")
    lines.append("| check | status | details |")
    lines.append("|---|---|---|")
    for check in checks:
        lines.append(
            f"| {check['name']} | {check['status']} | "
            f"`{json.dumps(check['details'], ensure_ascii=False)}` |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "fail_n": fail_n, "weak_n": weak_n}, ensure_ascii=False, indent=2))
    print("wrote:", json_path)
    print("wrote:", md_path)


if __name__ == "__main__":
    main()
