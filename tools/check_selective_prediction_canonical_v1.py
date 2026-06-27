from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


CANONICAL_MARKERS = [
    "plus_school_lines_10k_context",
    "10k_context",
    "plus_10k",
]

BAD_MODEL_MARKERS = [
    "graph_fusion",
    "gated",
    "zero_graph",
    "graph_vector",
]

REQUIRED_VARIANT_MARKERS = {
    "confidence": ["confidence"],
    "graph_or_quality": ["graph", "quality", "foreground", "skeleton"],
    "confidence_graph": [
        "confidence_graph",
        "conf_graph",
        "confidence+graph",
        "confidence_quality",
    ],
}

LEAKAGE_RISK_MARKERS = [
    "text_len",
    "target_len",
    "ref_len",
    "label_len",
    "gt_len",
    "transcription",
    "reference",
    "target_text",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path, max_rows: int = 5) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        out = []
        for i, row in enumerate(csv.DictReader(f)):
            if i >= max_rows:
                break
            out.append(dict(row))
        return out


def text_from_file(path: Path, max_chars: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def flatten_json_keys(obj: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            keys.append(full)
            keys.extend(flatten_json_keys(value, full))
    elif isinstance(obj, list):
        for i, value in enumerate(obj[:5]):
            full = f"{prefix}[{i}]"
            keys.extend(flatten_json_keys(value, full))
    return keys


def path_score_for_canonical(path: Path, text: str) -> tuple[bool, list[str]]:
    blob = f"{path}\n{text}".lower()
    hits = [marker for marker in CANONICAL_MARKERS if marker.lower() in blob]
    return bool(hits), hits


def path_bad_model_hits(path: Path, text: str) -> list[str]:
    blob = f"{path}\n{text}".lower()
    return [marker for marker in BAD_MODEL_MARKERS if marker.lower() in blob]


def variant_hits(path: Path, text: str) -> dict[str, list[str]]:
    blob = f"{path}\n{text}".lower()
    out: dict[str, list[str]] = {}
    for variant, markers in REQUIRED_VARIANT_MARKERS.items():
        hits = [marker for marker in markers if marker.lower() in blob]
        out[variant] = hits
    return out


def leakage_hits_from_keys(keys: list[str]) -> list[str]:
    hits: list[str] = []
    lower_keys = [key.lower() for key in keys]

    for marker in LEAKAGE_RISK_MARKERS:
        for key in lower_keys:
            if marker in key:
                hits.append(key)
    return sorted(set(hits))


def extract_metrics_from_json(obj: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    target_keys = [
        "auc",
        "roc_auc",
        "pr_auc",
        "average_precision",
        "cer_at_coverage",
        "coverage",
        "risk",
        "cer",
        "wer",
        "exact",
    ]

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).lower()
                full = f"{prefix}.{key}" if prefix else str(key)
                if any(target in key_lower for target in target_keys):
                    if isinstance(child, (int, float, str, list, dict)):
                        metrics[full] = child
                walk(child, full)
        elif isinstance(value, list):
            for i, child in enumerate(value[:10]):
                walk(child, f"{prefix}[{i}]")

    walk(obj)
    return metrics


def discover_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        lower = str(path).lower()
        if any(token in lower for token in ["selective", "coverage", "confidence", "risk"]):
            if path.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
                candidates.append(path)

    return sorted(candidates)


def inspect_file(path: Path) -> dict[str, Any]:
    text = text_from_file(path)
    keys: list[str] = []
    metrics: dict[str, Any] = {}

    if path.suffix.lower() == ".json":
        try:
            obj = read_json(path)
            keys = flatten_json_keys(obj)
            metrics = extract_metrics_from_json(obj)
        except Exception as exc:
            keys = [f"JSON_READ_ERROR: {repr(exc)}"]

    elif path.suffix.lower() == ".csv":
        try:
            rows = read_csv_rows(path, max_rows=5)
            if rows:
                keys = list(rows[0].keys())
            else:
                keys = []
        except Exception as exc:
            keys = [f"CSV_READ_ERROR: {repr(exc)}"]

    else:
        keys = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)))[:200]

    canonical, canonical_hits = path_score_for_canonical(path, text)
    bad_hits = path_bad_model_hits(path, text)
    variants = variant_hits(path, text)
    leakage_hits = leakage_hits_from_keys(keys)

    return {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "canonical_path_or_text": canonical,
        "canonical_hits": canonical_hits,
        "bad_model_hits": bad_hits,
        "variant_hits": variants,
        "leakage_risk_key_hits": leakage_hits,
        "keys_sample": keys[:80],
        "metrics_sample": metrics,
    }


def summarize(inspections: list[dict[str, Any]]) -> dict[str, Any]:
    if not inspections:
        return {
            "verdict": "FAIL",
            "reason": "No selective/confidence/coverage/risk artifacts found.",
        }

    canonical_files = [item for item in inspections if item["canonical_path_or_text"]]
    bad_files = [item for item in inspections if item["bad_model_hits"]]
    leakage_files = [item for item in inspections if item["leakage_risk_key_hits"]]

    variant_coverage = {
        variant: any(item["variant_hits"].get(variant) for item in inspections)
        for variant in REQUIRED_VARIANT_MARKERS
    }

    has_all_variants = all(variant_coverage.values())

    if canonical_files and has_all_variants and not leakage_files:
        verdict = "PASS"
        reason = (
            "Selective prediction artifacts appear to reference canonical +10k context, "
            "contain required variant evidence, and show no obvious leakage-risk feature keys."
        )
    elif canonical_files and has_all_variants and leakage_files:
        verdict = "WEAK_PASS_LEAKAGE_REVIEW"
        reason = (
            "Selective prediction artifacts reference canonical +10k and required variants, "
            "but leakage-risk keys were detected. Manual review required."
        )
    elif canonical_files and not has_all_variants:
        verdict = "WEAK_PASS_VARIANTS_INCOMPLETE"
        reason = (
            "Canonical +10k evidence found, but not all required selective variants "
            "were detected."
        )
    else:
        verdict = "FAIL_OR_EXPLORATORY_ONLY"
        reason = (
            "Could not establish that selective prediction is based on canonical +10k context."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "files_n": len(inspections),
        "canonical_files_n": len(canonical_files),
        "bad_model_files_n": len(bad_files),
        "leakage_review_files_n": len(leakage_files),
        "variant_coverage": variant_coverage,
        "canonical_files": [item["path"] for item in canonical_files[:20]],
        "bad_model_files": [
            {"path": item["path"], "hits": item["bad_model_hits"]}
            for item in bad_files[:20]
        ],
        "leakage_review_files": [
            {"path": item["path"], "hits": item["leakage_risk_key_hits"]}
            for item in leakage_files[:20]
        ],
    }


def write_markdown(summary: dict[str, Any], inspections: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Selective prediction canonical check v1\n")
    lines.append(f"Verdict: **{summary['verdict']}**\n")
    lines.append(f"{summary['reason']}\n")

    lines.append("## Variant coverage\n")
    for key, value in summary.get("variant_coverage", {}).items():
        lines.append(f"- {key}: `{value}`")

    lines.append("\n## Canonical files\n")
    for item_path in summary.get("canonical_files", []):
        lines.append(f"- `{item_path}`")

    lines.append("\n## Bad-model / exploratory references\n")
    bad = summary.get("bad_model_files", [])
    if bad:
        for item in bad:
            lines.append(f"- `{item['path']}` hits={item['hits']}")
    else:
        lines.append("None detected.")

    lines.append("\n## Leakage review files\n")
    leak = summary.get("leakage_review_files", [])
    if leak:
        for item in leak:
            lines.append(f"- `{item['path']}` hits={item['hits']}")
    else:
        lines.append("None detected.")

    lines.append("\n## Inspected files\n")
    lines.append("| path | canonical | bad hits | leakage hits |")
    lines.append("|---|---:|---|---|")
    for item in inspections:
        lines.append(
            f"| `{item['path']}` "
            f"| {item['canonical_path_or_text']} "
            f"| `{item['bad_model_hits']}` "
            f"| `{item['leakage_risk_key_hits']}` |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "outputs",
            "outputs/final_result_package_v1",
        ],
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/final_result_package_v1",
    )
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for root in roots:
        files.extend(discover_files(root))

    files = sorted(set(files))

    inspections = [inspect_file(path) for path in files]
    summary = summarize(inspections)

    report = {
        "summary": summary,
        "inspections": inspections,
    }

    json_path = out_dir / "selective_prediction_canonical_check.json"
    md_path = out_dir / "selective_prediction_canonical_check.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, inspections, md_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", json_path)
    print("wrote:", md_path)


if __name__ == "__main__":
    main()
