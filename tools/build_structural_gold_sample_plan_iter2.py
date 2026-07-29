from __future__ import annotations

import argparse
import base64
import html
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


STRATA_TARGETS = {
    "clean_core_correct": 40,
    "hard_real_correct": 40,
    "hard_real_error": 40,
    "high_confidence_error": 30,
    "rejected_correct_low_confidence": 30,
    "numeric_mixed_rare_format": 20,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    names = row.get("graph_feature_names") or []
    values = row.get("graph_features") or []
    return {
        str(name): float(value)
        for name, value in zip(names, values)
    }


def token_type(text: str) -> str:
    chars = [ch for ch in text.strip() if not ch.isspace()]
    if not chars:
        return "empty"
    alpha = [ch for ch in chars if ch.isalpha()]
    numeric = [ch for ch in chars if ch.isdigit()]
    punct = [
        ch for ch in chars
        if unicodedata.category(ch).startswith("P")
        or unicodedata.category(ch).startswith("S")
    ]
    if len(alpha) == len(chars):
        return "alpha"
    if len(numeric) == len(chars):
        return "numeric"
    if len(punct) == len(chars):
        return "punctuation"
    return "mixed"


def image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{encoded}"


def load_school_quality(quality_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for bucket in ["clean_core", "hard_real", "invalid_or_review"]:
        path = quality_root / f"test.{bucket}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            out[str(row["sample_id"])] = {
                "bucket": bucket,
                "reasons": row.get("iter2_quality_reasons", []),
                "diagnostics": row.get("school_foreground_diagnostics", {}),
            }
    return out


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("dataset", "")),
        str(row.get("sample_id", "")),
    )


def choose(
    rows: list[dict[str, Any]],
    *,
    n: int,
    rng: random.Random,
    selected: set[str],
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if str(row["sample_id"]) not in selected
    ]
    candidates = sorted(candidates, key=rank_key)
    rng.shuffle(candidates)
    chosen = candidates[:n]
    selected.update(str(row["sample_id"]) for row in chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quality_root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--confidence_rows", required=True)
    parser.add_argument("--accepted_errors", required=True)
    parser.add_argument("--rejected_correct", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        str(row["sample_id"]): row
        for row in read_jsonl(Path(args.manifest))
    }
    quality = load_school_quality(Path(args.quality_root))
    predictions = {
        str(row["sample_id"]): row
        for row in read_jsonl(Path(args.predictions))
    }
    confidence = {
        str(row["sample_id"]): row
        for row in read_jsonl(Path(args.confidence_rows))
    }
    accepted_errors = read_jsonl(Path(args.accepted_errors))
    rejected_correct = read_jsonl(Path(args.rejected_correct))

    enriched = []
    for sample_id, row in manifest.items():
        if sample_id not in predictions:
            continue
        pred = predictions[sample_id]
        conf = confidence.get(sample_id, {})
        q = quality.get(sample_id, {})
        features = feature_dict(row)
        diagnostics = q.get("diagnostics", {})
        for key, value in diagnostics.items():
            try:
                features[key] = float(value)
            except Exception:
                pass

        text = str(row.get("text") or row.get("normalized_transcription") or pred.get("target", ""))
        enriched.append({
            "sample_id": sample_id,
            "dataset": row.get("dataset"),
            "image_path": row.get("image_path"),
            "text": text,
            "target": pred.get("target", text),
            "pred": pred.get("pred", ""),
            "cer": float(pred.get("cer", 0.0)),
            "wer": float(pred.get("wer", 0.0)),
            "exact": float(pred.get("exact", 0.0)),
            "risk": float(conf.get("risk", 0.0)),
            "token_type": token_type(text),
            "text_len": len(text),
            "school_quality_bucket": q.get("bucket", ""),
            "school_quality_reasons": q.get("reasons", []),
            "fg_fraction": features.get("fg_fraction"),
            "skel_fraction": features.get("skel_fraction"),
            "cc_count": features.get("cc_count"),
            "dir_h_frac": features.get("dir_h_frac"),
            "stroke_width_mean": features.get("stroke_width_mean"),
            "ruling_response_mean": features.get("ruling_response_mean"),
            "ruling_response_p95": features.get("ruling_response_p95"),
        })

    by_id = {
        str(row["sample_id"]): row
        for row in enriched
    }

    selected: set[str] = set()
    strata: dict[str, list[dict[str, Any]]] = {}

    school = [
        row for row in enriched
        if row["dataset"] == "school_notebooks_clean"
    ]

    strata["clean_core_correct"] = choose(
        [
            row for row in school
            if row["school_quality_bucket"] == "clean_core"
            and row["exact"] >= 1.0
        ],
        n=STRATA_TARGETS["clean_core_correct"],
        rng=rng,
        selected=selected,
    )
    strata["hard_real_correct"] = choose(
        [
            row for row in school
            if row["school_quality_bucket"] == "hard_real"
            and row["exact"] >= 1.0
        ],
        n=STRATA_TARGETS["hard_real_correct"],
        rng=rng,
        selected=selected,
    )
    strata["hard_real_error"] = choose(
        [
            row for row in school
            if row["school_quality_bucket"] == "hard_real"
            and row["exact"] < 1.0
        ],
        n=STRATA_TARGETS["hard_real_error"],
        rng=rng,
        selected=selected,
    )

    strata["high_confidence_error"] = choose(
        [
            by_id[str(row["sample_id"])]
            for row in accepted_errors
            if str(row["sample_id"]) in by_id
        ],
        n=STRATA_TARGETS["high_confidence_error"],
        rng=rng,
        selected=selected,
    )
    strata["rejected_correct_low_confidence"] = choose(
        [
            by_id[str(row["sample_id"])]
            for row in rejected_correct
            if str(row["sample_id"]) in by_id
        ],
        n=STRATA_TARGETS["rejected_correct_low_confidence"],
        rng=rng,
        selected=selected,
    )
    strata["numeric_mixed_rare_format"] = choose(
        sorted(
            [
                row for row in enriched
                if row["token_type"] in {"numeric", "mixed", "punctuation"}
            ],
            key=lambda row: (
                row["token_type"] != "numeric",
                -float(row.get("risk", 0.0)),
                str(row["sample_id"]),
            ),
        ),
        n=STRATA_TARGETS["numeric_mixed_rare_format"],
        rng=random.Random(args.seed + 1000),
        selected=selected,
    )

    sample_rows = []
    for stratum, rows in strata.items():
        for row in rows:
            out = dict(row)
            out["gold_stratum"] = stratum
            sample_rows.append(out)

    sample_rows = sorted(
        sample_rows,
        key=lambda row: (
            list(STRATA_TARGETS).index(row["gold_stratum"]),
            row["dataset"],
            row["sample_id"],
        ),
    )

    write_jsonl(sample_rows, out_root / "sample_manifest.jsonl")

    plan = {
        "seed": args.seed,
        "target_total": sum(STRATA_TARGETS.values()),
        "actual_total": len(sample_rows),
        "targets": STRATA_TARGETS,
        "actual_counts": Counter(row["gold_stratum"] for row in sample_rows),
        "dataset_counts": Counter(row["dataset"] for row in sample_rows),
        "token_type_counts": Counter(row["token_type"] for row in sample_rows),
        "notes": [
            "Samples are selected from test split for structural gold annotation.",
            "Predictions/risk use plus_10k_context confidence_graph artifacts.",
            "Duplicate sample_ids are removed across strata by priority order.",
        ],
    }
    (out_root / "sample_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cards = []
    csv_lines = [
        "sample_id,gold_stratum,foreground_usable,skeleton_matches_ink,line_residual,neighbor_noise,label_issue,topology_issue,notes"
    ]
    for i, row in enumerate(sample_rows, start=1):
        img_path = Path(str(row["image_path"]))
        uri = image_data_uri(img_path) if img_path.exists() else ""
        sample_id = html.escape(str(row["sample_id"]))
        stratum = html.escape(str(row["gold_stratum"]))
        target = html.escape(str(row["target"]))
        pred = html.escape(str(row["pred"]))
        reasons = html.escape(";".join(row.get("school_quality_reasons") or []))
        cards.append(f"""
<section class="card">
  <div class="idx">#{i}</div>
  <div class="meta">
    <b>{sample_id}</b> <span>{stratum}</span><br>
    dataset={html.escape(str(row['dataset']))} |
    quality={html.escape(str(row.get('school_quality_bucket', '')))} |
    token={html.escape(str(row['token_type']))} |
    len={row['text_len']} |
    risk={float(row['risk']):.4f} |
    CER={float(row['cer']):.3f} |
    exact={float(row['exact']):.0f}
  </div>
  <div class="textline">
    target: <b>{target}</b><br>
    pred: <b>{pred}</b>
  </div>
  <img src="{uri}" alt="{sample_id}">
  <div class="features">
    fg={row.get('fg_fraction')} |
    skel={row.get('skel_fraction')} |
    cc={row.get('cc_count')} |
    dir_h={row.get('dir_h_frac')} |
    stroke={row.get('stroke_width_mean')} |
    ruling_mean={row.get('ruling_response_mean')} |
    reasons={reasons}
  </div>
</section>
""")
        csv_lines.append(f"{sample_id},{stratum},,,,,,,")

    browser = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Structural Gold v1 Browser</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 24px;
  background: #f5f5f5;
  color: #222;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 16px;
}}
.idx {{
  float: right;
  color: #777;
  font-size: 13px;
}}
.meta {{
  font-size: 13px;
  line-height: 1.45;
}}
.meta span {{
  display: inline-block;
  margin-left: 8px;
  padding: 2px 6px;
  border-radius: 4px;
  background: #eceff3;
}}
.textline {{
  margin: 10px 0;
  font-size: 16px;
}}
img {{
  max-width: 100%;
  background: white;
  border: 1px solid #ccc;
  image-rendering: auto;
}}
.features {{
  margin-top: 8px;
  font-size: 12px;
  color: #555;
}}
</style>
</head>
<body>
<h1>Structural Gold v1 Browser</h1>
<p>n={len(sample_rows)}</p>
{''.join(cards)}
</body>
</html>
"""
    (out_root / "browser.html").write_text(browser, encoding="utf-8")
    (out_root / "annotations_template.csv").write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
