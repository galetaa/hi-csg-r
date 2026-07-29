from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_block(summary: dict[str, Any]) -> dict[str, Any]:
    # Image-only summaries may store the actual metrics under "metrics".
    metrics = summary.get("metrics", summary)

    return {
        "n": metrics.get("n"),
        "cer": metrics.get("cer"),
        "wer": metrics.get("wer"),
        "exact": metrics.get("exact"),
        "grouped": metrics.get("grouped", {}),
    }


def get_group(m: dict[str, Any], dataset: str) -> dict[str, Any]:
    return m.get("grouped", {}).get(dataset, {})


def fmt(x: Any) -> str:
    try:
        return f"{float(x):.5f}"
    except Exception:
        return "n/a"


def delta(new: Any, old: Any) -> str:
    try:
        return f"{float(new) - float(old):+.5f}"
    except Exception:
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_only_summary", required=True)
    parser.add_argument("--graph_v2_summary", required=True)
    parser.add_argument("--graph_v3_summary", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    image = metric_block(load(args.image_only_summary))
    v2 = metric_block(load(args.graph_v2_summary))
    v3 = metric_block(load(args.graph_v3_summary))

    result = {
        "image_only_summary": args.image_only_summary,
        "graph_v2_summary": args.graph_v2_summary,
        "graph_v3_summary": args.graph_v3_summary,
        "image_only": image,
        "graph_v2": v2,
        "graph_v3": v3,
        "deltas": {
            "v3_minus_v2_cer": None if v3["cer"] is None or v2["cer"] is None else v3["cer"] - v2["cer"],
            "v3_minus_image_cer": None if v3["cer"] is None or image["cer"] is None else v3["cer"] - image["cer"],
        },
    }

    datasets = sorted(set(image.get("grouped", {})) | set(v2.get("grouped", {})) | set(v3.get("grouped", {})))

    lines = []
    lines.append("# Graph fusion v3 school foreground comparison")
    lines.append("")
    lines.append("## 1. Overall")
    lines.append("")
    lines.append("| model | CER | WER | exact | ΔCER vs image-only | ΔCER vs graph-v2 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| `image_only` | {fmt(image['cer'])} | {fmt(image['wer'])} | {fmt(image['exact'])} | 0.00000 | n/a |"
    )
    lines.append(
        f"| `graph_v2_old_features` | {fmt(v2['cer'])} | {fmt(v2['wer'])} | {fmt(v2['exact'])} | "
        f"{delta(v2['cer'], image['cer'])} | 0.00000 |"
    )
    lines.append(
        f"| `graph_v3_school_fg_auto` | {fmt(v3['cer'])} | {fmt(v3['wer'])} | {fmt(v3['exact'])} | "
        f"{delta(v3['cer'], image['cer'])} | {delta(v3['cer'], v2['cer'])} |"
    )

    lines.append("")
    lines.append("## 2. By dataset")
    lines.append("")
    lines.append("| dataset | image CER | graph-v2 CER | graph-v3 CER | v3-v2 CER | v3-image CER |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for ds in datasets:
        im = get_group(image, ds)
        g2 = get_group(v2, ds)
        g3 = get_group(v3, ds)

        lines.append(
            f"| `{ds}` | {fmt(im.get('cer'))} | {fmt(g2.get('cer'))} | {fmt(g3.get('cer'))} | "
            f"{delta(g3.get('cer'), g2.get('cer'))} | {delta(g3.get('cer'), im.get('cer'))} |"
        )

    lines.append("")
    lines.append("## 3. Strict interpretation")
    lines.append("")

    try:
        v3_cer = float(v3["cer"])
        v2_cer = float(v2["cer"])
        image_cer = float(image["cer"])

        if v3_cer < v2_cer:
            lines.append("Graph fusion v3 improves over graph fusion v2 in absolute CER.")
        else:
            lines.append("Graph fusion v3 does not improve over graph fusion v2 in absolute CER.")

        if v3_cer < image_cer:
            lines.append("Graph fusion v3 beats image-only in absolute CER.")
        else:
            lines.append("Graph fusion v3 still does not beat image-only in absolute CER.")
    except Exception:
        lines.append("Could not compute strict CER interpretation.")

    lines.append("")
    lines.append(
        "Even if v3 improves graph-fusion CER, this should be interpreted as a controlled preprocessing-pipeline improvement, "
        "not as evidence that graph fusion architecture itself is solved."
    )

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", out_md)
    print("wrote:", out_json)


if __name__ == "__main__":
    main()