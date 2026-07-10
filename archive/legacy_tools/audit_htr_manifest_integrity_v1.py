from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ["train", "val", "test"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_vocab(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    for key in ["idx_to_token", "itos", "idx_to_char", "characters", "chars"]:
        if key in obj:
            chars = set(obj[key])
            break
    else:
        raise ValueError(f"Unsupported vocab format in {path}: {sorted(obj.keys())}")

    chars.discard(str(obj.get("blank_token", "<blank>")))
    return chars


def row_text(row: dict[str, Any]) -> str:
    return str(
        row.get("text")
        or row.get("normalized_transcription")
        or row.get("raw_transcription")
        or ""
    )


def duplicate_ids(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("sample_id")) for row in rows)
    return {
        sample_id: count
        for sample_id, count in sorted(counts.items())
        if count > 1
    }


def split_overlap(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids_by_split = {
        split: {str(row.get("sample_id")) for row in rows}
        for split, rows in rows_by_split.items()
    }
    out: dict[str, Any] = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = sorted(ids_by_split[left] & ids_by_split[right])
            out[f"{left}_vs_{right}"] = {
                "n": len(overlap),
                "examples": overlap[:20],
            }
    return out


def oov_summary(rows: list[dict[str, Any]], vocab: set[str]) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    count = 0
    char_counts: Counter[str] = Counter()
    empty_text = 0

    for row in rows:
        text = row_text(row)
        if not text:
            empty_text += 1
        missing = sorted(set(text) - vocab)
        if not missing:
            continue
        count += 1
        char_counts.update(missing)
        if len(examples) < 20:
            examples.append({
                "sample_id": row.get("sample_id"),
                "dataset": row.get("dataset"),
                "text": text,
                "missing_characters": missing,
            })

    return {
        "oov_rows": count,
        "oov_characters": dict(sorted(char_counts.items())),
        "oov_examples": examples,
        "empty_text_rows": empty_text,
    }


def split_summary(rows: list[dict[str, Any]], vocab: set[str]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "datasets": dict(Counter(str(row.get("dataset")) for row in rows)),
        "levels": dict(Counter(str(row.get("level")) for row in rows)),
        "source_type": dict(Counter(str(row.get("source_type", "")) for row in rows)),
        "augmentation_source": dict(Counter(str(row.get("augmentation_source", "")) for row in rows)),
        "duplicates": duplicate_ids(rows),
        **oov_summary(rows, vocab),
    }


def audit_root(name: str, root: Path) -> dict[str, Any]:
    vocab_path = root / "vocab.json"
    vocab = read_vocab(vocab_path)
    rows_by_split = {
        split: read_jsonl(root / f"{split}.jsonl")
        for split in SPLITS
    }
    return {
        "name": name,
        "root": str(root),
        "vocab": str(vocab_path),
        "vocab_size": len(vocab),
        "splits": {
            split: split_summary(rows, vocab)
            for split, rows in rows_by_split.items()
        },
        "split_overlap": split_overlap(rows_by_split),
    }


def fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def build_md(result: dict[str, Any]) -> str:
    lines = [
        "# HTR Manifest Integrity Audit v1",
        "",
        "This audit checks manifest-level reproducibility risks for the baseline, natural-line context augmentation, and same-size image-only controls.",
        "",
        "| manifest | split sizes | train duplicate ids | split id overlap | OOV rows | empty text rows | train composition |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for audit in result["audits"]:
        splits = audit["splits"]
        split_sizes = ", ".join(f"{split}={splits[split]['n']}" for split in SPLITS)
        duplicate_n = len(splits["train"]["duplicates"])
        overlap_n = sum(row["n"] for row in audit["split_overlap"].values())
        oov_rows = sum(splits[split]["oov_rows"] for split in SPLITS)
        empty_rows = sum(splits[split]["empty_text_rows"] for split in SPLITS)
        composition = ", ".join(
            f"{dataset}:{count}"
            for dataset, count in sorted(splits["train"]["datasets"].items())
        )
        lines.append(
            f"| `{audit['name']}` | {split_sizes} | {duplicate_n} | {overlap_n} | "
            f"{oov_rows} | {empty_rows} | {composition} |"
        )

    lines.extend([
        "",
        "## Pass/Fail Summary",
        "",
        "| manifest | no train duplicates | no split overlap | no OOV | no empty text |",
        "|---|---:|---:|---:|---:|",
    ])
    for audit in result["audits"]:
        splits = audit["splits"]
        no_dupes = len(splits["train"]["duplicates"]) == 0
        no_overlap = all(row["n"] == 0 for row in audit["split_overlap"].values())
        no_oov = all(splits[split]["oov_rows"] == 0 for split in SPLITS)
        no_empty = all(splits[split]["empty_text_rows"] == 0 for split in SPLITS)
        lines.append(
            f"| `{audit['name']}` | {fmt_bool(no_dupes)} | {fmt_bool(no_overlap)} | "
            f"{fmt_bool(no_oov)} | {fmt_bool(no_empty)} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- A clean audit reduces the risk that the diagnostic control results are caused by sample-id leakage or vocabulary mismatches.",
        "- This is a manifest-level audit only; it does not prove that near-duplicate handwriting images or writer identities are disjoint.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_root",
        action="append",
        nargs=2,
        metavar=("NAME", "ROOT"),
        required=True,
        help="Named manifest root containing train.jsonl, val.jsonl, test.jsonl, vocab.json.",
    )
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    result = {
        "audits": [
            audit_root(name, Path(root))
            for name, root in args.manifest_root
        ],
        "limitations": [
            "Checks sample_id-level split leakage only.",
            "Does not detect visual near-duplicates, writer overlap, or page-level dependence.",
        ],
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md.write_text(build_md(result), encoding="utf-8")

    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "manifests": [audit["name"] for audit in result["audits"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
