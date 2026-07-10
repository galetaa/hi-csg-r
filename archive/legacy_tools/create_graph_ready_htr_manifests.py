from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


DATASETS = [
    "cyrillic_handwriting",
    "hkr_words",
    "school_notebooks_clean",
]

GRAPH_FEATURE_KEYS = [
    "width",
    "height",
    "aspect_ratio",
    "text_len",
    "fg_fraction",
    "bbox_x0_frac",
    "bbox_y0_frac",
    "bbox_w_frac",
    "bbox_h_frac",
    "bbox_area_frac",
    "cc_count",
    "cc_area_mean",
    "cc_area_median",
    "cc_area_max_frac",
    "skel_pixels",
    "skel_fraction",
    "skel_components",
    "graph_nodes",
    "graph_edges_8n",
    "graph_avg_degree",
    "graph_endpoint_count",
    "graph_branchpoint_count",
    "graph_isolated_count",
    "endpoint_per_100_skel",
    "branchpoint_per_100_skel",
    "degree_hist_0",
    "degree_hist_1",
    "degree_hist_2",
    "degree_hist_3",
    "degree_hist_4",
    "degree_hist_5plus",
    "dir_h_frac",
    "dir_v_frac",
    "dir_diag_down_frac",
    "dir_diag_up_frac",
    "stroke_width_mean",
    "stroke_width_std",
    "stroke_width_p50",
    "stroke_width_p90",
    "warning_count",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_id(row: dict[str, Any], dataset: str, split: str, idx: int) -> str:
    for key in ["sample_id", "id", "uid", "record_id"]:
        if row.get(key):
            return str(row[key])

    raw = "|".join([
        dataset,
        split,
        str(idx),
        str(row.get("image_path", row.get("path", ""))),
        str(row.get("text", "")),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "label", "transcription", "target"]:
        if key in row:
            return str(row[key])
    return ""


def load_feature_map(features_path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for r in read_jsonl(features_path):
        out[str(r["sample_id"])] = r
    return out


def collect_chars(rows: list[dict[str, Any]]) -> list[str]:
    chars = set()
    for r in rows:
        for ch in str(r["text"]):
            chars.add(ch)
    return sorted(chars)


def write_vocab(chars: list[str], template_path: Path, out_path: Path) -> None:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    blank_token = (
        template.get("blank_token")
        or template.get("blank")
        or template.get("blank_symbol")
        or "<blank>"
    )

    tokens = [blank_token] + chars

    if "idx_to_token" in template and "token_to_idx" in template:
        vocab = {
            **template,
            "idx_to_token": tokens,
            "token_to_idx": {t: i for i, t in enumerate(tokens)},
            "blank_index": 0,
            "blank_token": blank_token,
            "num_classes": len(tokens),
        }
    elif "itos" in template and "stoi" in template:
        vocab = {
            **template,
            "itos": tokens,
            "stoi": {t: i for i, t in enumerate(tokens)},
            "blank_index": 0,
            "blank_token": blank_token,
            "num_classes": len(tokens),
        }
    elif "chars" in template:
        vocab = {
            **template,
            "chars": chars,
            "blank_index": 0,
            "blank_token": blank_token,
            "num_classes": len(chars) + 1,
        }
        if "char_to_idx" in template:
            vocab["char_to_idx"] = {ch: i + 1 for i, ch in enumerate(chars)}
    elif "idx_to_char" in template and "char_to_idx" in template:
        vocab = {
            **template,
            "idx_to_char": tokens,
            "char_to_idx": {t: i for i, t in enumerate(tokens)},
            "blank_index": 0,
            "blank_token": blank_token,
            "num_classes": len(tokens),
        }
    else:
        raise ValueError(f"Unknown vocab format: {sorted(template.keys())}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def enrich_rows(
    manifest_path: Path,
    features_path: Path,
    dataset: str,
    split: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    fmap = load_feature_map(features_path)

    out = []
    missing = []

    for idx, row in enumerate(rows):
        sid = stable_id(row, dataset, split, idx)
        feat = fmap.get(sid)

        if feat is None:
            missing.append(sid)
            continue

        graph_features = [float(feat[k]) for k in GRAPH_FEATURE_KEYS]

        r = dict(row)
        r["sample_id"] = sid
        r["dataset"] = dataset
        r["source_dataset"] = dataset
        r["split"] = split
        r["text"] = get_text(row)
        r["graph_features"] = graph_features
        r["graph_feature_names"] = GRAPH_FEATURE_KEYS
        r["graph_warning_count"] = int(feat.get("warning_count", 0))
        r["graph_binarization"] = feat.get("binarization", "")
        out.append(r)

    if missing:
        raise RuntimeError(
            f"Missing graph features for {dataset}/{split}: {len(missing)}. "
            f"First missing: {missing[:5]}"
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_dir", default="data/experiments/htr_graph_v1/subsets/tri10k")
    parser.add_argument("--features_dir", default="data/experiments/htr_graph_v1/features/tri10k")
    parser.add_argument("--out_dir", default="data/experiments/htr_graph_v1/graph_ready/tri10k_mixed")
    parser.add_argument("--template_vocab", default="data/experiments/htr_baseline_v1_ctc_ready/mixed_cyrillic_natural_full_v1/vocab.json")
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    combined = {"train": [], "val": [], "test": []}
    summary: dict[str, Any] = {
        "name": out_dir.name,
        "seed": args.seed,
        "datasets": {},
        "graph_feature_names": GRAPH_FEATURE_KEYS,
        "graph_feature_dim": len(GRAPH_FEATURE_KEYS),
    }

    for dataset in DATASETS:
        summary["datasets"][dataset] = {}

        for split in ["train", "val", "test"]:
            manifest = subset_dir / dataset / f"{split}.jsonl"
            features = features_dir / f"{dataset}_{split}_features.jsonl"

            rows = enrich_rows(
                manifest_path=manifest,
                features_path=features,
                dataset=dataset,
                split=split,
            )

            ds_out = out_dir / "eval_manifests" / f"{dataset}_{split}.jsonl"
            write_jsonl(rows, ds_out)

            combined[split].extend(rows)

            summary["datasets"][dataset][split] = {
                "n": len(rows),
                "path": str(ds_out),
                "warning_rows": sum(1 for r in rows if r["graph_warning_count"] > 0),
            }

    for split in ["train", "val", "test"]:
        rng.shuffle(combined[split])
        write_jsonl(combined[split], out_dir / f"{split}.jsonl")
        summary[split] = {
            "n": len(combined[split]),
            "by_dataset": {
                ds: sum(1 for r in combined[split] if r["dataset"] == ds)
                for ds in DATASETS
            },
            "path": str(out_dir / f"{split}.jsonl"),
        }

    chars = collect_chars(combined["train"] + combined["val"] + combined["test"])
    write_vocab(chars, Path(args.template_vocab), out_dir / "vocab.json")

    summary["num_chars_without_blank"] = len(chars)
    summary["num_classes_with_blank"] = len(chars) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()