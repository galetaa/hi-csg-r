from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from src.htr.xaligned_hi_csg_r import read_jsonl, resolve_path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_values(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def audit_manifest(path: Path, graph_fields: list[str]) -> dict[str, Any]:
    rows = read_jsonl(path)
    ids = [str(row.get("sample_id", "")) for row in rows]
    resolved_images: list[str] = []
    missing_images: list[dict[str, str]] = []
    missing_graph_sources: list[str] = []
    graph_source_counts: Counter[str] = Counter()
    oov_ready_rows = 0

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        try:
            image = resolve_path(str(row["image_path"]), path)
            resolved_images.append(str(image))
        except Exception as error:
            missing_images.append({"sample_id": sample_id, "error": str(error)})
            continue

        graph_source = None
        for field in graph_fields:
            if row.get(field):
                try:
                    resolve_path(str(row[field]), path)
                    graph_source = field
                    break
                except FileNotFoundError:
                    continue
        if graph_source is None:
            # The v1 builder deterministically extracts current-image graphs.
            graph_source = "rebuildable_current_image"
        graph_source_counts[graph_source] += 1
        if graph_source is None:
            missing_graph_sources.append(sample_id)
        if isinstance(row.get("text"), str):
            oov_ready_rows += 1

    return {
        "path": str(path),
        "sha256": sha256(path),
        "n": len(rows),
        "duplicate_sample_ids": duplicate_values(ids),
        "duplicate_image_paths": duplicate_values(resolved_images),
        "sample_ids": ids,
        "image_paths": resolved_images,
        "missing_images": missing_images,
        "missing_graph_sources": missing_graph_sources,
        "graph_source_counts": dict(graph_source_counts),
        "rows_with_text": oov_ready_rows,
    }


def audit_checkpoint(path: Path, expected_seed: int | None) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    actual_seed = config.get("seed")
    model = checkpoint.get("model")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "epoch": checkpoint.get("epoch"),
        "seed": actual_seed,
        "expected_seed": expected_seed,
        "seed_match": expected_seed is None or int(actual_seed) == expected_seed,
        "has_model_state": isinstance(model, dict) and bool(model),
        "has_config": bool(config),
        "vocab": config.get("vocab"),
        "train_manifest": config.get("train_manifest"),
        "val_manifest": config.get("val_manifest"),
    }


def status_row(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# HI-CSG-R Adapter Input Audit v1",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for check in summary["checks"]:
        detail = json.dumps(check["detail"], ensure_ascii=False)
        if len(detail) > 180:
            detail = detail[:177] + "..."
        lines.append(f"| `{check['check']}` | **{check['status']}** | {detail} |")
    lines.extend(
        [
            "",
            "## Manifests",
            "",
            "| split | n | missing images | duplicate ids | graph sources |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for split, item in summary["manifests"].items():
        lines.append(
            f"| `{split}` | {item['n']} | {len(item['missing_images'])} | "
            f"{len(item['duplicate_sample_ids'])} | "
            f"`{json.dumps(item['graph_source_counts'], ensure_ascii=False)}` |"
        )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            "| seed | epoch | seed match | model state | SHA256 |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for item in summary["checkpoints"]:
        lines.append(
            f"| {item['seed']} | {item['epoch']} | {item['seed_match']} | "
            f"{item['has_model_state']} | `{item['sha256']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--checkpoint_seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--vocab", required=True)
    parser.add_argument(
        "--graph_fields",
        nargs="+",
        default=["xaligned_graph_npz", "graph_json", "local_graph_npz"],
    )
    parser.add_argument("--out_dir", default="outputs/htr_adapter_v1/input_audit")
    args = parser.parse_args()

    manifests = {
        "train": audit_manifest(Path(args.train_manifest), args.graph_fields),
        "val": audit_manifest(Path(args.val_manifest), args.graph_fields),
        "test": audit_manifest(Path(args.test_manifest), args.graph_fields),
    }
    checkpoints = [
        audit_checkpoint(
            Path(path),
            args.checkpoint_seeds[index] if index < len(args.checkpoint_seeds) else None,
        )
        for index, path in enumerate(args.checkpoints)
    ]
    vocab = Path(args.vocab)
    vocab_hash = sha256(vocab)
    checkpoint_vocab_hashes: list[str] = []
    for checkpoint in checkpoints:
        value = checkpoint.get("vocab")
        if value and Path(value).exists():
            checkpoint_vocab_hashes.append(sha256(value))

    id_sets = {split: set(item["sample_ids"]) for split, item in manifests.items()}
    path_sets = {split: set(item["image_paths"]) for split, item in manifests.items()}
    checks = [
        status_row(
            "missing_images",
            all(not item["missing_images"] for item in manifests.values()),
            {split: len(item["missing_images"]) for split, item in manifests.items()},
        ),
        status_row(
            "missing_graphs",
            all(not item["missing_graph_sources"] for item in manifests.values()),
            {split: len(item["missing_graph_sources"]) for split, item in manifests.items()},
        ),
        status_row(
            "duplicate_sample_id",
            all(not item["duplicate_sample_ids"] for item in manifests.values()),
            {split: len(item["duplicate_sample_ids"]) for split, item in manifests.items()},
        ),
        status_row(
            "train_val_test_sample_overlap",
            not (id_sets["train"] & id_sets["val"])
            and not (id_sets["train"] & id_sets["test"])
            and not (id_sets["val"] & id_sets["test"]),
            {
                "train_val": len(id_sets["train"] & id_sets["val"]),
                "train_test": len(id_sets["train"] & id_sets["test"]),
                "val_test": len(id_sets["val"] & id_sets["test"]),
            },
        ),
        status_row(
            "train_val_test_path_overlap",
            not (path_sets["train"] & path_sets["val"])
            and not (path_sets["train"] & path_sets["test"])
            and not (path_sets["val"] & path_sets["test"]),
            {
                "train_val": len(path_sets["train"] & path_sets["val"]),
                "train_test": len(path_sets["train"] & path_sets["test"]),
                "val_test": len(path_sets["val"] & path_sets["test"]),
            },
        ),
        status_row(
            "vocab_mismatch",
            bool(checkpoint_vocab_hashes)
            and all(value == vocab_hash for value in checkpoint_vocab_hashes),
            {"expected": vocab_hash, "checkpoint_vocab_hashes": checkpoint_vocab_hashes},
        ),
        status_row(
            "checkpoint_seed_mismatch",
            all(item["seed_match"] for item in checkpoints),
            {str(item["expected_seed"]): item["seed"] for item in checkpoints},
        ),
        status_row(
            "checkpoint_metadata",
            all(item["has_model_state"] and item["has_config"] for item in checkpoints),
            {
                str(item["seed"]): {
                    "epoch": item["epoch"],
                    "has_model_state": item["has_model_state"],
                    "has_config": item["has_config"],
                }
                for item in checkpoints
            },
        ),
    ]
    summary = {
        "status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        "checks": checks,
        "manifests": manifests,
        "checkpoints": checkpoints,
        "vocab": {"path": str(vocab), "sha256": vocab_hash},
    }
    for item in summary["manifests"].values():
        item.pop("sample_ids", None)
        item.pop("image_paths", None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_dir": str(out_dir)}, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
