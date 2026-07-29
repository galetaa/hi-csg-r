from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.htr.xaligned_hi_csg_r import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    WIDTH_DOWNSAMPLE,
    build_feature_record,
    feature_record_metadata,
    load_feature_record,
    locate_graph_path,
    read_jsonl,
    resolve_path,
    save_feature_record,
    write_jsonl,
)


def _safe_filename(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__")


def _process(task: dict[str, Any]) -> dict[str, Any]:
    row = task["row"]
    manifest_path = Path(task["manifest"])
    sample_id = str(row["sample_id"])
    image_path = resolve_path(str(row["image_path"]), manifest_path)
    graph_path = locate_graph_path(
        row,
        graph_field=task["graph_field"],
        graph_root=task["graph_root"],
        manifest_path=manifest_path,
    )
    output = Path(task["out_dir"]) / f"{_safe_filename(sample_id)}.npz"
    if task["reuse"] and output.exists():
        record = load_feature_record(output)
        if (
            str(record["sample_id"]) == sample_id
            and str(record["feature_version"]) == task["feature_version"]
        ):
            metadata = feature_record_metadata(record)
            metadata.update({"path": str(output), "reused": True})
            return metadata

    record = build_feature_record(
        sample_id=sample_id,
        image_path=image_path,
        dataset=str(row.get("dataset") or row.get("source_dataset") or "unknown"),
        graph_path=graph_path,
        width_downsample=int(task["width_downsample"]),
        smooth=bool(task["smooth"]),
        feature_version=str(task["feature_version"]),
        binarization=row.get("local_graph_binarization"),
        sauvola_window=int(task["sauvola_window"]),
        min_object_size=int(task["min_object_size"]),
    )
    save_feature_record(record, output)
    metadata = feature_record_metadata(record)
    metadata.update(
        {
            "path": str(output),
            "reused": False,
            "graph_path": str(graph_path) if graph_path else None,
        }
    )
    return metadata


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest)
    rows = read_jsonl(manifest)
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if any(sample_id in {"", "None"} for sample_id in sample_ids):
        raise ValueError("Every manifest row must have sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Input manifest has duplicate sample_id values")

    tasks = [
        {
            "row": row,
            "manifest": str(manifest),
            "out_dir": str(args.out_dir),
            "graph_field": args.graph_field,
            "graph_root": args.graph_root,
            "feature_version": args.feature_version,
            "width_downsample": args.width_downsample,
            "smooth": args.smooth,
            "sauvola_window": args.sauvola_window,
            "min_object_size": args.min_object_size,
            "reuse": args.reuse,
        }
        for row in rows
    ]
    results_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    if args.workers <= 1:
        for index, task in enumerate(tasks, start=1):
            try:
                result = _process(task)
                results_by_id[str(result["sample_id"])] = result
            except Exception as error:
                failures.append(
                    {
                        "sample_id": task["row"].get("sample_id"),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if index % args.log_every == 0:
                print(f"processed {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(_process, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results_by_id[str(result["sample_id"])] = result
                except Exception as error:
                    failures.append(
                        {
                            "sample_id": task["row"].get("sample_id"),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                if index % args.log_every == 0:
                    print(f"processed {index}/{len(tasks)}", flush=True)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        result = results_by_id.get(sample_id)
        if result is None:
            continue
        output_row = dict(row)
        output_row.update(
            {
                "xaligned_graph_npz": result["path"],
                "xaligned_graph_version": args.feature_version,
                "xaligned_feature_dim": len(FEATURE_NAMES),
                "xaligned_feature_source": "graph_extracted_from_current_image",
                "xaligned_time_steps": result["time_steps"],
                "xaligned_source_image_sha1": result["source_image_sha1"],
            }
        )
        output_rows.append(output_row)

    write_jsonl(output_rows, args.out_manifest)
    domain_manifests: dict[str, str] = {}
    if args.domain_manifest_dir:
        aliases = {
            "cyrillic_handwriting": "cyrillic",
            "hkr_words": "hkr",
            "school_notebooks_clean": "school",
            "school_notebooks_line": "school",
        }
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in output_rows:
            dataset = str(row.get("dataset") or row.get("source_dataset") or "unknown")
            alias = aliases.get(dataset, dataset.replace("/", "_"))
            grouped.setdefault(alias, []).append(row)
        domain_dir = Path(args.domain_manifest_dir)
        for alias, domain_rows in sorted(grouped.items()):
            domain_path = domain_dir / f"{Path(args.out_manifest).stem}_{alias}.jsonl"
            write_jsonl(domain_rows, domain_path)
            domain_manifests[alias] = str(domain_path)
    failure_path = Path(args.out_manifest).with_suffix(".failures.jsonl")
    write_jsonl(failures, failure_path)

    summary = {
        "manifest": str(manifest),
        "out_dir": str(args.out_dir),
        "out_manifest": str(args.out_manifest),
        "feature_version": args.feature_version,
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": len(FEATURE_NAMES),
        "width_downsample": args.width_downsample,
        "smooth": args.smooth,
        "expected_n": len(rows),
        "written_n": len(output_rows),
        "failures_n": len(failures),
        "reused_n": sum(bool(result.get("reused")) for result in results_by_id.values()),
        "domain_manifests": domain_manifests,
        "time_steps": {
            "min": min((int(result["time_steps"]) for result in results_by_id.values()), default=0),
            "max": max((int(result["time_steps"]) for result in results_by_id.values()), default=0),
            "mean": (
                sum(int(result["time_steps"]) for result in results_by_id.values())
                / max(len(results_by_id), 1)
            ),
        },
        "status": "PASS" if len(output_rows) == len(rows) and not failures else "FAIL",
    }
    summary_path = Path(args.summary) if args.summary else Path(args.out_manifest).with_suffix(
        ".summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph_root", default=None)
    parser.add_argument("--graph_field", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--domain_manifest_dir", default=None)
    parser.add_argument("--feature_version", default=FEATURE_VERSION)
    parser.add_argument("--width_downsample", type=int, default=WIDTH_DOWNSAMPLE)
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sauvola_window", type=int, default=25)
    parser.add_argument("--min_object_size", type=int, default=4)
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()
    summary = build(args)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
