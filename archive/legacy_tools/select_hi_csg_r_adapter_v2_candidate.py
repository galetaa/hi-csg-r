from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from tools.compare_hi_csg_r_adapter_v2_results import compare, load_summary


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--candidate must use NAME=EVALUATION_DIR")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.exists():
        raise ValueError(f"Invalid candidate: {value}")
    return name, path


def candidate_record(name: str, root: Path) -> dict[str, Any]:
    decision = compare(
        load_summary(root / "correct"),
        load_summary(root / "shuffle"),
        load_summary(root / "zero"),
        stage="dev",
    )
    correct = load_summary(root / "correct")
    checkpoint = Path(correct["checkpoint"])
    return {
        "name": name,
        "evaluation_dir": str(root.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "variant": correct["variant"],
        "lambda_preservation": float(
            (
                torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=False,
                ).get("metadata")
                or {}
            )
            .get("config", {})
            .get("lambda_preservation", 0.05)
        ),
        "decision": decision,
    }


def write_selection(
    candidates: list[dict[str, Any]],
    output: Path,
    *,
    purpose: str,
) -> dict[str, Any]:
    eligible = [
        row for row in candidates if row["decision"]["status"] == "PASS"
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["decision"]["correct"]["cer"],
            -row["decision"]["correct"]["exact"],
            row["name"],
        ),
        default=None,
    )
    result = {
        "protocol": "crnn_ctc_hi_csg_r_late_correction_protocol_v2",
        "purpose": purpose,
        "created_at": datetime.now(UTC).isoformat(),
        "selection_metric": "development micro-CER",
        "holdout_used": False,
        "test_used": False,
        "status": "PASS" if selected else "STOP",
        "selected": selected,
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path = output.with_suffix(".md")
    lines = [
        "# HI-CSG-R Late Correction v2: development selection",
        "",
        f"**Status:** `{result['status']}`",
        "",
        "Выбор выполнен только по development micro-CER. Holdout и test не "
        "использовались.",
        "",
        "| Candidate | Gate | CER | WER | Exact | Correct-shuffle CER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        decision = row["decision"]
        lines.append(
            f"| {row['name']} | {decision['status']} | "
            f"{decision['correct']['cer']:.6f} | "
            f"{decision['correct']['wer']:.6f} | "
            f"{decision['correct']['exact']:.6f} | "
            f"{decision['correct_vs_shuffle_cer']:+.6f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Selected: **`{selected['name']}`**"
                if selected
                else "Ни один кандидат не прошел dev gate: **STOP**."
            ),
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def resolve_p10_config(selection: dict[str, Any], template: Path, out: Path) -> None:
    if selection["status"] != "PASS" or not selection["selected"]:
        raise ValueError("A passing p05 selection is required for p10")
    config = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    selected = selection["selected"]
    config["variant"] = selected["variant"]
    config["selected_from"] = selected["name"]
    config["selected_checkpoint_sha256"] = selected["checkpoint_sha256"]
    config["selection_artifact"] = str(
        Path(selection["_source_path"]).resolve()
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--candidate", action="append", required=True)
    select_parser.add_argument("--purpose", required=True)
    select_parser.add_argument("--out", required=True)
    p10_parser = subparsers.add_parser("resolve-p10")
    p10_parser.add_argument("--selection", required=True)
    p10_parser.add_argument("--template", required=True)
    p10_parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.command == "select":
        records = [
            candidate_record(*parse_candidate(value))
            for value in args.candidate
        ]
        result = write_selection(records, Path(args.out), purpose=args.purpose)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(2)
        return

    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["_source_path"] = str(selection_path)
    resolve_p10_config(selection, Path(args.template), Path(args.out))
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
