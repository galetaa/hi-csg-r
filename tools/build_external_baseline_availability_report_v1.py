from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


OUT_DIR = Path("outputs/htr_publication_v3/external_baseline_availability_v1")

PYTHON_PACKAGES = [
    "easyocr",
    "pytesseract",
    "tesserocr",
    "kraken",
    "keras_ocr",
    "doctr",
    "paddleocr",
    "transformers",
    "torch",
    "cv2",
]

CLI_TOOLS = [
    "tesseract",
    "kraken",
    "calamari-predict",
    "paddleocr",
]

EXISTING_BASELINES = {
    "external_trocr_zero_shot_full": Path(
        "outputs/htr_publication_v3/external_trocr_zero_shot_full/summary.json"
    ),
    "external_trocr_finetuned_tri10k_base_test": Path(
        "outputs/htr_publication_v3/external_trocr_finetuned_tri10k_base_test/summary.json"
    ),
    "mixed_cyrillic_natural_full_v1": Path(
        "outputs/htr_publication_v3/strong_internal_baselines/mixed_cyrillic_natural_full_v1_tri10k_test_fixed_m04/summary.json"
    ),
    "mixed_cyrillic_balanced50k_v1": Path(
        "outputs/htr_publication_v3/strong_internal_baselines/mixed_cyrillic_balanced50k_v1_tri10k_test_fixed_m04/summary.json"
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def package_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {
            "name": name,
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "name": name,
        "available": True,
        "version": getattr(module, "__version__", None),
    }


def cli_status(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    row: dict[str, Any] = {
        "name": name,
        "available": path is not None,
        "path": path,
    }
    if path is None:
        return row
    try:
        proc = subprocess.run(
            [name, "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        row["returncode"] = proc.returncode
        row["version_output_first_line"] = proc.stdout.splitlines()[0] if proc.stdout else ""
    except Exception as exc:
        row["version_error"] = f"{type(exc).__name__}: {exc}"
    return row


def cached_hf_models() -> list[str]:
    root = Path.home() / ".cache" / "huggingface" / "hub"
    if not root.exists():
        return []
    models = []
    for path in root.glob("models--*"):
        if path.is_dir():
            models.append(path.name.removeprefix("models--").replace("--", "/"))
    return sorted(models)


def existing_baselines() -> list[dict[str, Any]]:
    rows = []
    for name, path in EXISTING_BASELINES.items():
        if not path.exists():
            rows.append({
                "name": name,
                "exists": False,
                "summary": str(path),
            })
            continue
        obj = read_json(path)
        metrics = obj["metrics"]
        rows.append({
            "name": name,
            "exists": True,
            "summary": str(path),
            "model_id": obj.get("model_id") or obj.get("checkpoint"),
            "protocol": obj.get("protocol"),
            "n": metrics.get("n"),
            "cer": metrics.get("cer"),
            "wer": metrics.get("wer"),
            "exact": metrics.get("exact"),
        })
    return rows


def build_summary() -> dict[str, Any]:
    python_packages = [package_status(name) for name in PYTHON_PACKAGES]
    cli_tools = [cli_status(name) for name in CLI_TOOLS]
    baselines = existing_baselines()
    cached = cached_hf_models()
    external_ready = [
        row for row in baselines
        if row["exists"] and row["name"].startswith("external_")
    ]
    strong_external_ready = [
        row for row in external_ready
        if row["name"] != "external_trocr_zero_shot_full"
        and row.get("cer") is not None
        and row["cer"] < 0.5
    ]
    return {
        "package": "external_baseline_availability_v1",
        "python_packages": python_packages,
        "cli_tools": cli_tools,
        "cached_huggingface_models": cached,
        "existing_baselines": baselines,
        "publication_interpretation": {
            "external_baseline_available_locally": bool(external_ready),
            "competitive_external_russian_cyrillic_baseline_available_locally": bool(strong_external_ready),
            "prepared_easyocr_eval_wrapper": "tools/evaluate_easyocr_baseline_v1.py",
            "example_easyocr_command": (
                "python tools/evaluate_easyocr_baseline_v1.py "
                "--manifest data/experiments/htr_publication_v3/page_disjoint_hkr_school_base_v1/test.jsonl "
                "--out_dir outputs/htr_publication_v3/external_easyocr_page_disjoint_test_v1"
            ),
            "current_external_boundary": (
                "Only TrOCR-base-handwritten is cached locally as an external HTR/OCR model. "
                "The completed external TrOCR zero-shot and decoder-only adaptation baselines are weak. "
                "No EasyOCR, Tesseract, Kraken, PaddleOCR, docTR, or Calamari runtime is available locally."
            ),
            "what_would_close_the_gap": [
                "Install/evaluate a suitable external Cyrillic/Russian OCR/HTR system on the same test protocol.",
                "For EasyOCR specifically, run the prepared wrapper after installing the package and model weights.",
                "Or download a relevant HuggingFace/Russian OCR checkpoint and run it on the fixed manifests.",
                "If no such model is available, report TrOCR as a negative/limited external baseline and avoid SOTA claims.",
            ],
        },
    }


def build_md(summary: dict[str, Any]) -> str:
    lines = [
        "# External Baseline Availability v1",
        "",
        "## Python Packages",
        "",
        "| package | available | version/error |",
        "|---|---:|---|",
    ]
    for row in summary["python_packages"]:
        detail = row.get("version") if row["available"] else row.get("error_type")
        lines.append(f"| `{row['name']}` | {row['available']} | {detail or ''} |")

    lines.extend([
        "",
        "## CLI Tools",
        "",
        "| tool | available | path | version |",
        "|---|---:|---|---|",
    ])
    for row in summary["cli_tools"]:
        lines.append(
            f"| `{row['name']}` | {row['available']} | "
            f"{row.get('path') or ''} | {row.get('version_output_first_line') or ''} |"
        )

    lines.extend([
        "",
        "## Cached HuggingFace Models",
        "",
    ])
    for model in summary["cached_huggingface_models"]:
        lines.append(f"- `{model}`")
    if not summary["cached_huggingface_models"]:
        lines.append("- none")

    lines.extend([
        "",
        "## Existing Baseline Results",
        "",
        "| baseline | n | CER | WER | exact | interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in summary["existing_baselines"]:
        if not row["exists"]:
            lines.append(f"| `{row['name']}` | n/a | n/a | n/a | n/a | missing |")
            continue
        interpretation = "external" if row["name"].startswith("external_") else "internal CRNN positioning baseline"
        lines.append(
            f"| `{row['name']}` | {row.get('n', 'n/a')} | {fmt(row.get('cer'))} | "
            f"{fmt(row.get('wer'))} | {fmt(row.get('exact'))} | {interpretation} |"
        )

    interp = summary["publication_interpretation"]
    lines.extend([
        "",
        "## Publication Interpretation",
        "",
        f"- external baseline available locally: {interp['external_baseline_available_locally']}",
        f"- competitive external Russian/Cyrillic baseline available locally: {interp['competitive_external_russian_cyrillic_baseline_available_locally']}",
        f"- prepared EasyOCR wrapper: `{interp['prepared_easyocr_eval_wrapper']}`",
        f"- EasyOCR command after install: `{interp['example_easyocr_command']}`",
        f"- boundary: {interp['current_external_boundary']}",
        "",
        "To close the gap:",
    ])
    for item in interp["what_would_close_the_gap"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "report.md").write_text(build_md(summary), encoding="utf-8")
    print(json.dumps({
        "out_json": str(OUT_DIR / "summary.json"),
        "out_md": str(OUT_DIR / "report.md"),
        "competitive_external_available": summary["publication_interpretation"][
            "competitive_external_russian_cyrillic_baseline_available_locally"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
