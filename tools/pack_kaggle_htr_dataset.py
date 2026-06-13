from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


MANIFEST_FILES = [
    "train.jsonl",
    "val.jsonl",
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


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    raise ValueError(f"Unknown mode: {mode}")


def rewrite_rows_and_collect_images(
    *,
    rows: list[dict[str, Any]],
    project_root: Path,
    out_root: Path,
    copy_mode: str,
) -> list[dict[str, Any]]:
    rewritten = []

    for r in rows:
        old_path = Path(r["image_path"])
        if not old_path.is_absolute():
            old_path = project_root / old_path

        if not old_path.exists():
            raise FileNotFoundError(old_path)

        dataset = r.get("dataset") or r.get("source_dataset") or "unknown_dataset"
        sample_id = r.get("sample_id") or old_path.stem
        ext = old_path.suffix.lower() or ".png"

        rel_img = Path("images") / dataset / f"{sample_id}{ext}"
        new_path = out_root / rel_img

        copy_or_link(old_path, new_path, copy_mode)

        nr = dict(r)
        nr["image_path"] = str(rel_img)
        nr["kaggle_relative_image_path"] = str(rel_img)
        nr["original_local_image_path"] = str(old_path)
        rewritten.append(nr)

    return rewritten


def collect_manifest_paths(mixed_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    for name in MANIFEST_FILES:
        src = mixed_dir / name
        dst = Path(name)
        if src.exists():
            pairs.append((src, dst))

    eval_dir = mixed_dir / "eval_manifests"
    if eval_dir.exists():
        for src in sorted(eval_dir.glob("*.jsonl")):
            dst = Path("eval_manifests") / src.name
            pairs.append((src, dst))

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--mixed_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--copy_mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    mixed_dir = Path(args.mixed_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy code.
    shutil.copytree(project_root / "src", out_dir / "code" / "src")
    shutil.copytree(project_root / "tools", out_dir / "code" / "tools")

    for optional in ["pyproject.toml", "requirements.txt", "setup.py"]:
        p = project_root / optional
        if p.exists():
            shutil.copy2(p, out_dir / "code" / optional)

    # Copy manifests and referenced images.
    manifest_pairs = collect_manifest_paths(mixed_dir)
    manifest_stats = {}

    for src_manifest, dst_rel in manifest_pairs:
        rows = read_jsonl(src_manifest)
        rewritten = rewrite_rows_and_collect_images(
            rows=rows,
            project_root=project_root,
            out_root=out_dir,
            copy_mode=args.copy_mode,
        )
        write_jsonl(rewritten, out_dir / "data" / dst_rel)
        manifest_stats[str(dst_rel)] = len(rewritten)

    # Copy vocab and summary.
    shutil.copy2(mixed_dir / "vocab.json", out_dir / "data" / "vocab.json")

    if (mixed_dir / "summary.json").exists():
        shutil.copy2(mixed_dir / "summary.json", out_dir / "data" / "source_summary.json")

    summary = {
        "mixed_dir": str(mixed_dir),
        "out_dir": str(out_dir),
        "copy_mode": args.copy_mode,
        "manifest_stats": manifest_stats,
    }

    (out_dir / "PACK_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
