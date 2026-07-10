from pathlib import Path
from collections import Counter, defaultdict
import json

RAW_ROOT = Path("data/raw")
MAX_EXAMPLES_PER_EXT = 10

def inspect_dataset(dataset_dir: Path) -> dict:
    files = [p for p in dataset_dir.rglob("*") if p.is_file()]
    dirs = [p for p in dataset_dir.rglob("*") if p.is_dir()]

    ext_counter = Counter(p.suffix.lower() or "<no_ext>" for p in files)
    examples = defaultdict(list)

    for p in files:
        ext = p.suffix.lower() or "<no_ext>"
        if len(examples[ext]) < MAX_EXAMPLES_PER_EXT:
            examples[ext].append(str(p))

    return {
        "dataset": dataset_dir.name,
        "path": str(dataset_dir),
        "num_files": len(files),
        "num_dirs": len(dirs),
        "extensions": dict(ext_counter.most_common()),
        "examples_by_extension": dict(examples),
    }

def main():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw root not found: {RAW_ROOT}")

    report = {}
    for dataset_dir in sorted([p for p in RAW_ROOT.iterdir() if p.is_dir()]):
        report[dataset_dir.name] = inspect_dataset(dataset_dir)

    out_path = Path("data/reports/raw_inventory.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()