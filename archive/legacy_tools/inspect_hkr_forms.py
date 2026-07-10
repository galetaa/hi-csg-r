from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path("data/interim/hkr/forms")


def read_text_preview(path: Path, limit: int = 2000) -> str:
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1251",
        "cp866",
        "utf-16",
        "utf-16le",
        "latin1",
    ]

    data = path.read_bytes()

    for enc in encodings:
        try:
            return data.decode(enc)[:limit]
        except UnicodeDecodeError:
            continue

    return "<cannot read>"


def inspect_structure() -> None:
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    dirs = [p for p in ROOT.rglob("*") if p.is_dir()]

    print("ROOT:", ROOT)
    print("dirs:", len(dirs))
    print("files:", len(files))
    print("extensions:", Counter(p.suffix.lower() or "<no_ext>" for p in files))

    print("\nTop-level dirs:")
    for p in sorted([d for d in ROOT.iterdir() if d.is_dir()]):
        print(" ", p.name, "files:", len([x for x in p.iterdir() if x.is_file()]))

    print("\nTop-level files:")
    for p in sorted([x for x in ROOT.iterdir() if x.is_file()]):
        print(" ", p.name, p.stat().st_size)


def inspect_text_files() -> None:
    txts = sorted(ROOT.rglob("*.txt"))
    inis = sorted(ROOT.rglob("*.ini"))

    print("\nTXT files:", len(txts))
    for p in txts:
        print("\n--- TXT", p, "---")
        print(read_text_preview(p, limit=3000))

    print("\nINI files:", len(inis))
    for p in inis:
        print("\n--- INI", p, "---")
        print(read_text_preview(p, limit=3000))


def inspect_images() -> None:
    images = [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    print("\nImages:", len(images))

    by_parent = Counter(str(p.parent.relative_to(ROOT)) for p in images)
    print("\nImages by parent:")
    for parent, count in by_parent.most_common(30):
        print(f"  {parent}: {count}")

    print("\nImage samples:")
    for p in images[:40]:
        try:
            with Image.open(p) as img:
                size = img.size
                mode = img.mode
        except Exception as exc:
            size = f"ERR {exc!r}"
            mode = "ERR"

        print(p, size, mode)


def inspect_possible_class_mapping() -> None:
    class_files = sorted(ROOT.glob("class_indexes_*.txt"))

    print("\nClass index files:", len(class_files))

    for p in class_files[:20]:
        print("\n---", p.name, "---")
        text = read_text_preview(p, limit=1500)
        lines = text.splitlines()
        print("num preview lines:", len(lines))
        for line in lines[:20]:
            print(repr(line))


if __name__ == "__main__":
    inspect_structure()
    inspect_text_files()
    inspect_images()
    inspect_possible_class_mapping()
