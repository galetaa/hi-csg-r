from __future__ import annotations

from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET


ROOT = Path("data/interim/iam")


def inspect_files() -> None:
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    print("total files:", len(files))
    print("extensions:", Counter(p.suffix.lower() or "<no_ext>" for p in files))

    print("\nASCII files:")
    for p in sorted((ROOT / "ascii").rglob("*")):
        if p.is_file():
            print(" ", p, p.stat().st_size)

    print("\nXML candidates:")
    xmls = sorted(ROOT.rglob("*.xml"))
    print("num xml:", len(xmls))
    for p in xmls[:20]:
        print(" ", p)

    print("\nLine image candidates:")
    imgs = sorted((ROOT / "lines").rglob("*.png"))
    print("num line images:", len(imgs))
    for p in imgs[:10]:
        print(" ", p)


def inspect_lines_txt() -> None:
    candidates = list((ROOT / "ascii").rglob("lines.txt"))
    print("\nlines.txt candidates:", candidates)

    for path in candidates:
        print(f"\n--- head {path} ---")
        with path.open("r", encoding="utf-8", errors="replace") as f:
            shown = 0
            for line in f:
                if line.startswith("#"):
                    continue
                print(repr(line[:300]))
                shown += 1
                if shown >= 10:
                    break


def inspect_xml() -> None:
    xmls = sorted(ROOT.rglob("*.xml"))
    if not xmls:
        print("No XML files found")
        return

    path = xmls[0]
    print(f"\n--- XML sample: {path} ---")

    tree = ET.parse(path)
    root = tree.getroot()

    print("root tag:", root.tag)
    print("root attrs:", root.attrib)

    # IAM XML usually has <handwritten-part> with <line ...>
    for elem in root.iter():
        if elem.tag.lower().endswith("line"):
            print("line tag:", elem.tag)
            print("line attrs:", elem.attrib)
            break

    line_count = sum(1 for elem in root.iter() if elem.tag.lower().endswith("line"))
    print("line count in sample XML:", line_count)


if __name__ == "__main__":
    inspect_files()
    inspect_lines_txt()
    inspect_xml()