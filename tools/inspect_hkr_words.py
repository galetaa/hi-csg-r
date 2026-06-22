from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path("data/interim/hkr/words")
ANN = ROOT / "ann"
IMG = ROOT / "img"


def load_json(path: Path):
    for enc in ["utf-8", "utf-8-sig", "cp1251"]:
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"Cannot read json: {path}")


def main() -> None:
    anns = sorted(ANN.glob("*.json"))
    imgs = sorted(IMG.glob("*.jpg"))

    print("ann files:", len(anns))
    print("img files:", len(imgs))

    ann_stems = {p.stem for p in anns}
    img_stems = {p.stem for p in imgs}

    print("ann without img:", len(ann_stems - img_stems))
    print("img without ann:", len(img_stems - ann_stems))

    print("\nAnn filename examples:")
    for p in anns[:20]:
        print(" ", p.name)

    print("\nImage filename examples:")
    for p in imgs[:20]:
        print(" ", p.name)

    key_counter = Counter()
    type_counter = Counter()

    print("\nSample annotations:")
    for p in anns[:30]:
        data = load_json(p)
        print("\n---", p.name, "---")
        print("type:", type(data))

        if isinstance(data, dict):
            print("keys:", list(data.keys()))
            key_counter.update(data.keys())
            for k, v in data.items():
                print(" ", k, "=", repr(str(v)[:500]))
        elif isinstance(data, list):
            print("len:", len(data))
            print("first:", data[0] if data else None)
            type_counter.update(["list"])
        else:
            print(repr(data))
            type_counter.update([type(data).__name__])

    print("\nKey counts:")
    for k, v in key_counter.most_common(100):
        print(f"  {k}: {v}")

    print("\nImage size samples:")
    for p in imgs[:50]:
        try:
            with Image.open(p) as im:
                print(p.name, im.size, im.mode)
        except Exception as exc:
            print(p.name, "ERR", repr(exc))


if __name__ == "__main__":
    main()