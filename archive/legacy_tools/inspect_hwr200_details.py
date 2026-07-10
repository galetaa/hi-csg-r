from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


ROOT = Path("data/interim/hwr200")


def safe_json_load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"__error__": repr(exc)}


def inspect_jsons() -> None:
    jsons = [
        p for p in sorted(ROOT.rglob("*.json"))
        if ".ipynb_checkpoints" not in str(p)
    ]

    print("JSON count:", len(jsons))

    by_parent = Counter()
    key_counter = Counter()

    for p in jsons:
        rel = p.relative_to(ROOT)
        # annotations/FPR/0/fpr0.json → annotations/FPR
        parts = rel.parts
        bucket = "/".join(parts[:2]) if len(parts) >= 2 else str(rel.parent)
        by_parent[bucket] += 1

        data = safe_json_load(p)
        if isinstance(data, dict):
            key_counter.update(data.keys())
        else:
            key_counter.update([type(data).__name__])

    print("\nJSON buckets:")
    for k, v in by_parent.most_common(30):
        print(f"  {k}: {v}")

    print("\nJSON keys:")
    for k, v in key_counter.most_common(50):
        print(f"  {k}: {v}")

    print("\nSample JSONs:")
    for p in jsons[:10]:
        data = safe_json_load(p)
        print("\n---", p, "---")
        print("type:", type(data))
        if isinstance(data, dict):
            print("keys:", list(data.keys()))
            for key, value in data.items():
                if key == "sentences" and isinstance(value, list):
                    print("sentences len:", len(value))
                    print("first sentence:", value[0] if value else None)
                elif key == "full_text":
                    print("full_text:", repr(str(value)[:300]))
                elif key == "words_count":
                    print("words_count:", value)
                else:
                    print(key, "=", repr(str(value)[:300]))
        else:
            print(repr(str(data)[:1000]))


def inspect_images() -> None:
    exts = {".jpg", ".jpeg", ".png"}
    images = [
        p for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in exts
        and ".git" not in str(p)
        and ".ipynb_checkpoints" not in str(p)
    ]

    print("\nIMAGE count:", len(images))

    by_parent = Counter()
    by_ext = Counter()
    name_patterns = Counter()

    for p in images:
        rel = p.relative_to(ROOT)
        parts = rel.parts
        bucket = "/".join(parts[:3]) if len(parts) >= 3 else str(rel.parent)
        by_parent[bucket] += 1
        by_ext[p.suffix.lower()] += 1

        # грубая группировка по началу имени
        stem = p.stem
        prefix = "".join(ch for ch in stem if not ch.isdigit())
        name_patterns[prefix or "<digits_only>"] += 1

    print("\nImage extensions:", by_ext)

    print("\nImage parent buckets:")
    for k, v in by_parent.most_common(50):
        print(f"  {k}: {v}")

    print("\nImage name prefixes:")
    for k, v in name_patterns.most_common(50):
        print(f"  {k}: {v}")

    print("\nImage samples with size:")
    for p in images[:30]:
        try:
            with Image.open(p) as img:
                size = img.size
                mode = img.mode
        except Exception as exc:
            size = f"ERR {exc!r}"
            mode = "ERR"
        print(p, size, mode)


def inspect_text_files() -> None:
    txts = [
        p for p in sorted(ROOT.rglob("*.txt"))
        if ".ipynb_checkpoints" not in str(p)
    ]

    print("\nTXT count:", len(txts))
    print("TXT samples:")
    for p in txts[:20]:
        print("\n---", p, "---")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print("ERR", exc)
            continue
        print(repr(text[:500]))


def find_possible_matches() -> None:
    """
    Проверяем простую гипотезу: annotation fpr0/original0/reuse0 может соответствовать image,
    в имени которого есть тот же numeric id.
    """
    jsons = [
        p for p in sorted(ROOT.rglob("*.json"))
        if ".ipynb_checkpoints" not in str(p)
    ][:50]

    images = [
        p for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and ".git" not in str(p)
    ]

    by_stem = defaultdict(list)
    for img in images:
        by_stem[img.stem].append(img)

    print("\nPossible stem matches:")
    for jp in jsons[:30]:
        stem = jp.stem
        matches = by_stem.get(stem, [])
        print(jp.relative_to(ROOT), "->", [str(m.relative_to(ROOT)) for m in matches[:5]])


if __name__ == "__main__":
    inspect_jsons()
    inspect_images()
    inspect_text_files()
    find_possible_matches()