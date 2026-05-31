from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path("data/interim/school_notebooks")
IMG_ROOT = ROOT / "images" / "images"

JSONS = {
    "train": ROOT / "annotations_train.json",
    "val": ROOT / "annotations_val.json",
    "test": ROOT / "annotations_test.json",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    for split, path in JSONS.items():
        data = load_json(path)

        print("\n==============================")
        print("SPLIT:", split)
        print("path:", path)

        categories = {c["id"]: c["name"] for c in data["categories"]}
        print("categories:", categories)
        print("images:", len(data["images"]))
        print("annotations:", len(data["annotations"]))

        print("\nImage samples:")
        for im in data["images"][:10]:
            p = IMG_ROOT / im["file_name"]
            print(im, "exists=", p.exists())

        ann_key_counter = Counter()
        cat_counter = Counter()
        has_seg = 0
        has_bbox = 0
        has_text_like = Counter()

        text_like_keys = {
            "text",
            "transcription",
            "label",
            "word",
            "line",
            "value",
            "utf8_string",
            "description",
        }

        for ann in data["annotations"]:
            ann_key_counter.update(ann.keys())
            cat_counter[categories.get(ann.get("category_id"), ann.get("category_id"))] += 1

            if ann.get("segmentation"):
                has_seg += 1

            if ann.get("bbox"):
                has_bbox += 1

            for k in ann.keys():
                if k.lower() in text_like_keys:
                    has_text_like[k] += 1

        print("\nAnnotation keys:")
        for k, v in ann_key_counter.most_common(50):
            print(f"  {k}: {v}")

        print("\nCategory annotation counts:")
        for k, v in cat_counter.most_common():
            print(f"  {k}: {v}")

        print("\nHas segmentation:", has_seg)
        print("Has bbox:", has_bbox)
        print("Text-like keys:", dict(has_text_like))

        print("\nAnnotation samples by category:")
        by_cat = defaultdict(list)
        for ann in data["annotations"]:
            name = categories.get(ann.get("category_id"), str(ann.get("category_id")))
            if len(by_cat[name]) < 3:
                by_cat[name].append(ann)

        for cat, anns in by_cat.items():
            print("\n--- category:", cat, "---")
            for ann in anns:
                print(json.dumps(ann, ensure_ascii=False)[:2000])

        print("\nImage size check:")
        for im in data["images"][:5]:
            p = IMG_ROOT / im["file_name"]
            if p.exists():
                with Image.open(p) as image:
                    print(p.name, image.size, image.mode)


if __name__ == "__main__":
    main()
