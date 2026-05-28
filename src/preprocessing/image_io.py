from __future__ import annotations

from pathlib import Path
from PIL import Image


def load_image_as_rgb(path: str | Path) -> Image.Image:
    """
    Корректно загружает PNG/JPG/TIFF, включая RGBA.

    Если есть alpha-channel, изображение композитится на белый фон.
    """
    path = Path(path)

    with Image.open(path) as img:
        img.load()

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
            return img.convert("RGB")

        if img.mode == "LA":
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
            return img.convert("RGB")

        return img.convert("RGB")


def load_image_as_grayscale(path: str | Path) -> Image.Image:
    return load_image_as_rgb(path).convert("L")