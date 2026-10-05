"""Controlled degradations for robustness experiments."""
from __future__ import annotations

import io
import random
from PIL import Image, ImageEnhance, ImageFilter


def degrade(image: Image.Image, kind: str, severity: int, seed: int = 42) -> Image.Image:
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    image = image.convert("RGB")
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur((0.8, 1.6, 2.8)[severity - 1]))
    if kind == "dark":
        return ImageEnhance.Brightness(image).enhance((0.75, 0.50, 0.25)[severity - 1])
    if kind == "bright":
        return ImageEnhance.Brightness(image).enhance((1.25, 1.60, 2.00)[severity - 1])
    if kind == "resolution":
        factor = (0.70, 0.45, 0.25)[severity - 1]
        small = image.resize((max(8, int(image.width * factor)), max(8, int(image.height * factor))))
        return small.resize(image.size)
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=(60, 30, 10)[severity - 1])
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    if kind == "noise":
        rng = random.Random(seed)
        amplitude = (10, 25, 45)[severity - 1]
        pixels = [
            tuple(max(0, min(255, channel + rng.randint(-amplitude, amplitude))) for channel in pixel)
            for pixel in image.getdata()
        ]
        output = Image.new("RGB", image.size)
        output.putdata(pixels)
        return output
    raise ValueError(f"Unknown degradation: {kind}")

