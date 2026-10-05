"""Dependency-light image quality checks used by training and serving."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from PIL import Image, ImageFilter, ImageStat


@dataclass(frozen=True)
class QualityResult:
    width: int
    height: int
    brightness: float
    sharpness: float
    score: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def assess_image(image: Image.Image) -> QualityResult:
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    brightness = ImageStat.Stat(gray).mean[0] / 255.0
    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = min(ImageStat.Stat(edges).var[0] / 1000.0, 1.0)
    resolution = min(min(rgb.size) / 256.0, 1.0)
    exposure = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
    score = round(0.40 * resolution + 0.35 * sharpness + 0.25 * exposure, 3)
    warnings: list[str] = []
    if min(rgb.size) < 128:
        warnings.append("Low resolution")
    if sharpness < 0.08:
        warnings.append("Image may be blurred")
    if brightness < 0.15:
        warnings.append("Image is very dark")
    if brightness > 0.90:
        warnings.append("Image is overexposed")
    return QualityResult(rgb.width, rgb.height, round(brightness, 3),
                         round(sharpness, 3), score, tuple(warnings))

