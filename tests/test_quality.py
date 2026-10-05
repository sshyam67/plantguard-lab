from PIL import Image
from src.quality import assess_image


def test_quality_returns_bounded_score():
    result = assess_image(Image.new("RGB", (256, 256), "green"))
    assert 0 <= result.score <= 1
    assert result.width == 256 and result.height == 256


def test_small_image_warns():
    assert "Low resolution" in assess_image(Image.new("RGB", (64, 64), "white")).warnings

